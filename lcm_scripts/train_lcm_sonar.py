"""
Train LCM using SONAR-like embeddings (SonarLite implemented in this repo).

Usage:
  python lcm_scripts/train_lcm_sonar.py --num_docs 1000 --epochs 3
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

import argparse
import time
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from datasets import logging as ds_logging
import logging as py_logging

from torch import amp

AmpModule = amp
AmpGradScaler = getattr(amp, "GradScaler", None)
AmpAutocast = getattr(amp, "autocast", None)

from sonar_module import SonarLite
from base_lcm import BaseLCM
from eval_metrics import compute_all
from experiment_config import setup_logging
from huggingface_hub import snapshot_download
from datasets import load_dataset
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def prepare_data(num_docs=1000, max_sent_per_doc=20, text_col="marathi", fraction=1.0):
    # deprecated: callers should pass data_path via CLI
    # load Marathi sentences from ParamTh/BhashaSetu

    # Improve visibility: set datasets / huggingface_hub logging so progress is visible
    ds_logging.set_verbosity_info()
    py_logging.getLogger("huggingface_hub").setLevel(py_logging.INFO)

    # Try local copies first (project data/ or repo root), otherwise download from Hugging Face Hub
    local_candidates = [
        os.path.join(os.path.dirname(__file__), "..", "data", "ParamTh_BhashaSetu"),
        os.path.join(os.getcwd(), "data", "ParamTh_BhashaSetu"),
        os.path.join(os.getcwd(), "ParamTh_BhashaSetu"),
    ]

    ds = None
    # If caller provided an explicit data_path via CLI, try it first
    data_path = os.environ.get("DATA_PATH_OVERRIDE")
    # callers will pass data_path argument from CLI by setting this env var before calling
    # (we set it from main below). If not set, data_path will be None and we continue.
    if data_path:
        data_path = os.path.normpath(data_path)
        if os.path.exists(data_path):
            # quick heuristic: only attempt if it looks like a dataset directory or file
            looks_like_dataset = False
            if os.path.isdir(data_path):
                if os.path.exists(os.path.join(data_path, "dataset_info.json")):
                    looks_like_dataset = True
                else:
                    for fname in os.listdir(data_path):
                        if fname.endswith(
                            (".arrow", ".parquet", ".jsonl", ".json", ".csv")
                        ):
                            looks_like_dataset = True
                            break
            else:
                if data_path.endswith(
                    (".py", ".jsonl", ".json", ".parquet", ".arrow", ".csv")
                ):
                    looks_like_dataset = True

            if looks_like_dataset:
                print(f"Loading dataset from explicit path: {data_path}")
                try:
                    # If a concrete data file (csv/json/jsonl/parquet/arrow) is provided,
                    # call load_dataset with the appropriate format and data_files.
                    if os.path.isfile(data_path):
                        lp = data_path.lower()
                        if lp.endswith(".csv"):
                            ds = load_dataset(
                                "csv",
                                data_files=data_path,
                                split="train",
                                streaming=False,
                            )
                        elif lp.endswith(".jsonl") or lp.endswith(".json"):
                            ds = load_dataset(
                                "json",
                                data_files=data_path,
                                split="train",
                                streaming=False,
                            )
                        elif lp.endswith(".parquet") or lp.endswith(".arrow"):
                            ds = load_dataset(
                                "parquet",
                                data_files=data_path,
                                split="train",
                                streaming=False,
                            )
                        elif lp.endswith(".py"):
                            # dataset script
                            ds = load_dataset(data_path, split="train", streaming=False)
                        else:
                            # fallback: let datasets try to infer
                            ds = load_dataset(data_path, split="train", streaming=False)
                    else:
                        # directory: may be a dataset snapshot or dataset script; let datasets handle it
                        ds = load_dataset(data_path, split="train", streaming=False)
                except Exception as e:
                    print(
                        f"Failed to load dataset from explicit path {data_path}: {e}. Falling back to local candidates / Hugging Face Hub."
                    )
                    ds = None
        else:
            print(f"Explicit data path does not exist: {data_path}. Ignoring.")
    for p in local_candidates:
        p = os.path.normpath(p)
        if not os.path.exists(p):
            continue
        # Heuristics to determine if this path is a dataset directory or dataset script
        looks_like_dataset = False
        if os.path.isdir(p):
            # common dataset artifacts
            if os.path.exists(os.path.join(p, "dataset_info.json")):
                looks_like_dataset = True
            else:
                # check for common data file extensions
                for fname in os.listdir(p):
                    if fname.endswith(
                        (".arrow", ".parquet", ".jsonl", ".json", ".csv")
                    ):
                        looks_like_dataset = True
                        break
        else:
            # file path: python dataset script or a single data file
            if p.endswith((".py", ".jsonl", ".json", ".parquet", ".arrow", ".csv")):
                looks_like_dataset = True

        if not looks_like_dataset:
            continue

        print(f"Loading dataset from local path: {p}")
        try:
            if os.path.isfile(p):
                lp = p.lower()
                if lp.endswith(".csv"):
                    ds = load_dataset(
                        "csv", data_files=p, split="train", streaming=False
                    )
                elif lp.endswith(".jsonl") or lp.endswith(".json"):
                    ds = load_dataset(
                        "json", data_files=p, split="train", streaming=False
                    )
                elif lp.endswith(".parquet") or lp.endswith(".arrow"):
                    ds = load_dataset(
                        "parquet", data_files=p, split="train", streaming=False
                    )
                elif lp.endswith(".py"):
                    ds = load_dataset(p, split="train", streaming=False)
                else:
                    ds = load_dataset(p, split="train", streaming=False)
            else:
                ds = load_dataset(p, split="train", streaming=False)
        except Exception as e:
            print(
                f"Failed to load dataset from local path {p}: {e}. Trying next candidate."
            )
            ds = None
        break

    if ds is None:
        hf_repo = "ParamTh/BhashaSetu"
        print(
            f"Attempting to download dataset snapshot from Hugging Face Hub: {hf_repo}"
        )
        # Attempt to download a repo snapshot to a local cache dir so progress is visible
        try:
            hf_cache = os.environ.get(
                "HF_CACHE_DIR", os.path.join(os.getcwd(), "data", "hf_cache")
            )
            os.makedirs(hf_cache, exist_ok=True)
            print(
                f"Downloading snapshot to: {hf_cache} (this may take time, progress will be logged)"
            )
            # snapshot_download returns a path to the downloaded repo snapshot
            repo_path = snapshot_download(
                repo_id=hf_repo, repo_type="dataset", cache_dir=hf_cache
            )
            print(
                f"Snapshot downloaded to: {repo_path}. Loading dataset from local snapshot..."
            )
            # load from local snapshot path (non-streaming) so dataset files are read from disk
            ds = load_dataset(repo_path, split="train", streaming=False)
        except Exception as e:
            print(
                f"snapshot_download failed or huggingface_hub unavailable: {e}. Falling back to streaming load from Hub."
            )
            try:
                print(
                    f"Loading dataset from Hugging Face Hub (non-streaming): {hf_repo}"
                )
                ds = load_dataset(hf_repo, split="train", streaming=False)
            except Exception as e2:
                print(f"Failed to load dataset from Hugging Face Hub: {e2}")
                raise
    total = len(ds)
    num_to_select = int(total * fraction)
    ds = ds.shuffle(seed=42).select(range(num_to_select))
    docs = []
    iterator = ds
    # datasets iterators are lazy; wrap with tqdm to show progress
    iterator = tqdm(ds, desc="loading dataset", unit="rows")

    for row in iterator:
        # Prefer the configured text column; if missing, try to auto-detect a suitable text field.
        text = ""
        used_col = None
        if isinstance(row, dict):
            # Try to extract and normalize the requested column first. Support
            # several possible stored types: string, list/tuple of strings or
            # other objects coercible to string.
            val = row.get(text_col, None)
            if val is not None:
                if isinstance(val, str):
                    if val.strip():
                        text = val
                        used_col = text_col
                elif isinstance(val, (list, tuple)):
                    # join string elements; ignore non-string elements
                    joined = " ".join(
                        [v for v in val if isinstance(v, str) and v.strip()]
                    )
                    if joined.strip():
                        text = joined
                        used_col = text_col
                else:
                    s = str(val)
                    if s and s.strip():
                        text = s
                        used_col = text_col

            # If requested column didn't yield usable text, apply fallback
            if not text:
                # fallback heuristics: pick a column named 'text' or the longest string field
                if "text" in row and row.get("text"):
                    text = row.get("text")
                    used_col = "text"
                else:
                    # pick the longest string-like field
                    used_col = None
                    max_len = 0
                    for k, v in row.items():
                        if isinstance(v, str) and len(v.strip()) > max_len:
                            max_len = len(v.strip())
                            used_col = k
                    if used_col:
                        text = row.get(used_col, "")
                if text and used_col and used_col != text_col:
                    # inform the user when we used a fallback column
                    print(
                        f"prepare_data: using column '{used_col}' as text source (requested '{text_col}')"
                    )
        else:
            # streaming datasets sometimes yield list/tuple rows; coerce to string
            text = str(row)
        if text and len(text.strip()) > 5:
            # simple sentence split by punctuation; better: SaT Capped in full pipeline
            sents = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
            # Keep single-sentence rows as single-element documents so the
            # higher-level grouping logic can detect sentence-level corpora and
            # form pseudo-documents. Previously single-sentence rows were
            # discarded which made auto-grouping impossible.
            if len(sents) >= 1:
                docs.append(sents[:max_sent_per_doc])
    return docs


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings_seqs):
        self.data = embeddings_seqs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        # seq is tensor [num_sent, embed_dim]
        return seq[:-1], seq[1:]


def collate(batch):
    srcs, tgts = zip(*batch)
    max_len = max(s.shape[0] for s in srcs)
    emb_dim = srcs[0].shape[1]
    B = len(srcs)
    src_p = torch.zeros(B, max_len, emb_dim)
    tgt_p = torch.zeros(B, max_len, emb_dim)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_p[i, : s.shape[0]] = s
        tgt_p[i, : t.shape[0]] = t
    return src_p, tgt_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_docs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--model_dim", type=int, default=2048)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument(
        "--checkpointing",
        action="store_true",
        help="Enable gradient checkpointing inside BaseLCM to reduce activation memory",
    )
    parser.add_argument(
        "--encode_batch_size",
        type=int,
        default=64,
        help="Number of documents to encode in a single batch (reduces GPU kernel launch overhead)",
    )
    parser.add_argument(
        "--embed_cache",
        type=str,
        default=None,
        help="Optional path to load/save precomputed embeddings (torch.save format)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader worker processes",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (simulate larger batch size)",
    )
    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard log dir")
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default="fyp-team-2513")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Explicit local dataset path (overrides automatic local/hub lookup)",
    )
    parser.add_argument(
        "--eval_hyp",
        type=str,
        default=None,
        help="Optional hypothesis file to evaluate after each epoch",
    )
    parser.add_argument(
        "--eval_ref",
        type=str,
        default=None,
        help="Optional reference file to evaluate after each epoch",
    )
    parser.add_argument(
        "--comet_model",
        type=str,
        default=None,
        help="Optional COMET model name or checkpoint path",
    )
    parser.add_argument(
        "--fraction", type=float, default=1.0, help="Fraction of dataset to use"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    writer = None
    if args.log_dir:
        writer = setup_logging(args.log_dir)
    wandb_module = None
    if args.wandb:
        from experiment_config import setup_wandb

        wandb_module = setup_wandb(
            args.log_dir or ".",
            project=args.wandb_project or "blt-lcm",
            name=args.wandb_name,
            entity=args.wandb_entity,
            config={"project": "blt-lcm"},
        )

    # Initialize lists for plotting
    train_losses = []
    step_losses = []
    grad_norms = []
    lrs = []
    eval_metrics_history = []

    print("Preparing data...")
    # If user provided a data path via CLI, pass it to prepare_data via environment
    if args.data_path:
        os.environ["DATA_PATH_OVERRIDE"] = args.data_path
    docs = prepare_data(num_docs=args.num_docs, fraction=args.fraction)

    # Fallback: some datasets (e.g. sentence-level corpora) provide one sentence
    # per row. The training pipeline expects at least one document with >=2
    # sentences. If we detect that every loaded "document" contains only a
    # single sentence, automatically group consecutive sentences into pseudo
    # documents so training can proceed. This mirrors the suggestions printed
    # later and saves the user the manual grouping step.
    if (
        (not getattr(args, "no_grouping", False))
        and len(docs) > 0
        and all(len(d) <= 1 for d in docs)
    ):
        flat = [s for d in docs for s in d if s and len(s.strip()) > 0]
        if len(flat) >= 2:
            group_size = 4  # heuristically group 4 sentences per pseudo-document
            new_docs = [
                flat[i : i + group_size] for i in range(0, len(flat), group_size)
            ]
            new_docs = new_docs[: args.num_docs]
            print(
                f"prepare_data: detected sentence-level dataset; grouped {len(flat)} sentences into {len(new_docs)} pseudo-documents (group_size={group_size})"
            )
            docs = new_docs

    print("Building SONAR-lite encoder...")
    sonar = SonarLite(device=device)
    # Ensure model parameters/buffers live on the intended device and keep the
    # internal device flag in sync (used by encode_sentences to allocate tensors).
    sonar.to(device)
    sonar.device = device

    # Estimate memory footprint for the chosen LCM configuration and batch
    def _estimate_memory(
        embed_dim, model_dim, n_layers, max_seq_len, batch_size, use_amp
    ):
        # dtype size in bytes (assume float32 weights/activations)
        dtype_size = 4

        # build a temporary model on CPU to count parameters
        try:
            tmp = BaseLCM(
                embed_dim=embed_dim,
                model_dim=model_dim,
                n_layers=n_layers,
                n_heads=args.n_heads,
                max_seq_len=max_seq_len,
            )
        except Exception:
            # fallback estimate using formula
            param_count = model_dim * model_dim * n_layers
        else:
            param_count = sum(p.numel() for p in tmp.parameters())
            # free temp model
            del tmp

        param_bytes = param_count * dtype_size

        # Optimizer (AdamW) keeps exp_avg and exp_avg_sq ~ 2x params
        optim_bytes = 2 * param_bytes

        # Gradients ~ params
        grad_bytes = param_bytes

        # Activation estimate: approximate memory needed for activations during forward
        # Rough per-layer factor: ~3 tensors (attention qkv/attn_out/ffn) per token
        activation_per_token = model_dim * dtype_size * (n_layers * 3)
        activations_bytes = batch_size * max_seq_len * activation_per_token

        # Input embeddings storage during forward
        embed_bytes = batch_size * max_seq_len * embed_dim * dtype_size

        # AMP FP32 master param copy overhead (if using mixed precision)
        amp_bytes = param_bytes if use_amp else 0

        total = (
            param_bytes
            + optim_bytes
            + grad_bytes
            + activations_bytes
            + embed_bytes
            + amp_bytes
        )
        reserve = int(total * 0.2)
        total_with_reserve = total + reserve

        return {
            "param_bytes": param_bytes,
            "optim_bytes": optim_bytes,
            "grad_bytes": grad_bytes,
            "activations_bytes": activations_bytes,
            "embed_bytes": embed_bytes,
            "amp_bytes": amp_bytes,
            "reserve": reserve,
            "total": total_with_reserve,
        }

    print("Estimating memory footprint for the chosen configuration...")
    try:
        est = _estimate_memory(
            embed_dim=getattr(sonar, "embed_dim", args.embed_dim),
            model_dim=args.model_dim,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            use_amp=(device.type == "cuda"),
        )

        def _to_mib(b):
            return float(b) / (1024.0 * 1024.0)

        print(
            f"Estimated GPU memory (including 20% reserve): {int(_to_mib(est['total']))} MiB"
        )
        print(
            f"  params: {int(_to_mib(est['param_bytes']))} MiB, optimizer: {int(_to_mib(est['optim_bytes']))} MiB, grads: {int(_to_mib(est['grad_bytes']))} MiB"
        )
        print(
            f"  activations (approx): {int(_to_mib(est['activations_bytes']))} MiB, embeddings: {int(_to_mib(est['embed_bytes']))} MiB, amp_overhead: {int(_to_mib(est['amp_bytes']))} MiB"
        )
        # If CUDA is available try to show device total memory
        if torch.cuda.is_available() and device.type == "cuda":
            try:
                free, total = torch.cuda.mem_get_info(device.index)
            except Exception:
                try:
                    total = torch.cuda.get_device_properties(device.index).total_memory
                    free = None
                except Exception:
                    total = None
                    free = None
            if total is not None:
                print(f"  GPU total memory: {int(_to_mib(total))} MiB")
                if free is not None:
                    print(f"  GPU free memory: {int(_to_mib(free))} MiB")

                if total is not None and est["total"] > total:
                    print(
                        "Warning: estimated required memory exceeds GPU total memory. Consider reducing --batch_size or model size or using --accum_steps."
                    )

                # Automatic batch size suggestion: estimate per-sample memory and
                # compute the largest batch that would fit in available memory.
                try:
                    dtype_size = 4
                    chosen_embed = getattr(sonar, "embed_dim", args.embed_dim)
                    # approximate activation cost per token (same formula as earlier)
                    activation_per_token = (
                        args.model_dim * dtype_size * (args.n_layers * 3)
                    )
                    if args.checkpointing:
                        # checkpointing reduces activation memory; use rough factor
                        activation_per_token = int(activation_per_token * 0.5)

                    per_sample = (
                        activation_per_token * args.max_seq_len
                        + chosen_embed * args.max_seq_len * dtype_size
                    )

                    fixed = (
                        est["param_bytes"]
                        + est["optim_bytes"]
                        + est["grad_bytes"]
                        + est["amp_bytes"]
                        + est["reserve"]
                    )

                    avail = free if free is not None else total
                    if avail is not None and per_sample > 0:
                        # leave a little room for system / other CUDA uses
                        safe_avail = int(avail * 0.95)
                        remaining = safe_avail - int(fixed)
                        if remaining <= 0:
                            suggested = 0
                        else:
                            suggested = int(remaining // per_sample)
                        if suggested < 1:
                            print(
                                "Auto-suggestion: no per-step batch size fits given this model & GPU. Try enabling --checkpointing, reduce model size, or run on CPU."
                            )
                        else:
                            # cap suggestion to a reasonable max (avoid huge numbers)
                            suggested = max(1, min(suggested, 1024))
                            if suggested < args.batch_size:
                                print(
                                    f"Auto-suggestion: reduce --batch_size to {suggested} (current {args.batch_size}) or increase --accum_steps to keep effective batch size."
                                )
                            else:
                                print(
                                    f"Auto-suggestion: --batch_size up to {suggested} should fit on this GPU."
                                )
                except Exception:
                    pass
    except Exception as e:
        print(f"Memory estimation failed: {e}")

    print("Encoding sentences (this may take a while)...")

    # Option: load precomputed embeddings to skip encoding step
    if args.embed_cache and os.path.exists(args.embed_cache):
        try:
            print(f"Loading precomputed embeddings from {args.embed_cache}")
            embeddings_seqs = torch.load(args.embed_cache)
        except Exception as e:
            print(f"Failed to load embed cache {args.embed_cache}: {e}. Recomputing.")
    else:
        embeddings_seqs = None

    if embeddings_seqs is None:
        # Flatten sentences across documents so we can batch many sentences per encoder call.
        flat_sents = []
        flat_doc_idx = []
        for di, sents in enumerate(docs):
            for s in sents:
                flat_sents.append(s)
                flat_doc_idx.append(di)
        print(
            f"Encoding {len(flat_sents)} sentences from {len(docs)} documents using device={device}"
        )

        embed_list = []
        batch_sent = args.encode_batch_size
        total = len(flat_sents)
        for i in tqdm(range(0, total, batch_sent), desc="encoding batches"):
            chunk = flat_sents[i : i + batch_sent]
            with torch.no_grad():
                emb_chunk = sonar.encode_sentences(chunk)
            # move to cpu to free GPU memory
            embed_list.extend([e.cpu() for e in emb_chunk])

        # Reconstruct per-document sequences
        embeddings_seqs = [[] for _ in range(len(docs))]
        for emb, didx in zip(embed_list, flat_doc_idx):
            embeddings_seqs[didx].append(emb)

        # stack per-document tensors
        for i in range(len(embeddings_seqs)):
            if len(embeddings_seqs[i]) == 0:
                embeddings_seqs[i] = torch.empty((0, sonar.embed_dim))
            else:
                embeddings_seqs[i] = torch.stack(embeddings_seqs[i], dim=0)

        # Optionally save cache
        if args.embed_cache:
            try:
                cache_dir = os.path.dirname(args.embed_cache)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                torch.save(embeddings_seqs, args.embed_cache)
                print(f"Saved embeddings to {args.embed_cache}")
            except Exception as e:
                print(f"Failed to save embeddings cache: {e}")

    # Ensure we have at least one usable document sequence before creating DataLoader.
    # A usable sequence must contain at least two sentence embeddings because
    # the dataset yields (seq[:-1], seq[1:]) pairs for training.
    usable_count = 0
    for seq in embeddings_seqs:
        if hasattr(seq, "shape") and seq.shape[0] >= 2:
            usable_count += 1

    if usable_count == 0:
        print(
            "No usable document sequences found. Need at least one document with >=2 sentences (after filtering)."
        )
        print(
            f"Documents loaded: {len(docs)}; usable sequences (>=2 sentences): {usable_count}"
        )
        print(
            "Possible fixes: 1) increase --num_docs, 2) pass --data_path to point to a local dataset, 3) check network/access to Hugging Face Hub, or 4) reduce filtering in prepare_data()."
        )
        sys.exit(1)

    dataset = EmbeddingDataset(embeddings_seqs)
    # Use DataLoader with configurable workers and pin_memory for GPU training
    dl_kwargs = dict(batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    if args.num_workers and args.num_workers > 0:
        # On Windows DataLoader sometimes deadlocks with persistent_workers=True.
        # Enable persistent_workers only on non-win platforms to avoid hangs.
        worker_kwargs = dict(num_workers=args.num_workers, prefetch_factor=2)
        if sys.platform != "win32":
            worker_kwargs["persistent_workers"] = True
        dl_kwargs.update(worker_kwargs)
    # pin_memory helps faster transfer to CUDA devices
    if device.type == "cuda":
        dl_kwargs["pin_memory"] = True

    dataloader = DataLoader(dataset, **dl_kwargs)
    # Quick runtime info to help diagnose stalls (dataset size and number of batches).
    n_batches = len(dataloader)
    print(
        f"DataLoader created. dataset_size={len(dataset)} usable_sequences={usable_count} batches={n_batches} num_workers={dl_kwargs.get('num_workers', 0)}"
    )

    print("Building LCM model...")
    # Ensure encoder embed dim matches sonar encoder output
    try:
        chosen_embed_dim = sonar.embed_dim
    except Exception:
        chosen_embed_dim = args.embed_dim

    model = BaseLCM(
        embed_dim=chosen_embed_dim,
        model_dim=args.model_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        checkpointing=args.checkpointing,
    ).to(device)
    if wandb_module is not None:
        wandb_module.watch(model, log="all", log_freq=100)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    mse = torch.nn.MSELoss()
    # Use automatic mixed precision (AMP) when running on CUDA to reduce memory
    # usage and improve performance. Only enable when a CUDA device is present.
    use_amp = device.type == "cuda"
    # Initialize GradScaler using the best available API
    if use_amp:
        if AmpGradScaler is not None:
            scaler = AmpGradScaler("cuda")
        else:
            from torch.cuda.amp import GradScaler as _CudaGradScaler

            scaler = _CudaGradScaler()
    else:
        scaler = None

    # Provide a context manager factory for autocast compatible with both APIs
    if AmpAutocast is not None:

        def autocast_cm(enabled: bool):
            return AmpAutocast("cuda", enabled=enabled)
    else:
        from torch.cuda.amp import autocast as _cuda_autocast

        def autocast_cm(enabled: bool):
            return _cuda_autocast(enabled=enabled)

    # Compute gradient accumulation steps automatically so the script is
    # usable on GPUs with small memory without requiring an extra flag.
    # We target an effective batch size (across accumulation) of ~32 by default
    # so users can pass a small per-step --batch_size and still get a reasonable
    # effective batch. This keeps the CLI unchanged while enabling accumulation
    # transparently.
    target_effective_batch = int(os.environ.get("TARGET_EFFECTIVE_BATCH", "32"))
    accum_steps = max(
        1, math.ceil(float(target_effective_batch) / float(max(1, args.batch_size)))
    )
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        start = time.time()
        global_step = (
            epoch * 1000000
        )  # large base to keep epoch steps distinct if desired
        # Gradient accumulation support: zero grads before accumulation loop
        optim.zero_grad()
        for batch_idx, (src, tgt) in enumerate(tqdm(dataloader)):
            src = src.to(device)
            tgt = tgt.to(device)
            # Predict for all positions in a single decoder call to reduce
            # memory overhead and avoid looping over sequence length.
            # src: [B, S, E], tgt: [B, L, E]
            with autocast_cm(use_amp):
                pred = model(src, tgt)  # [B, L, E]

                # Compute per-position MSE across embedding dimension, then
                # mask out padding positions where target embeddings are zeros.
                per_pos_mse = ((pred - tgt) ** 2).mean(dim=-1)  # [B, L]
                mask = (tgt.norm(dim=-1) > 0).float()  # [B, L]
                valid = mask.sum()
                if valid == 0:
                    # fallback (shouldn't normally happen)
                    loss = per_pos_mse.mean()
                else:
                    loss = (per_pos_mse * mask).sum() / valid

            # Normalize loss by accumulation steps
            loss = loss / accum_steps

            # Backward using scaler if AMP enabled
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_last = batch_idx == (n_batches - 1)
            # Step when we've accumulated enough gradients or at the last batch
            if (batch_idx + 1) % accum_steps == 0 or is_last:
                # Unscale for accurate grad clipping / logging
                if use_amp:
                    try:
                        scaler.unscale_(optim)
                    except Exception:
                        pass

                # per-step logging (loss is the scaled per-accum-step loss; report scaled back)
                report_loss = (loss * accum_steps).detach()
                lr = optim.param_groups[0]["lr"]
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2).item()
                        total_norm += param_norm * param_norm
                total_norm = total_norm**0.5
                if writer is not None:
                    global_step += 1
                    writer.add_scalar(
                        "train/step_loss", report_loss.item(), global_step
                    )
                    writer.add_scalar("train/lr", lr, global_step)
                    writer.add_scalar("train/grad_norm", total_norm, global_step)

                step_losses.append(report_loss.item())
                lrs.append(lr)
                grad_norms.append(total_norm)

                # Optimizer step (wrap to provide clearer guidance on OOM)
                try:
                    if use_amp:
                        scaler.step(optim)
                        scaler.update()
                    else:
                        optim.step()
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    # Provide a clearer, action-oriented message when CUDA runs out of memory
                    msg = str(e)
                    if "out of memory" in msg.lower() or isinstance(
                        e, torch.cuda.OutOfMemoryError
                    ):
                        print("CUDA out of memory during optimizer step.")
                        print(
                            "Possible mitigations: 1) reduce --batch_size, 2) reduce model size (--n_layers, --model_dim), 3) increase --accum_steps, 4) run on a GPU with more memory or use --device cpu"
                        )
                    raise
                optim.zero_grad()

                total += report_loss.item()
                n += 1

        elapsed = time.time() - start
        print(f"Epoch {epoch + 1} avg loss: {total / n:.4f} time: {elapsed:.1f}s")
        train_losses.append(total / n)
        os.makedirs("lcm_models", exist_ok=True)
        torch.save(model.state_dict(), f"lcm_models/lcm_sonar_epoch{epoch + 1}.pth")

        # Log epoch-level training loss
        if writer is not None:
            writer.add_scalar("train/loss", total / n if n > 0 else 0.0, epoch + 1)
            # parameter histograms
            for name, param in model.named_parameters():
                writer.add_histogram(
                    f"params/{name}", param.clone().cpu().data.numpy(), epoch + 1
                )
        if wandb_module is not None:
            wandb_module.log(
                {"train/loss": total / n if n > 0 else 0.0}, step=epoch + 1
            )

        # Optional evaluation using supplied hypothesis/reference files
        if args.eval_hyp and args.eval_ref:
            with open(args.eval_hyp, encoding="utf-8") as f:
                hyps = [l.strip() for l in f if l.strip()]
            with open(args.eval_ref, encoding="utf-8") as f:
                refs = [l.strip() for l in f if l.strip()]
            metrics = compute_all(hyps, refs, comet_model_name=args.comet_model)
            eval_metrics_history.append(metrics.copy())
            for k, v in metrics.items():
                if writer is not None:
                    writer.add_scalar(f"eval/{k}", v, epoch + 1)
            # also write a CSV summary in the log dir
            try:
                out_csv = os.path.join(args.log_dir, "eval_summary.csv")
                header = sorted(metrics.keys())
                header = ["epoch"] + header
                if not os.path.exists(out_csv):
                    with open(out_csv, "w", encoding="utf-8") as f:
                        f.write(",".join(header) + "\n")
                with open(out_csv, "a", encoding="utf-8") as f:
                    vals = [str(metrics[k]) for k in sorted(metrics.keys())]
                    f.write(str(epoch + 1) + "," + ",".join(vals) + "\n")
            except Exception:
                pass
            # checkpoint best by preferred metric (chrF++ then BLEU then METEOR)
            monitor_score = None
            for cand in ["chrF++", "BLEU", "METEOR"]:
                val = metrics.get(cand, float("nan"))
                try:
                    if not (val is None) and (val == val):
                        monitor_score = float(val)
                        break
                except Exception:
                    continue
            if monitor_score is not None:
                # higher is better
                best_path = f"lcm_models/lcm_sonar_best.pth"
                prev_best = getattr(main, "_best_score", None)
                if prev_best is None or monitor_score > prev_best:
                    os.makedirs("lcm_models", exist_ok=True)
                    torch.save(model.state_dict(), best_path)
                    setattr(main, "_best_score", monitor_score)
                    # upload model artifact to W&B
                    try:
                        if wandb_module is not None:
                            art = wandb_module.Artifact("lcm_sonar_model", type="model")
                            art.add_file(best_path)
                            wandb_module.run.log_artifact(art)
                    except Exception:
                        pass
        # Log example reconstructions (first doc first sentence)
        try:
            if (
                writer is not None
                and len(embeddings_seqs) > 0
                and len(embeddings_seqs[0]) > 0
            ):
                sample_emb = embeddings_seqs[0][0].unsqueeze(0).to(device)
                recon = sonar.decode_embeddings(sample_emb.cpu(), max_len=128)[0]
                writer.add_text("examples/reconstruction", recon, epoch + 1)
                if wandb_module is not None:
                    try:
                        wandb_module.log(
                            {"examples/reconstruction": recon}, step=epoch + 1
                        )
                        # also save a small text artifact
                        art = wandb_module.Artifact("recon_text", type="reconstruction")
                        tmp = os.path.join(
                            args.log_dir or ".", f"recon_epoch{epoch + 1}.txt"
                        )
                        with open(tmp, "w", encoding="utf-8") as f:
                            f.write(recon)
                        art.add_file(tmp)
                        wandb_module.run.log_artifact(art)
                    except Exception:
                        pass
        except Exception:
            pass
        else:
            # no external eval: checkpoint by negative training loss (higher better)
            monitor_score = -(total / n if n > 0 else float("inf"))
            best_path = f"lcm_models/lcm_sonar_best.pth"
            prev_best = getattr(main, "_best_score", None)
            if prev_best is None or monitor_score > prev_best:
                os.makedirs("lcm_models", exist_ok=True)
                torch.save(model.state_dict(), best_path)
                setattr(main, "_best_score", monitor_score)

    # Generate and save plots for paper
    if train_losses:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Plot 1: Training loss per epoch
        axes[0, 0].plot(range(1, len(train_losses) + 1), train_losses, marker="o")
        axes[0, 0].set_title("Training Loss per Epoch")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")

        # Plot 2: Step losses
        if step_losses:
            axes[0, 1].plot(step_losses)
            axes[0, 1].set_title("Step Loss")
            axes[0, 1].set_xlabel("Step")
            axes[0, 1].set_ylabel("Loss")

        # Plot 3: Learning rate
        if lrs:
            axes[1, 0].plot(lrs)
            axes[1, 0].set_title("Learning Rate")
            axes[1, 0].set_xlabel("Step")
            axes[1, 0].set_ylabel("LR")

        # Plot 4: Gradient norm
        if grad_norms:
            axes[1, 1].plot(grad_norms)
            axes[1, 1].set_title("Gradient Norm")
            axes[1, 1].set_xlabel("Step")
            axes[1, 1].set_ylabel("Norm")

        plt.tight_layout()
        save_dir = args.log_dir if args.log_dir else "."
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, "training_curves.png"), dpi=300, bbox_inches="tight"
        )
        plt.savefig(os.path.join(save_dir, "training_curves.pdf"), bbox_inches="tight")
        plt.close()

        # If eval metrics
        if eval_metrics_history:
            keys = list(eval_metrics_history[0].keys())
            fig, ax = plt.subplots(figsize=(8, 6))
            for key in keys:
                values = [m.get(key, float("nan")) for m in eval_metrics_history]
                ax.plot(range(1, len(values) + 1), values, marker="o", label=key)
            ax.set_title("Evaluation Metrics")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.legend()
            plt.tight_layout()
            save_dir = args.log_dir if args.log_dir else "."
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, "eval_metrics.png"), dpi=300, bbox_inches="tight"
            )
            plt.savefig(os.path.join(save_dir, "eval_metrics.pdf"), bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    main()
