import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json

# Lazy BLT/SONAR integration: try importing libraries but avoid loading large
# pretrained weights at import time. If imports fail we fall back to simple
# entropy-based patching so the rest of the pipeline can run on Windows/without
# HF access.
USE_BLT = False
entropy_model = None
blt_model = None
tok_and_patcher = None
tokenizer = None
patcher = None


def ensure_blt_loaded(entropy_repo="facebook/blt-entropy", blt_repo="facebook/blt-1b"):
    """Attempt to import BLT and load pretrained pieces. Returns True on success."""
    global USE_BLT, entropy_model, blt_model, tok_and_patcher, tokenizer, patcher
    if USE_BLT:
        return True
    try:
        from bytelatent.transformer import LMTransformer
        from bytelatent.model.blt import ByteLatentTransformer
        from bytelatent.hf import BltTokenizerAndPatcher

        # Attempt to load (may require HF auth / accepted terms)
        entropy_model = LMTransformer.from_pretrained(entropy_repo)
        blt_model = ByteLatentTransformer.from_pretrained(blt_repo)
        tok_and_patcher = BltTokenizerAndPatcher.from_pretrained(blt_repo)
        # tokenizers in bytelatent expose builders; keep a simple reference
        try:
            tokenizer = tok_and_patcher.tokenizer_args.build()
            patcher = tok_and_patcher.patcher_args.build()
        except Exception:
            # Some builds may require explicit constructors; fall back to None
            tokenizer = None
            patcher = None

        USE_BLT = True
        print("BLT components loaded.")
        return True
    except Exception as e:
        print(f"BLT initialization failed ({e}). Falling back to entropy patcher.")
        USE_BLT = False
        entropy_model = None
        blt_model = None
        tok_and_patcher = None
        tokenizer = None
        patcher = None
        return False


def _simple_word_patcher(text, threshold=0.5):
    # Minimal, deterministic fallback: split on whitespace
    return text.split()


# Ensure a usable patcher exists
if patcher is None:
    patcher = _simple_word_patcher


class LCM(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.encoder = nn.Linear(embed_dim, 512)
        self.decoder = nn.Linear(512, embed_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def get_blt_patches(sentence):
    inputs = blt_tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = blt_model(**inputs)
    # Simplified: use mean hidden state as patch embedding
    patches = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    return patches


def get_sonar_concepts(sentence):
    import sonar

    # Load SONAR model
    sonar_model = sonar.load("sonar_basic_encoder")
    # Get embedding
    embedding = sonar_model.embed(sentence)
    return torch.tensor(embedding).squeeze(0)


def get_fallback_patches(sentence, theta=0.5):
    # Simplified entropy patching
    byte_seq = sentence.encode("utf-8")
    patches = []
    current = []
    for b in byte_seq:
        current.append(b)
        if (
            len(current) > 1 and sum(current) / len(current) > theta * 255
        ):  # Dummy entropy
            patches.append(bytes(current[:-1]))
            current = [b]
    patches.append(bytes(current))
    return torch.tensor(
        [len(p) for p in patches], dtype=torch.float
    )  # Dummy embeddings


class PatchDataset(Dataset):
    def __init__(self, data_path, mode="fallback"):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item["original"]
        if self.mode == "blt":
            patches = get_blt_patches(sentence)
        elif self.mode == "sonar":
            patches = get_sonar_concepts(sentence)
        else:
            patches = get_fallback_patches(sentence)
        return patches, torch.tensor([])  # Boundaries placeholder


def train_lcm(mode, lr, steps, data_path):
    dataset = PatchDataset(data_path, mode)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Small batch
    model = LCM()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for step in range(steps):
        for batch in dataloader:
            patches, _ = batch
            if isinstance(patches, list):
                continue  # Skip if not tensor
            output = model(patches.unsqueeze(0) if patches.dim() == 1 else patches)
            target = patches.unsqueeze(0) if patches.dim() == 1 else patches
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if "loss" in locals() and step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="blt")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--data_path", default="patching_results.jsonl")
    args = parser.parse_args()
    train_lcm(args.mode, args.lr, args.steps, args.data_path)
