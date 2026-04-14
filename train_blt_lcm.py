import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class PatchDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # The JSONL produced by the entropy patcher stores per-threshold data
        # under keys like "theta_0.5" with a nested "patches" list. Fall
        # back to top-level "patches" if present. Produce a deterministic
        # fixed-size embedding per sentence so the small LCM scaffold can run
        # without the full BLT pipeline.
        patches = item.get("patches")
        if patches is None:
            # look for theta_* entries and pick one (prefer 0.5)
            theta_entry = item.get("theta_0.5") or next(
                (v for k, v in item.items() if k.startswith("theta_")), {}
            )
            patches = (
                theta_entry.get("patches") if isinstance(theta_entry, dict) else None
            )

        # Generate a deterministic embedding from the original sentence if we
        # don't have numeric patch embeddings. This keeps the scaffold runnable.
        boundaries = item.get("boundaries") or []
        if patches is None:
            text = item.get("original", "")
            patches_t = _deterministic_text_embedding(text, dim=1024)
        else:
            # If patches exist as a list of strings, convert to a simple
            # numeric embedding by taking per-patch length statistics and
            # expanding to fixed size.
            try:
                lens = [len(p) for p in patches]
                # create a fixed-size vector: mean, std, min, max, and pad
                import math

                mean = sum(lens) / len(lens) if len(lens) > 0 else 0.0
                var = (
                    sum((x - mean) ** 2 for x in lens) / len(lens)
                    if len(lens) > 0
                    else 0.0
                )
                std = math.sqrt(var)
                vec = [
                    mean,
                    std,
                    min(lens) if lens else 0.0,
                    max(lens) if lens else 0.0,
                ]
                # pad/repeat to 1024
                rep = (vec * ((1024 // len(vec)) + 1))[:1024]
                patches_t = torch.tensor(rep, dtype=torch.float32)
            except Exception:
                patches_t = _deterministic_text_embedding(
                    item.get("original", ""), dim=1024
                )

        return patches_t, boundaries


def _deterministic_text_embedding(text: str, dim: int = 1024) -> torch.Tensor:
    import hashlib

    h = hashlib.md5(text.encode("utf-8") if isinstance(text, str) else b"")
    b = h.digest()  # 16 bytes
    # Repeat the digest to fill dim and normalize to [0,1]
    reps = (b * ((dim // len(b)) + 1))[:dim]
    arr = [x / 255.0 for x in reps]
    return torch.tensor(arr, dtype=torch.float32)


class LCM(nn.Module):
    def __init__(self, embed_dim=1024):
        super(LCM, self).__init__()
        self.encoder = nn.Linear(embed_dim, 512)
        self.decoder = nn.Linear(512, embed_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_lcm(mode, lr, steps, data_path):
    dataset = PatchDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = LCM()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    global_step = 0
    for epoch in range(steps):
        for batch in dataloader:
            patches, boundaries = batch
            # Skip non-tensor patches
            if not isinstance(patches, torch.Tensor):
                continue
            if patches.dim() == 1:
                patches = patches.unsqueeze(0)

            output = model(patches)
            # Ensure shapes match for loss computation
            if output.shape != patches.shape:
                # Try to broadcast or reshape if possible; otherwise skip
                try:
                    target = patches.view_as(output)
                except Exception:
                    continue
            else:
                target = patches

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                print(f"Epoch {epoch} Step {global_step}, Loss: {loss.item():.6f}")
            global_step += 1

    torch.save(model.state_dict(), f"lcm_{mode}.pth")
    print(f"Model saved as lcm_{mode}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="fallback")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--data_path", default="patching_results.jsonl")
    args = parser.parse_args()
    train_lcm(args.mode, args.lr, args.steps, args.data_path)
