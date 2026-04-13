import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from blt_lcm import PatchDataset

# Sweep entropy thresholds
thresholds = [0.1, 0.5, 1.0, 1.5, 2.0]

for theta in thresholds:
    # Modify get_fallback_patches with theta
    def get_fallback_patches_theta(sentence):
        byte_seq = sentence.encode("utf-8")
        patches = []
        current = []
        for b in byte_seq:
            current.append(b)
            if len(current) > 1 and sum(current) / len(current) > theta * 255:
                patches.append(bytes(current[:-1]))
                current = [b]
        patches.append(bytes(current))
        return torch.tensor([len(p) for p in patches], dtype=torch.float)

    dataset = PatchDataset("marathi_mt_pairs.json", mode="fallback")
    dataloader = DataLoader(dataset, batch_size=4)
    model = nn.Sequential(nn.Linear(10, 512), nn.Linear(512, 10))  # Simple
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for step in range(50):
        for batch in dataloader:
            patches, _ = batch
            output = model(patches.unsqueeze(0))
            loss = criterion(output, patches.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"Theta {theta}: Final loss {loss.item()}")
