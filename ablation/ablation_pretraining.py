import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from blt_lcm import PatchDataset, blt_model  # Assume BLT model available

# Compare pretraining
pretrains = ["random", "pretrained"]

for pre in pretrains:
    if pre == "pretrained":
        # Use pre-trained BLT
        pass  # Already loaded
    else:
        # Random init (modify blt_model weights to random)
        for param in blt_model.parameters():
            param.data = torch.randn_like(param)

    dataset = PatchDataset("marathi_mt_pairs.json", mode="blt")
    dataloader = DataLoader(dataset, batch_size=4)
    model = nn.Sequential(nn.Linear(1024, 512), nn.Linear(512, 1024))  # LCM simplified
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for step in range(50):
        for batch in dataloader:
            patches, _ = batch
            if patches is None:
                continue
            output = model(patches.unsqueeze(0))
            loss = criterion(output, patches.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"Pretraining {pre}: Final loss {loss.item()}")
