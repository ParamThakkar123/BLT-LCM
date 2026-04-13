import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from blt_lcm import PatchDataset

# Test aggregation: mean vs max
aggregations = ["mean", "max"]

for agg in aggregations:

    class LCMAgg(nn.Module):
        def __init__(self, embed_dim=1024):
            super().__init__()
            self.encoder = nn.Linear(embed_dim, 512)
            self.decoder = nn.Linear(512, embed_dim)
            self.agg = agg

        def forward(self, x):
            if self.agg == "mean":
                x = x.mean(dim=1) if x.dim() > 1 else x
            elif self.agg == "max":
                x = x.max(dim=1)[0] if x.dim() > 1 else x
            return self.decoder(self.encoder(x))

    dataset = PatchDataset("marathi_mt_pairs.json", mode="blt")
    dataloader = DataLoader(dataset, batch_size=4)
    model = LCMAgg()
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

    print(f"Aggregation {agg}: Final loss {loss.item()}")
