import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from blt_lcm import PatchDataset

# Vary layers
layers = [1, 2, 3]

for num_layers in layers:

    class LCMLayers(nn.Module):
        def __init__(self, embed_dim=1024, layers=num_layers):
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.Linear(embed_dim, embed_dim) for _ in range(layers)]
            )
            self.encoder = nn.Linear(embed_dim, 512)
            self.decoder = nn.Linear(512, embed_dim)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.decoder(self.encoder(x))

    dataset = PatchDataset("marathi_mt_pairs.json", mode="blt")
    dataloader = DataLoader(dataset, batch_size=4)
    model = LCMLayers()
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

    print(f"Layers {num_layers}: Final loss {loss.item()}")
