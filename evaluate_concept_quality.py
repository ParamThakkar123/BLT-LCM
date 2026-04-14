import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from blt_lcm import PatchDataset, LCM
import json


def evaluate_concept_quality(mode, model_path="trained_lcm.pth"):
    # Load trained LCM model
    model = LCM()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dataset = PatchDataset("patching_results.jsonl", mode=mode)
    dataloader = DataLoader(dataset, batch_size=4)

    criterion = nn.MSELoss()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            patches, _ = batch
            if patches is None or patches.numel() == 0:
                continue
            output = model(patches.unsqueeze(0))
            loss = criterion(output, patches.unsqueeze(0))
            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / count if count > 0 else float("inf")
    print(f"Concept reconstruction loss ({mode}): {avg_loss}")
    return avg_loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="blt")
    parser.add_argument("--model_path", default="trained_lcm.pth")
    args = parser.parse_args()
    evaluate_concept_quality(args.mode, args.model_path)
