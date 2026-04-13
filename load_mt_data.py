from datasets import load_dataset
import json

# Load BhashaSetu for Marathi-English
dataset = load_dataset("ParamTh/BhashaSetu", "default")

# Extract parallel pairs (first 1000 for testing)
pairs = []
for example in dataset["train"].select(range(1000)):
    pairs.append(
        {
            "source": example["english"],  # English source
            "target": example["marathi"],  # Marathi target
        }
    )

# Save for evaluation
with open("marathi_mt_pairs.json", "w", encoding="utf-8") as f:
    json.dump(pairs, f, ensure_ascii=False, indent=2)

print("Extracted 1000 Marathi-English parallel pairs.")
