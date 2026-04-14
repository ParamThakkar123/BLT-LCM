from datasets import load_dataset
import json

# Load the dataset with streaming
ds = load_dataset("ParamTh/BhashaSetu", streaming=True, split="train")

marathi_sentences = []
count = 0
for sample in ds:
    marathi_sentences.append(sample["marathi"])
    count += 1
    if count >= 50000:
        break

# Save to a file
with open("marathi_sentences.json", "w", encoding="utf-8") as f:
    json.dump(marathi_sentences, f, ensure_ascii=False, indent=2)

print(f"Extracted {len(marathi_sentences)} Marathi sentences.")
