import json
import math

# Load sentences
with open("marathi_sentences.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)


def compute_entropy(byte_seq):
    freq = {}
    for b in byte_seq:
        freq[b] = freq.get(b, 0) + 1
    entropy = 0
    n = len(byte_seq)
    for count in freq.values():
        p = count / n
        entropy -= p * math.log2(p)
    return entropy


def patch_sentence(sentence, theta):
    byte_seq = sentence.encode("utf-8")
    patches = []
    boundaries = []
    current_patch = []
    for i, byte in enumerate(byte_seq):
        current_patch.append(byte)
        if len(current_patch) > 1:
            ent = compute_entropy(current_patch)
            if ent > theta:
                patches.append(
                    bytes(current_patch[:-1]).decode("utf-8", errors="ignore")
                )
                boundaries.append(i - 1)
                current_patch = [byte]
        if i == len(byte_seq) - 1:
            patches.append(bytes(current_patch).decode("utf-8", errors="ignore"))
    return patches, boundaries


# Ablation on theta
thetas = [0.1, 0.5, 1.0, 1.5, 2.0]
results = {}
for theta in thetas:
    all_patches = []
    all_boundaries = []
    for sent in sentences[:10]:  # Small subset for quick test
        patches, boundaries = patch_sentence(sent, theta)
        all_patches.extend(patches)
        all_boundaries.extend(boundaries)
    avg_patch_size = (
        sum(len(p) for p in all_patches) / len(all_patches) if all_patches else 0
    )
    results[f"theta_{theta}"] = {
        "num_patches": len(all_patches),
        "avg_patch_size": avg_patch_size,
        "total_boundaries": len(all_boundaries),
    }

with open("ablation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Ablation completed.")
