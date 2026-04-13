import json
import math
from collections import defaultdict, Counter

# Load Marathi sentences
with open("marathi_sentences.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

# Encode to bytes
byte_sequences = [s.encode("utf-8") for s in sentences]

# Compute transition probabilities: P(next_byte | prev_byte)
transitions = defaultdict(Counter)
for seq in byte_sequences:
    for i in range(len(seq) - 1):
        prev = seq[i]
        next_b = seq[i + 1]
        transitions[prev][next_b] += 1

# Normalize to probabilities
probs = {}
for prev, counter in transitions.items():
    total = sum(counter.values())
    probs[prev] = {nb: count / total for nb, count in counter.items()}


# Function to compute entropy at each position
def compute_entropy(seq, probs, default_entropy=8.0):  # log2(256) ≈ 8
    entropies = []
    for i in range(len(seq) - 1):
        prev = seq[i]
        if prev in probs:
            p_dict = probs[prev]
            h = 0
            for p in p_dict.values():
                if p > 0:
                    h -= p * math.log2(p)
            entropies.append(h)
        else:
            entropies.append(default_entropy)
    entropies.append(default_entropy)  # for the last byte
    return entropies


# Now, for patching: patch boundaries when entropy > θ
def entropy_patching(seq, entropies, theta):
    patches = []
    current_patch = []
    for i, (byte, h) in enumerate(zip(seq, entropies)):
        current_patch.append(byte)
        if h > theta and len(current_patch) > 0:  # end patch
            patches.append(bytes(current_patch))
            current_patch = []
    if current_patch:
        patches.append(bytes(current_patch))
    return patches


# ... existing code ...

# For each sentence, compute patches for different θ
thetas = [0.5, 1.0, 1.5, 2.0, 2.5]

results = []
for idx, seq in enumerate(byte_sequences):  # all 10K
    entropies = compute_entropy(seq, probs)
    sentence_result = {
        "sentence_id": idx,
        "original": sentences[idx],
        "byte_length": len(seq),
        "entropies": entropies,
    }
    for theta in thetas:
        patches = entropy_patching(seq, entropies, theta)
        patch_lengths = [len(p) for p in patches]
        num_patches = len(patches)
        boundaries = [i for i, h in enumerate(entropies) if h > theta]
        sentence_result[f"theta_{theta}"] = {
            "num_patches": num_patches,
            "avg_patch_size": sum(patch_lengths) / num_patches
            if num_patches > 0
            else 0,
            "boundaries": boundaries,
            "entropy_at_boundaries": [entropies[i] for i in boundaries],
        }
    results.append(sentence_result)

# Save to JSONL
with open("blt_patch_extraction_results.jsonl", "w", encoding="utf-8") as f:
    for res in results:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")

print("Processed all 10000 sentences.")
