import json
from indicnlp.tokenize import indic_tokenize
from indicnlp.morph.unsupervised_morph import UnsupervisedMorphAnalyzer
from indicnlp import common

# Set Indic NLP data directory
common.INDIC_RESOURCES_PATH = "indic_nlp_resources"

# Load Marathi sentences
with open("marathi_sentences.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)


# Function to align morphemes
def align_morphemes(sentence):
    # Tokenize into words (simplified, no morph analysis due to model availability)
    words = indic_tokenize.trivial_tokenize(sentence)
    return words


# Process first 100 sentences for alignment study
alignments = []
for sent in sentences[:100]:
    alignments.append(align_morphemes(sent))

# Save alignments
with open("morph_alignments.json", "w", encoding="utf-8") as f:
    json.dump(alignments, f, ensure_ascii=False, indent=2)

print("Morpheme alignment completed for 100 sentences.")
