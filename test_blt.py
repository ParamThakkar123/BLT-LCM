from transformers import AutoTokenizer, AutoModel
import torch

blt_tokenizer = AutoTokenizer.from_pretrained("facebook/blt", use_fast=False)
blt_model = AutoModel.from_pretrained("facebook/blt")

sentence = "Hello world"
inputs = blt_tokenizer(sentence, return_tensors="pt")
with torch.no_grad():
    outputs = blt_model(**inputs)
print("BLT model loaded, output shape:", outputs.last_hidden_state.shape)
