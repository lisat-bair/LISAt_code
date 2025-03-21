import torch

file_path = "sam_decoder_multi_text.pth"
ckpt = torch.load(file_path, map_location="cuda:4")  # or "cuda"
print("Checkpoint loaded successfully on GPU!")
print("Keys in checkpoint:", ckpt.keys())
