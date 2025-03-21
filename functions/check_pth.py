import torch

ckpt = torch.load("sam_vit_h_4b8939.pth", map_location="cpu")
for key, val in ckpt.items():
    print(key, val.shape)
