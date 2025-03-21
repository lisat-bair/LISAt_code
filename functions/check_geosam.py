import torch

decoder_path = "sam_decoder_multi_text.pth"
decoder_state_dict = torch.load(decoder_path, map_location="cpu")

# Explore the structure
print("Keys:", decoder_state_dict.keys())
print("Projection layer keys:", decoder_state_dict['projection_layer'].keys())
print("SAM keys:", decoder_state_dict['sam'].keys())
