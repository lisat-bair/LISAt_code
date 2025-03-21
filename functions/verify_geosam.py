from model.segment_anything.build_sam import build_sam_vit_h

# Test with the updated checkpoint
checkpoint_path = "sam_decoder_multi_text.pth"
model = build_sam_vit_h(checkpoint=checkpoint_path)

# Check if the projection_layer weights are loaded
print(model.projection_layer.state_dict())
