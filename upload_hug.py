from huggingface_hub import HfApi

# Define repository name
repo_id = "wen-han/LISAt-7k"

# Initialize Hugging Face API
api = HfApi()

# List of files to upload
files_to_upload = [
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "pytorch_model-00001-of-00002.bin",
    "pytorch_model-00002-of-00002.bin",
    "pytorch_model.bin.index.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.model"
]

# Upload each file
for file in files_to_upload:
    api.upload_file(
        path_or_fileobj=f"LISAt-7k/{file}",  # Local path
        path_in_repo=file,  # Path in repo
        repo_id=repo_id,
        repo_type="model"
    )

print("Upload complete!")
