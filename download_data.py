from huggingface_hub import hf_hub_download

def download_dataset():
    repo = "Ruturaj0077/product-recommendation-system"  # your HF space id
    filename = "amz_uk_processed_data.csv"

    file = hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type="space"   # because it's a HF Space, not a dataset repo
    )

    print("✅ Dataset downloaded successfully:")
    print(file)

if __name__ == "__main__":
    download_dataset()
