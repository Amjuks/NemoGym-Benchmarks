from pathlib import Path
from huggingface_hub import snapshot_download

root = Path("benchmarks/owasp_llm02/data/raw")

sources = {
    "privawarebench": "LeoWang0318/PrivAwareBench",
    "password_eval": "haritzpuerto/password_eval-contextual-integrity",
    "peep": "haritzpuerto/PEEP-contextual-integrity",
    "comparison": "v1adam/Comparison_Dataset",
    "alert": "Babelscape/ALERT",
}

for name, repo_id in sources.items():
    print(f"\nDownloading {name}: {repo_id}")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=root / name,
        )
        print(f"Downloaded: {name}")
    except Exception as exc:
        print(f"FAILED: {name}: {exc}")
