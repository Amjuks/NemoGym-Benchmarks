from pathlib import Path
from datasets import load_dataset
import json

ROOT = Path("benchmarks/owasp_llm02/data/raw")


def inspect(name, ds):
    print(f"\n{'=' * 60}")
    print(name)

    for split, data in ds.items():
        print(f"Split: {split}")
        print(f"Rows: {len(data)}")
        print(f"Fields: {list(data.features)}")

        if len(data):
            print("Example:")
            print(json.dumps(
                data[0],
                indent=2,
                ensure_ascii=False,
                default=str
            )[:2500])


# Standard datasets
for name in ["password_eval", "peep", "comparison"]:
    try:
        ds = load_dataset(str(ROOT / name))
        inspect(name, ds)
    except Exception as e:
        print(f"{name}: ERROR: {e}")


# ALERT: explicitly load both configurations
for config in ["alert", "alert_adversarial"]:
    try:
        filename = f"{config}.jsonl"
        ds = load_dataset(
            "json",
            data_files=str(ROOT / "alert" / filename)
        )
        inspect(f"alert/{config}", ds)
    except Exception as e:
        print(f"alert/{config}: ERROR: {e}")


# PrivAwareBench: load each JSONL independently
print("\n" + "=" * 60)
print("PrivAwareBench individual files")

root = ROOT / "privawarebench"

for path in sorted(root.rglob("*.jsonl")):
    try:
        ds = load_dataset(
            "json",
            data_files=str(path)
        )

        data = ds["train"]

        print(f"\nFile: {path.relative_to(root)}")
        print(f"Rows: {len(data)}")
        print(f"Fields: {list(data.features)}")

        if len(data):
            print("Example:")
            print(json.dumps(
                data[0],
                indent=2,
                ensure_ascii=False,
                default=str
            )[:1000])

    except Exception as e:
        print(f"ERROR: {path}: {e}")