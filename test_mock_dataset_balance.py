"""Validate balanced mock dataset generation defaults."""

from pathlib import Path
import shutil

import pandas as pd

from dataset_loader import create_mock_dataset


def main() -> None:
    out_dir = Path("/tmp/test_mock_dataset_balance")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    csv_path, image_dir = create_mock_dataset(str(out_dir))
    df = pd.read_csv(csv_path)

    counts = df["Risk of macular edema"].value_counts().sort_index().to_dict()
    expected = {0: 50, 1: 50, 2: 50, 3: 50}

    assert len(df) == 200, f"Expected 200 rows, found {len(df)}"
    assert counts == expected, f"Expected class counts {expected}, got {counts}"

    image_count = len(list(Path(image_dir).glob("*.jpg")))
    assert image_count == 200, f"Expected 200 images, found {image_count}"

    print("PASS: balanced mock dataset has 200 images and 50 samples per class.")


if __name__ == "__main__":
    main()
