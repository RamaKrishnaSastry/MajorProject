"""Validate balanced mock dataset generation defaults."""

from pathlib import Path
import shutil
import tempfile

import pandas as pd

from dataset_loader import create_mock_dataset


def main() -> None:
    out_dir = Path(tempfile.gettempdir()) / "test_mock_dataset_balance"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # Default num_samples=150 → 50 per DME class × 3 classes
    csv_path, image_dir = create_mock_dataset(str(out_dir))
    df = pd.read_csv(csv_path)

    counts = df["Risk of macular edema"].value_counts().sort_index().to_dict()
    expected = {0: 50, 1: 50, 2: 50}

    assert len(df) == 150, f"Expected 150 rows, found {len(df)}"
    assert counts == expected, f"Expected class counts {expected}, got {counts}"

    # Verify DR labels are also present with values in 0-4
    assert "Retinopathy grade" in df.columns, "Missing 'Retinopathy grade' column"
    dr_values = set(df["Retinopathy grade"].unique())
    assert dr_values.issubset({0, 1, 2, 3, 4}), f"DR labels out of range: {dr_values}"

    image_count = len(list(Path(image_dir).glob("*.jpg")))
    assert image_count == 150, f"Expected 150 images, found {image_count}"

    print("PASS: balanced mock dataset has 150 images, 50 samples per DME class, and DR labels 0-4.")


if __name__ == "__main__":
    main()
