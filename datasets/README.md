# Datasets

Local image data lives here, but real images are ignored by git.

Expected structure:

```text
datasets/
  external/             # downloaded/unpacked third-party datasets
  source/               # source originals for generated data
  generated/
    original/
    edited/
    manifest.csv
  public/
    original/
    edited/
    manifest.csv
```

Use this command to keep a small local sample:

```powershell
python scripts/06_trim_datasets.py --limit 10
```

