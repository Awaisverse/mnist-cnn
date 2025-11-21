# Fix Notebook Metadata

The notebook is missing kernel metadata. To fix it:

## Option 1: Run the Fix Script (Recommended)

If you have Python installed, run:

```bash
python fix_notebook_metadata.py
```

This will add the required `kernelspec` metadata to the notebook.

## Option 2: Fix in Kaggle

1. Upload the notebook to Kaggle as-is
2. Kaggle will automatically detect it's a Python notebook and add the kernel metadata
3. This usually works automatically!

## Option 3: Manual Fix (if needed)

Open `mnist_training.ipynb` in a text editor and find the metadata section near the end (around line 383). Change:

```json
"metadata": {
  "language_info": {
    "name": "python"
  }
}
```

To:

```json
"metadata": {
  "kernelspec": {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
  },
  "language_info": {
    "name": "python"
  }
}
```

Then save the file.
