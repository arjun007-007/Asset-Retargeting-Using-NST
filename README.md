# Automated Asset Retargeting Using Neural Style Transfer

This project provides an **AdaIN-based** Neural Style Transfer pipeline to retarget 2D texture assets (e.g., from Poly Haven) into various styles (modern, antique, industrial, etc.).

---

## Project Structure

```
AutomatedAssetRetargeting/
├── data/
│   ├── raw/
│   │   ├── content/
│   │   └── style/
│   └── processed/
├── models/
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── style_transfer/
│   │   ├── networks.py
│   │   └── transforms.py
│   │   └── utils.py
│   ├── config.py
│   ├── train.py
│   └── inference.py
└── tests/
    ├── test_data_loading.py
    └── test_style_transfer.py
```

---

## Setup

Below is the recommended environment. You do **not** need a virtual environment, but we strongly recommend it for clean installations.

### Requirements

Install the following Python packages (Python 3.8+ recommended):

```bash
pip install torch torchvision torchaudio
pip install pillow
pip install tqdm
pip install numpy
pip install opencv-python  # optional, if needed
pip install pytest         # optional, if you want to run tests with pytest
```

A more comprehensive set of dependencies:
```
# requirements.txt (if you decide to put this in a file)
torch==2.0.0
torchvision==0.15.1
pillow==9.5.0
tqdm==4.65.0
numpy==1.24.3
opencv-python==4.7.0.72
pytest==7.3.1
```

Adjust version numbers as needed.  

---

## Downloading Dataset

We use **Poly Haven** for style textures. Here’s how to download:

1. Go to [Poly Haven Textures](https://polyhaven.com/textures).  
2. Select a texture category (wood, metal, fabric, etc.), choose a resolution (1K/2K/4K), and download the color (albedo) maps.  
3. Organize them under `data/raw/style/` (e.g., `data/raw/style/wood/wood_planks_001_Color.png`).  
4. For **content images**, either create your own or place any images in `data/raw/content/`.

**Example directory**:
```
data/
├── raw/
│   ├── content/
│   │   └── example_content.jpg
│   └── style/
│       ├── wood/
│       │   ├── wood_planks_001_Color.png
```

---

## Usage

### 1. Training

Edit `src/config.py` if you want to change hyperparameters. Then run:

```bash
python src/train.py
```

By default:
- `CONTENT_DIR` points to `data/raw/content`
- `STYLE_DIR` points to `data/raw/style`
- Model checkpoints get saved in `models/`

### 2. Inference

After training, a checkpoint file (e.g., `decoder_epoch_5.pth`) is saved in `models/`. Update `INFERENCE_DECODER_PATH` in `src/config.py` if needed, then run:

```bash
python src/inference.py
```

This will apply style transfer to the example content and style images specified in `inference.py`.

---

## Tests

We have two test files under `tests/`:
- **`test_data_loading.py`**: Validates the dataset code.
- **`test_style_transfer.py`**: Quick check to ensure style transfer code produces a valid output.

To run tests (using built-in unittest):

```bash
python -m unittest discover -s tests
```

Or, if you have `pytest` installed:

```bash
pytest tests
```

---

## License

All assets and code in this repository are for demonstration. Some external assets (e.g., from Poly Haven) are licensed under [CC0](https://creativecommons.org/share-your-work/public-domain/cc0). Check each dataset’s license individually.

---

**Happy Training & Stylizing!**  
If you encounter issues or want to contribute, feel free to open a pull request or create an issue.
```