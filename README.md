# Indonesian Vowel Recognition with DTW and Gaussian Models

A pipeline to recognize `/a/`, `/i/`, `/u/`, `/e/`, and `/o/` using DTW templates and MFCC-based multivariate Gaussian models.

## Installation

Use Python 3.10+.

```bash
pip install -r requirements.txt
pip install -e .
```

## Data Structure

Store recordings directly in `data/S01/a/train.wav` and `data/S01/a/test.wav`, for all vowels and speakers. Speaker configurations are located in [config.yaml](https://www.google.com/search?q=config.yaml).

## Running via Notebook

Open [vowel_recognition_pipeline.ipynb](https://www.google.com/search?q=vowel_recognition_pipeline.ipynb) in VS Code or Jupyter, then run each cell.

CSV/JSON results are located in `results/tables` and `results`, and charts are in `results/figures`.