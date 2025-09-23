# Screen-SBERT Reproducibility Materials

This repository provides the reproducibility materials for the ICLR 2026 submission **“Screen-SBERT: Embedding Functional Semantics of GUI Screens to Support GUI Agents.”**

In accordance with the ethics statement in the paper, the **original screenshots are not released**. Instead, we provide the **preprocessed outputs of each screenshot obtained through the GUI Parsing Module.**

Because of this data format, it is not possible to re-train baselines such as PW2SS or CLIP directly. However, you can evaluate their performance using the provided embeddings files (pre-computed embeddings for all screenshots).

For all other models:
- You may attempt to train them from scratch using the provided code.
- You can also evaluate the performance of already trained models using the provided weights files.

## Install
First clone the repo, and then install environment:
```bash
cd ICLR2026_Screen_SBERT
conda create -n "ScreenSBERT" python==3.12
conda activate ScreenSBERT
pip install --upgrade pip
pip install -r requirements.txt
```

## Screen-SBERT

### GUI Parsing Module
Run:
```bash
cd ScreenSBERT
python GUIParsingModule.py
```
Output:
```bash
Parsing results of ./example_screenshot.jpg:
Coordinates: (16, 6)
Feature Maps: (16, 25088)
Text Embeddings: (16, 768)
Functional Types: (16,)
```

### Evaluation Pre-trained Model
Run:
```
```