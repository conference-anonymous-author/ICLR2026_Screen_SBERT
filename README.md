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
```bash
cd ScreenSBERT
```

### GUI Parsing Module
Run:
```bash
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
```bash
python evaluate.py
```

Output:
```bash
P: 0.920, R: 0.892, F1: 0.901
Top1: 0.921
Top2: 0.938
Top3: 0.942
```

### Train
Run:
```bash
python train.py
```

Output:
```bash
Train 0 >> Loss: 0.7186, Negative Only: False
Train 1 >> Loss: 0.7043, Negative Only: False
Train 2 >> Loss: 4.0293, Negative Only: False
Train 3 >> Loss: 0.6352, Negative Only: False
Train 4 >> Loss: 1.2887, Negative Only: False
Train 5 >> Loss: 0.4721, Negative Only: False
Train 6 >> Loss: 1.3727, Negative Only: False
Train 7 >> Loss: 0.3044, Negative Only: False
Train 8 >> Loss: 2.5262, Negative Only: False
Train 9 >> Loss: 0.2081, Negative Only: False
Val 10 >> Intra: 0.346283, Inter: 2.114342, Score: 1.768059
Train 10 >> Loss: 0.1077, Negative Only: False
Train 11 >> Loss: 2.7948, Negative Only: False
Train 12 >> Loss: 1.0864, Negative Only: False
Train 13 >> Loss: 1.4073, Negative Only: False
...
```

## Baselines
You can also train and test the other models inside the baselines directory in the same way as shown above.
(However, PW2SS and CLIP cannot be retrained with the dataset format currently provided.)