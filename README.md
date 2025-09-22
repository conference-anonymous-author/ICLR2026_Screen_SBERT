# Screen-SBERT Reproducibility Materials

This repository provides the reproducibility materials for the ICLR 2026 submission **“Screen-SBERT: Embedding Functional Semantics of GUI Screens to Support GUI Agents.”**

In accordance with the ethics statement in the paper, the **original screenshots are not released**. Instead, we provide the **preprocessed outputs of each screenshot obtained through the GUI Parsing Module.**

Because of this data format, it is not possible to re-train baselines such as PW2SS or CLIP directly. However, you can evaluate their performance using the provided embeddings files (pre-computed embeddings for all screenshots).

For all other models:
- You may attempt to train them from scratch using the provided code.
- You can also evaluate the performance of already trained models using the provided weights files.