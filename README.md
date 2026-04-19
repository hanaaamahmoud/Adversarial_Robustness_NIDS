# Adversarial Robustness of ML-Based Intrusion Detection Systems

## TL;DR
This project evaluates the adversarial robustness of machine learning–based intrusion
detection systems and shows that high detection accuracy does not guarantee security.

---

## Project Overview
This project studies how different machine learning models behave when exposed to
realistic adversarial perturbations in network traffic data. The focus is on understanding
model robustness rather than accuracy alone.

---

## Project Structure
The project is organized as a multi-phase pipeline:

- Phase 1: Data preprocessing and semantic constraints
- Phase 2: Baseline model training
- Phase 3: Adversarial attack generation
- Phase 4: Robustness and impact analysis
- Phase 5: Visualization and result interpretation

Each phase is implemented as a standalone Python script.

---

## Running the Code
Each phase of the project is implemented as a standalone Python script and can be executed
independently after installing the required dependencies.

---

## Key Results (Summary)

| Model | Baseline Accuracy | ASR (PGD @ ε = 0.2) | Robustness Score |
|------|-------------------|---------------------|------------------|
| Random Forest | 76.42% | 37.41% | 0.4783 |
| XGBoost | 79.09% | 33.91% | 0.5227 |
| MLP | 80.62% | 96.74% | 0.0263 |

---

## Pipeline Artifacts
Pre-trained models and preprocessed data are available 
upon request. 
Dataset: NSL-KDD — Download from:
https://www.kaggle.com/datasets/hassan06/nslkdd

---

## Requirements
The required Python libraries are listed in `requirements.txt`.

---

## Notes
Large datasets, trained models, and adversarial samples are not included in the repository
due to size constraints.
