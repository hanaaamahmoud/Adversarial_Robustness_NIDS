# Robustness of Machine Learning–Based Intrusion Detection Systems

## Overview
This project addresses an important security question:

**Does high detection accuracy really mean that an intrusion detection system is secure?**

The project's clear conclustion is that: **No.**

Machine learning models often perform very accurately under normal conditions, but can fail significantly when small, intentional changes are made to the data. This project provides a practical and comprehensive study illustrating the difference between statistical accuracy and actual security.
---

## Why This Project Matters
In recent years, Machine Learning and Deep Learning have been widely adopted in intrusion detection systems due to their strong performance on complex datasets. However, this performance can be **misleading from a security perspective**.

This project shows that:
- A model can achieve high accuracy,
- while still being extremely fragile when facing realistic adversarial manipulations.

For this reason, the work moves beyond accuracy and focuses on a deeper concept: **adversarial robustness**.

---

## Methodology (Project Stages)
The project is divided into five clear phases, each phase complementing the one before it, and adopting a gradual vision of the system's performance under threat.

---

### Phase 1: Data Preparation and Realistic Constraints
- Uses the **NSL‑KDD dataset**, with emphasis on the more challenging **KDDTest+** split.
- Categorical features are encoded, and numerical features are normalized.
- A **semantic masking strategy** is applied:
  - Adversarial perturbations are not allowed to modify protocol‑critical fields such as protocol type or service.
  - This guarantees that all adversarial examples remain **realistic network traffic**, rather than artificial mathematical manipulations.

---

### Phase 2: Baseline Model Training
Three fundamentally different model architectures are trained:
- **Random Forest**
- **XGBoost**
- **Multi‑Layer Perceptron (MLP)**

This architectural diversity helps reveal how different models respond to the same threat. Models are evaluated using standard metrics and confusion matrices, which already expose hidden blind spots—especially in R2L and U2R attack types.

---

### Phase 3: Adversarial Attack Generation
- Uses **IBM Adversarial Robustness Toolbox (ART)**.
- Implements **FGSM** and **PGD** attacks.
- Adversarial examples are crafted against the MLP (white‑box setting) and then transferred to Random Forest and XGBoost (black‑box transfer attacks).
- Performance is evaluated across multiple perturbation levels (ε from 0.01 to 0.2).

---

### Phase 4: Robustness and Impact Analysis
This phase focuses on understanding *how* and *why* models fail:
- Measuring accuracy degradation and Attack Success Rate (ASR).
- Identifying which network features are most exploited.
- Analyzing vulnerability across attack categories.
- Evaluating the impact of attacks on False Positive rates.

---

### Phase 5: Visualization and Insights
Results are summarized through:
- Robustness curves showing the relationship between attack strength and model accuracy.
- Heatmaps illustrating adversarial vulnerability by attack type.
- Direct comparisons between different model architectures.
- A final dashboard clearly visualizing the gap between accuracy and security.

---

## Key Findings
- **MLP Collapse:**  
  The most accurate model under normal conditions (~81%) becomes the most vulnerable under attack, reaching nearly **97% attack success rate**.
- **Random Forest Resilience:**  
  Tree‑based models show stronger natural resistance to transfer attacks.
- **Small Changes Have Big Effects:**  
  Very small perturbations (ε = 0.05) are sufficient to bypass neural‑network‑based detectors.

---

## Final Conclusion
**High accuracy does not guarantee security.**

Intrusion detection systems that rely on machine learning, especially deep learning, need to be evaluated and have specific protections developed against malicious attacks. The problem isn't just with the data or the training method, but rather with** the way the model thinks**.

---

## Threat Model
This project assumes an adversary with general knowledge of the feature space and model behavior, but constrained by realistic network protocol rules.

All adversarial perturbations preserve valid network semantics and do not modify categorical protocol fields such as protocol type or service. The goal is to evaluate model robustness under realistic conditions rather than simplified theoretical scenarios.

---

## Future Work
This work opens the door to several future research directions, including:
- Applying **adversarial training** to improve model robustness.
- Deeper analysis of attack transferability across different architectures.
- Studying the most exploited features and designing targeted defenses.
- Extending the evaluation to more modern datasets such as CIC‑IDS2017.

---

## Repository Notes
- Large datasets, trained models, and adversarial samples are intentionally excluded due to size constraints.
- All results can be reproduced using the provided code.

---
