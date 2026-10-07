# Trust-Drift Fusion: A Context-Aware, Explainable, and Drift-Adaptive Framework for Real-Time Intrusion Detection in IoT Networks

Supplementary evaluation code for the paper of the same title. The repository contains a single Jupyter notebook, `TDF.ipynb`, implementing Trust-Drift Fusion (TDF), four baselines, and the associated experiments. The primary benchmark and validation diagnostics use the TON_IoT network-flow dataset; the drift-adaptive retraining experiment uses a synthetic regime-shift stream generated inside the notebook.

> **Execution status.** The notebook is distributed with **cleared outputs**. No result in this repository should be treated as freshly executed until you perform a fresh-kernel *Run All* (see [Reproducing the results](#reproducing-the-results)). See [Verification status](#verification-status).

## Overview and research purpose

TDF is a binary (Normal vs. Attack) intrusion detector that fuses three per-window signals into one Hybrid Threat (HT) score, then converts it to a decision:

| Signal | Implementation in the notebook |
|---|---|
| **AS** – anomaly score | Unsupervised Isolation Forest score, min-max normalised on the fit data |
| **RDS** – resource-drift score | Mean absolute Z-score of the (unscaled) features against mean/std of Normal-labelled training rows, divided by 4 and clipped to [0, 1] |
| **RF** – attack probability | Random Forest (`class_weight="balanced"`) positive-class probability |

`HT = w_as·AS + w_rds·RDS + w_rf·RF`. The weights (grid step 0.05, summing to 1) and the decision threshold (grid 0.01–0.99) are selected by grid search on a validation split carved out of the training data. The notebook additionally implements a context-aware (per-regime) variant, cost-sensitive thresholds, SHAP explanations, drift-triggered retraining, and an adversarial stress test.

The purpose of the notebook is to let a reviewer inspect and re-run the evaluation that accompanies the paper.

## Repository contents

| File | Description |
|---|---|
| `TDF.ipynb` | Complete evaluation notebook (data loading, preprocessing, models, all experiments, integrity checks) |
| `train_test_network.csv` | Dataset used for training and testing |
| `README.md` | This file |

No license file, requirements file, or pre-computed results are included.

## Dataset

**TON_IoT network dataset — `train_test_network.csv`**

The dataset is a third-party research dataset developed by UNSW Canberra. The official UNSW project page provides the TON_IoT datasets, including the network-data collection and Train_Test_datasets resources.

**Original source:**
https://research.unsw.edu.au/projects/toniot-datasets

**Dataset publication:**

The details of the TON_IoT datasets were published in following the papers. For the academic/public use of these datasets, the authors have to cities the following papers:

1. N. Moustafa, "A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets," *Sustainable Cities and Society*, 2021, art. 102994. <https://www.sciencedirect.com/science/article/pii/S2210670721002808>
2. T. M. Booij, I. Chiscop, E. Meeuwissen, N. Moustafa, and F. T. H. den Hartog, "ToN_IoT: The role of heterogeneity and the need for standardization of features and attack types in IoT network intrusion datasets," *IEEE Internet of Things Journal*, 2021. <https://ieeexplore.ieee.org/document/9444348>
3. A. Alsaedi, N. Moustafa, Z. Tari, A. Mahmood, and A. Anwar, "TON_IoT telemetry dataset: A new generation dataset of IoT and IIoT for data-driven intrusion detection systems," *IEEE Access*, vol. 8, pp. 165130–165150, 2020. <https://ieeexplore.ieee.org/document/9189760>
4. N. Moustafa, M. Keshk, E. Debie, and H. Janicke, "Federated TON_IoT Windows datasets for evaluating AI-based security applications," in *Proc. 2020 IEEE 19th Int. Conf. Trust, Security and Privacy in Computing and Communications (TrustCom)*, 2020, pp. 848–855, doi: 10.1109/TrustCom50675.2020.00114. <https://arxiv.org/abs/2010.08522>
5. N. Moustafa, M. Ahmed, and S. Ahmed, "Data analytics-enabled intrusion detection: Evaluations of ToN_IoT Linux datasets," in *Proc. 2020 IEEE 19th Int. Conf. TrustCom*, 2020, pp. 727–735, doi: 10.1109/TrustCom50675.2020.00100. <https://arxiv.org/abs/2010.08521>
6. N. Moustafa, "New generations of Internet of Things datasets for cybersecurity applications based machine learning: TON_IoT datasets," in *Proc. eResearch Australasia Conference*, Brisbane, Australia, 2019. <https://conference.eresearch.edu.au/wp-content/uploads/2019/08/2019_eResearch_59_New-Generations-of-Internet-of-Things-Datasets-for-Cybersecurity.pdf>
7. N. Moustafa, "A systemic IoT-Fog-Cloud architecture for big-data analytics and cyber security systems: A review of fog computing," arXiv:1906.01055, 2019. <https://arxiv.org/abs/1906.01055>
8. J. Ashraf, M. Keshk, N. Moustafa, M. Abdel-Basset, H. Khurshid, A. D. Bakhshi, and R. R. Mostafa, "IoTBoT-IDS: A novel statistical learning-enabled botnet detection framework for protecting networks of smart cities," *Sustainable Cities and Society*, 2021, art. 103041. <https://www.sciencedirect.com/science/article/abs/pii/S2210670721003255>

### Obtaining the data

1. **Load directly from GitHub (default).**  
   Section 1 reads the CSV from the repository specified in `GITHUB_CSV_URL` in the configuration cell, currently:
   `https://github.com/l0keshkum4r/TDF/blob/main/train_test_network.csv`
   A public repository requires no account or token. The standard `github.com/.../blob/...` URL is automatically converted to the corresponding raw-file URL.

2. **Download option.**  
   Either set `SAVE_LOCAL_COPY = True` in the configuration cell, which saves the file as `./train_test_network.csv` while loading, or download it manually:
   - **Browser:** Open the file page on GitHub and click **Download raw file**.
   - **Command line:**
     ```bash
     curl -L -o train_test_network.csv https://raw.githubusercontent.com/l0keshkum4r/TDF/main/train_test_network.csv
     ```
   - **Whole repository:**
     ```bash
     git clone https://github.com/l0keshkum4r/TDF.git
     ```

3. **Use a local copy (offline).**  
   Place the file at `./train_test_network.csv` (or in `data/` or `TON_IoT/`), or point the `TON_IOT_CSV` environment variable to its location.
   To skip GitHub loading, set:
   ```python
   GITHUB_CSV_URL = ""
   ```
   If the GitHub download fails, the notebook automatically falls back to these local locations.
   
4. **Run the notebook.**  
   Section 1 downloads and loads the CSV automatically.

## Methodology and workflow

1. **Load** `train_test_network.csv` (Section 1).
2. **Schema preprocessing** (no fitted statistics): drop `src_ip`, `dst_ip`, `ts` (if present) and the fine-grained `type` column; map `label` 0/1 → `Normal`/`Attack`; fill missing categorical values with `"missing"` and missing numeric values with `0`.
3. **Split** (Section 3): stratified 70/30 train/test split of the raw rows (`random_state=42`).
4. **Leakage-safe one-hot encoding:** the vocabulary is fit on the training rows only and applied to the test rows with a frozen column set (test-only categories are dropped).
5. **Validation split:** each model uses a stratified 25% validation split of the training partition (`random_state=42`) for threshold selection; TDF additionally uses it for fusion-weight selection. Scalers and detectors are fit on the sub-train partition only. The test partition is used only to compute reported metrics.
6. **Models:** Static Threshold, Naive Bayes, Isolation Forest, Logistic Regression, and TDF (Proposed).
7. **Experiments** (below), followed by final integrity and consistency checks that write a manifest of artifact hashes.

## Experimental components

| Section | Component | Data used |
|---|---|---|
| 3 | Baseline comparison – 5 models, 8 metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, Balanced Accuracy); confusion matrix, ROC and PR figures | TON_IoT, held-out test for final metrics |
| 4 | Ablation – each signal alone, pairwise equal-weight combinations, and full TDF with learned weights (thresholds chosen on validation) | TON_IoT |
| 5 | Context-aware TDF – KMeans regimes (3) discovered on the AS/RDS/RF space; per-regime weights/thresholds tuned on a disjoint subset | TON_IoT |
| 6 | Cost-sensitive threshold (FN cost 5× FP) and severity bands from a 1×/3×/8× cost schedule | TON_IoT |
| 7 | SHAP (TreeExplainer on the Random Forest component) with a 4-field explanation structure; examples taken from the TDF validation partition | TON_IoT (validation) |
| 8 | Drift-adaptive, **label-assisted** retraining vs. a frozen model, EMA of RDS with a sustained-breach trigger | **Synthetic** stream generated in the notebook (3 device-profile blocks, seed 42) — *not* TON_IoT |
| 9 | Adversarial robustness – feature-space evasion crafted against the Random Forest, applied to up to 150 validation Attack samples caught by TDF | TON_IoT (validation) |
| Reviewer diagnostic | Feature-group sensitivity (ports / DNS / HTTP / SSL removed, fresh Random Forest) | TON_IoT training partition only; test set not used |

## Requirements

Python 3.11 was recorded in the notebook metadata. The notebook imports: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `shap`. Its first code cell installs any missing packages with `pip`. `google.colab` is used only optionally, for automatic downloads. The notebook does not pin package versions, and no tested version set was recorded (see caveats).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib shap jupyter
```

## Reproducing the results

1. Install the requirements.
2. Start Jupyter in the repository directory and open `TDF.ipynb`: `jupyter notebook TDF.ipynb`.
3. Restart the kernel and **Run All**, top to bottom, without skipping cells. Later cells depend on variables from earlier ones, and re-running a single early cell in isolation is not supported.
4. The final cell group prints `STATUS: PASS — all final integrity checks passed.` if the structural checks succeed. This confirms row conservation, train/test disjointness, valid TDF weights/threshold, artifact presence (existing and non-empty), and Table 1/CSV consistency with in-memory results. It does not prove every file was written during the current run, and it does **not** validate the science.

### Generated artifacts (written to the working directory)

`table1_results.csv`, `ablation_results.csv`, `confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`, `drift_adaptive_comparison.png`, and `tdf_current_run_manifest.json` (timestamp, split sizes, TDF weights/threshold/headline metrics, and SHA-256 hashes of the six artifacts). Keep them together with the executed notebook that produced them.

## Reproducibility notes

- Fixed seed 42 is used for the train/test split, validation splits, Isolation Forest, Random Forest, KMeans, the synthetic stream, and the adversarial sample selection. No global NumPy seed is set.
- Results can depend on library versions and platform; package versions are not pinned, so do not expect bit-identical numbers across environments.

## Verification status

- **Notebook validation:** all code cells are syntactically valid and the notebook passes `nbformat` validation.
- **Real-data execution:** the distributed notebook has cleared outputs and requires a fresh-kernel Run All on the pinned TON_IoT dataset version before the results can be considered freshly reproduced. Real-data numbers are not re-verified by this release.
- **Section 8:** the synthetic drift-adaptation experiment is self-contained within the notebook and uses seed 42.

## Limitations and caveats

- **Random i.i.d. split.** Very high benchmark scores on TON_IoT under a random stratified split are not evidence of chronological, cross-device or deployment-level generalisation. The notebook does not perform temporal validation and does not check for duplicate or near-duplicate rows across the split.
- **Dataset source.** Third-party Kaggle mirror, provenance unverified.
- **Isolation Forest direction.** The AS branch and the standalone Isolation Forest baseline are fit on data containing both classes with Attack as the majority, so the anomaly signal can be inverted; the notebook documents that the baseline's ROC-AUC was below 0.5 in the recorded run.
- **RDS scaling.** RDS uses unscaled features with Normal-row standard deviations plus `1e-9`. Columns that are constant among Normal training rows (common among sparse one-hot features) can produce very large Z-scores, so RDS can saturate at 1.0.
- **Context-aware comparison is not like-for-like.** `ContextAwareTDF` first reserves 40% of the training partition for regime discovery/tuning. Its remaining 60% is then split again inside the base TDF into model-fit and validation portions, so its detectors are fitted on about 45% of the overall training partition, compared with about 75% for the global TDF.
- **Severity bands.** Cost-derived bands can collapse to identical cutoffs, the cost values are illustrative, and the single-window SHAP explanation starts from `prev_trust=100`, which keeps the Trust Score ≥ 70, so even high-threat windows land in the "Monitored" band in the examples.
- **Drift experiment.** Synthetic data, label-assisted retraining, a single seed; retraining recovery is not instantaneous, and no real-data drift is evaluated.
- **Adversarial experiment.** A white-box, feature-space stress test against the Random Forest only, on validation samples. In the original recorded run the fused TDF evasion rate was higher than the RF-alone rate (70.7% vs. 25.3%); the notebook does not claim fusion improves robustness. Treat these as unverified until you re-run.
- **Preprocessing isolation.** The one-hot vocabulary is fit on the full training partition, before TDF's internal model-fit/validation split. Test-set leakage is prevented, but preprocessing is not strictly nested inside the validation split.
- **Explainability scope.** SHAP attributions cover the Random Forest component only, not AS or RDS individually.
- **"Real-time".** The notebook contains no latency or throughput benchmark; it demonstrates per-window scoring and a synthetic batch stream only.
- **Feature-group diagnostic.** Validation-only and not part of the reported manuscript results unless independently reviewed.
- **Paper cross-references.** Equation/section numbers mentioned in code comments (e.g. "Eq. 4", "Section IV") refer to the paper and are not cross-checked by this repository.
- **Metric naming.** "PR-AUC" in the tables is `average_precision_score`.

## Citation

**TON_IoT dataset:** the UNSW TON_IoT page (<https://research.unsw.edu.au/projects/toniot-datasets>) asks users to cite its listed papers. The details of the TON_IoT datasets were published in following the papers. For the academic/public use of these datasets, the authors have to cities the following papers:

1. N. Moustafa, "A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets," *Sustainable Cities and Society*, 2021, art. 102994. <https://www.sciencedirect.com/science/article/pii/S2210670721002808>
2. T. M. Booij, I. Chiscop, E. Meeuwissen, N. Moustafa, and F. T. H. den Hartog, "ToN_IoT: The role of heterogeneity and the need for standardization of features and attack types in IoT network intrusion datasets," *IEEE Internet of Things Journal*, 2021. <https://ieeexplore.ieee.org/document/9444348>
3. A. Alsaedi, N. Moustafa, Z. Tari, A. Mahmood, and A. Anwar, "TON_IoT telemetry dataset: A new generation dataset of IoT and IIoT for data-driven intrusion detection systems," *IEEE Access*, vol. 8, pp. 165130–165150, 2020. <https://ieeexplore.ieee.org/document/9189760>
4. N. Moustafa, M. Keshk, E. Debie, and H. Janicke, "Federated TON_IoT Windows datasets for evaluating AI-based security applications," in *Proc. 2020 IEEE 19th Int. Conf. Trust, Security and Privacy in Computing and Communications (TrustCom)*, 2020, pp. 848–855, doi: 10.1109/TrustCom50675.2020.00114. <https://arxiv.org/abs/2010.08522>
5. N. Moustafa, M. Ahmed, and S. Ahmed, "Data analytics-enabled intrusion detection: Evaluations of ToN_IoT Linux datasets," in *Proc. 2020 IEEE 19th Int. Conf. TrustCom*, 2020, pp. 727–735, doi: 10.1109/TrustCom50675.2020.00100. <https://arxiv.org/abs/2010.08521>
6. N. Moustafa, "New generations of Internet of Things datasets for cybersecurity applications based machine learning: TON_IoT datasets," in *Proc. eResearch Australasia Conference*, Brisbane, Australia, 2019. <https://conference.eresearch.edu.au/wp-content/uploads/2019/08/2019_eResearch_59_New-Generations-of-Internet-of-Things-Datasets-for-Cybersecurity.pdf>
7. N. Moustafa, "A systemic IoT-Fog-Cloud architecture for big-data analytics and cyber security systems: A review of fog computing," arXiv:1906.01055, 2019. <https://arxiv.org/abs/1906.01055>
8. J. Ashraf, M. Keshk, N. Moustafa, M. Abdel-Basset, H. Khurshid, A. D. Bakhshi, and R. R. Mostafa, "IoTBoT-IDS: A novel statistical learning-enabled botnet detection framework for protecting networks of smart cities," *Sustainable Cities and Society*, 2021, art. 103041. <https://www.sciencedirect.com/science/article/abs/pii/S2210670721003255>

### Dataset terms

The TON_IoT dataset (`train_test_network.csv`) is **not covered by the MIT License**. It was created by Dr Nour Moustafa and the UNSW Canberra Cyber team and remains under the dataset authors' terms:

- Free use for academic research purposes is granted.
- Commercial use requires permission from the dataset author.
- Anyone using the data must cite the papers listed in [Citation](#Citation).

Original source: <https://research.unsw.edu.au/projects/toniot-datasets>

## Rights

Copyright (c) [2026] [Information Science and Engineering, Dayananda Sagar Academy of Technology and Management, Bangalore, Karnataka, India, Department of Computing, College of Computing and Informatics, Universiti Tenaga Nasional, Selangor, Selangor, Malaysia] and the project authors
([Pavani Cherukuru], [Neesha Jothi], [K N Likith Gagan], [Lokesh Kumar N], [S N Nikhil], [S N Sathya Sai Prasad]). All rights reserved.

This repository is shared for viewing and academic evaluation only. No license is granted to copy, modify, distribute, or use this code or the methods it describes for any other purpose, and no license under any patent or patent application is granted or implied. The TON_IoT dataset remains under its authors' terms (see Dataset terms).
