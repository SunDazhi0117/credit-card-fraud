# Seven-Day Project Plan

Last updated: 2026-10-22. Days 1–6 and Day 7 local deliverables are complete. GitHub repository: SunDazhi0117/credit-card-fraud (private).

## Target and Working Style

Complete the first version within one week, targeting 2–3 hours per day. This replaces the earlier 4–6 week learning plan. Day 1 was completed using the Python script on 2026-09-24. Day 2 was completed in the script and English README on 2026-09-26.

Learn Python and pandas through the real project rather than completing a separate introductory course first. For every code block, explain its input, output, and purpose. Provide short steps, then interpret the output together.

The learner knows basic Python, including imports, print, lists, loops, and conditions, but is new to pandas. Recent practice covered creating a DataFrame from a dictionary, selecting columns, and boolean row filtering. Distinguish row selection from column selection explicitly; do not assume pandas knowledge or restart elementary Python lessons unnecessarily.

Use Chinese for teaching and English for project documentation and code comments. Keep supporting exercises and notes in `../learning-notes/`. Leave automatic code suggestions disabled unless requested otherwise.

## Schedule

| Day | Tasks | Deliverable |
| --- | --- | --- |
| 1 | Complete project structure and dependencies; obtain the ULB dataset; learn read_csv, shape, head, info, describe, missing-value checks, and class counts. | Dataset overview in `data_review.py` and English README; consolidate into the notebook later. |
| 2 | Reserve stratified validation and test sets before detailed EDA; explore class balance, Amount, Time, and selected features mainly on training data. | 4–6 meaningful plots with interpretations. |
| 3 | Build an always-normal DummyClassifier and ordinary logistic regression; fit scaling only on training data through a Pipeline. | Reproducible baseline training and prediction. |
| 4 | Evaluate on validation data using the confusion matrix, precision, recall, F1, Average Precision (AP), and ROC-AUC; plot PR and ROC curves. | Evaluation table and explanation of why accuracy can mislead. |
| 5 | Compare ordinary logistic regression, class-weighted logistic regression, and random forest using the same splits. | Validation comparison table. |
| 6 | Compare a small number of thresholds on validation data; select a final approach, then evaluate on the held-out test set. | Final results and false-positive/false-negative trade-off discussion. |
| 7 | Run the notebook from beginning to end; finish README and dependencies; prepare GitHub repository and a 90-second project introduction. | Reproducible first version and presentation. |

## Scope and Evaluation Rules

- Keep the existing `credit-card-fraud/` directory name.
- Planned structure: `data/`, `notebooks/fraud_detection.ipynb`, `images/`, `README.md`, `requirements.txt`, and `.gitignore`.
- Exclude raw data and the virtual environment from Git.
- Leave SMOTE, extensive tuning, a separate decision tree comparison, and handwritten logistic regression for later versions.
- Do not use test results to choose models or thresholds.
- Report AP explicitly; do not treat it as interchangeable with every calculation called PR-AUC.
- Label hypothetical business costs as assumptions, and never invent results or financial savings.
- Discuss limitations of the dataset and random splits; do not claim production readiness.

## Current Progress and Resume Point

- Project and learning-notes directories created.
- Python 3.13.15 installed through uv and verified.
- Project virtual environment created at `~/project/credit-card-fraud/.venv` and verified.
- pandas installed successfully in that environment, as confirmed by the learner's installation output.
- Basic pandas practice started in `../learning-notes/panda_df.py`.
- Downloaded and verified the original Kaggle creditcard.csv on 2026-09-23 (284,807 rows, 31 columns). Guided notebook analysis has not yet been completed.
- Created the starter notebook, images directory, and Day 1 requirements file. Installed ipykernel and nbformat; validated the notebook and its path setup. scikit-learn and matplotlib are now installed and used in the script.
- Dataset is available at `data/creditcard.csv`; ZIP integrity and CSV schema checks passed.

Day 1 is complete in `data_review.py`: the learner practiced CSV loading, shape, head, info, class counts and proportions, missing-value checks, and Amount.describe(). They explained large amounts pulling up the mean, the always-normal accuracy trap, and why Class is the prediction target. English dataset findings are saved in README.md. The notebook remains a starter and has not been completed or executed by the learner.

Teaching preference: ask the learner to write previously taught operations, inspect their file, and offer hints before giving corrected code. Explain new syntax with small examples first. Do not abruptly introduce complex paths or notebook workflows. Avoid marking unseen code as mastered or writing exercise answers for the learner.

Day 2 complete: stratified 60/20/20 splits with seed 5; groupby statistics; six saved figures covering class counts, full and local Amount distributions, Amount by class, Time by class, and V1 by class. The learner interpreted class imbalance, skewness, density comparisons, lack of clear Time separation, and V1 overlap. The full script was run successfully using a non-interactive plotting backend; findings and image links are in README.md. Notebook consolidation is still pending.

Days 3–4 are now complete; see the latest resume update below. Continue to explain new syntax before asking the learner to use it.

Activate the environment in a new terminal with:

```bash
source ~/project/credit-card-fraud/.venv/bin/activate
```

## Previous Resume Update — 2026-10-06

Days 3 and 4 are complete in `train_model.py` and the English README. The learner implemented standardization, ordinary logistic regression, predictions and probabilities, manual precision/recall/F1 and confusion counts, PR and ROC curves, AP, and ROC-AUC. They then replaced separate scaling/model calls with a Pipeline. A rerun verified identical predicted labels and numerically matching probabilities against the separate workflow.

The modeling script removes 1,081 extra exact rows in memory before the stratified 60/20/20 split (seed 5), leaving 283,726 records: 170,235 training, 56,745 validation, and 56,746 reserved test records. Exact train/validation row overlap is zero. The raw CSV and original Day 2 EDA remain unchanged; full-row deduplication does not rule out other dependencies.

Validation results: TN 56,639; FP 12; FN 36; TP 58; precision 0.8285714286; recall 0.6170212766; F1 0.7073170732; AP 0.7073182124; ROC-AUC 0.9720340705. The always-normal baseline has 99.83435% accuracy but misses all 94 fraud cases. Both `images/precision_recall_curve.png` and `images/roc_curve.png` are generated, and results and interpretations are documented in README.md.

The user wanted to finish Days 3–4 today and stop. Next session: Day 5, explain class weighting and compare balanced logistic regression, then random forest, using the same splits. Do not evaluate the test set or optimize thresholds yet. Neither of those additional models has been trained. Notebook consolidation, machine-specific path cleanup, final test evaluation, and GitHub presentation remain pending.

Teaching preference: concise Chinese explanations with English terminology; explain inputs, outputs, and purpose with concrete examples. Let the learner write code unless direct edits are requested. Avoid trivial quiz questions. Refresh metric meanings when needed, and distinguish completed code from the learner's understanding.

## Previous Resume Update — 2026-10-17

Day 5 is complete: ordinary logistic regression, class-weighted logistic regression, and random forest were trained and evaluated on the same validation set. The learner's completed run supplies all results, now recorded in the README comparison table. Balanced logistic regression: TP=83, FP=1158, FN=11, precision=0.0668815471, recall=0.8829787234, F1=0.1243445693, AP=0.6722676143, ROC-AUC=0.9769411218. Random forest (100 trees, seed 5, no class weights): TP=73, FP=10, FN=21, precision=0.8795180723, recall=0.7765957447, F1=0.8248587571, AP=0.8253136063, ROC-AUC=0.9504114780.

Next: Day 6. Explain business trade-offs and agree on an explicit validation selection criterion; compare a small number of thresholds on validation data, select and freeze the model/threshold, then evaluate the held-out test set once. Do not assume random forest is universally best from one metric. Do not invent business cost figures. The test set has not been evaluated. Existing PR/ROC plots are for ordinary logistic regression only. Notebook consolidation, path portability, final reporting, and GitHub publishing remain pending for Day 7.


## Previous Resume Update — 2026-10-20

Day 6 is complete. The learner compared three models at thresholds 0.2, 0.3, and 0.5 on validation data, using score >= threshold. Highest validation F1 among these nine candidates selected the already-trained random forest at threshold 0.3: TP=79, FP=15, FN=15, precision=recall=F1=0.8404255319. The model and threshold were fixed before test evaluation; the forest was not refitted on training plus validation data.

Final test results reported by the learner: 56,746 transactions, including 95 fraud cases; TP=71, FP=11, FN=24, TN=56640. Precision=0.8658536585, recall=0.7473684211, F1=0.8022598870, AP=0.7691039402, ROC-AUC=0.9296781122. Reporting reruns added metrics without changing the selected model or threshold. The test set is now exposed and must not guide further selection.

The English README records the nine validation candidates, selection criterion, final test metrics and confusion matrix, business trade-offs, and limitations. F1 was the selection criterion; no financial costs or savings were estimated. Existing PR/ROC images still show ordinary logistic regression on validation data.

Next: Day 7 only. Consolidate the notebook and verify execution, replace machine-specific paths, review requirements and Git exclusions, finish repository presentation, and prepare a 90-second project introduction. Do not add models or tune thresholds as part of this cleanup. Guide the learner step by step, explaining inputs, outputs, and purpose; avoid trivial quizzes. GitHub publishing remains pending.


## Latest Resume Update — 2026-10-22

Day 7 local preparation is complete. Both scripts use portable project-relative paths. Training output has named sections and a nine-row comparison table. Dependencies reflect the working environment, including nbclient for execution verification. The report notebook calls the scripts, includes input/output/purpose explanations, embeds eight plots, and exports four aggregate CSV files to results/. All nine code cells executed successfully in a fresh kernel; final confusion counts and F1 match the established result. See VALIDATION.md for verification limits.

The README includes installation, dataset acquisition, execution, results, and limitations. PROJECT_PRESENTATION.md contains an English introduction and Chinese study notes. A local Git repository on main has been initialized. Data, .venv, and environment files are excluded. GitHub publication is the only remaining delivery step; the connected account is SunDazhi0117, but no remote repository has been selected or created. Do not describe the project as uploaded yet.

The user authorized direct completion of housekeeping and presentation, and requested involvement only when needed for upload. No models or thresholds should be tuned using the exposed test set.


## Publication — 2026-10-22

Published the prepared first version to the private repository https://github.com/SunDazhi0117/credit-card-fraud. Dataset and virtual environment are excluded. All seven days of first-version delivery are complete. No model changes were made during publication.
