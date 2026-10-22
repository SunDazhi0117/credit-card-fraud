# Local Verification

Verified on 2026-10-22 using the project's Python 3.13 virtual environment on macOS.

- Parsed both Python scripts successfully.
- Executed all nine code cells of `notebooks/fraud_detection.ipynb` from a fresh kernel, with the working directory set to `notebooks/`.
- The notebook executed both scripts, regenerated eight figures, and exported four aggregate result CSVs without cell errors.
- Confirmed nine model/threshold candidates in the exported comparison table.
- Confirmed final test confusion counts match the previously recorded experiment: TN=56640, FP=11, FN=24, TP=71; F1=0.8022598870056498.
- `python -m pip check` reports no broken requirements in the existing environment.
- Verified Git exclusions for the CSV, virtual environment, and local environment files.

No new models, thresholds, or split choices were introduced. Re-execution was for verification and report generation, not test-set tuning.

Limits: a clean dependency installation on a second machine or operating system has not been tested. Notebook output includes aggregate results and a five-row preview of the anonymized public dataset, not the full CSV. The notebook uses the Agg plotting backend and displays saved PNGs; standalone scripts retain interactive plotting.

To repeat the check interactively, follow README setup instructions and Run All in the notebook. Select the project virtual environment kernel. The code must run from the project or notebooks directory; the CSV is downloaded separately.
