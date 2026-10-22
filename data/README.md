# Data Directory

Download the original [ULB Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). A Kaggle account may be required. Extract the download and place `creditcard.csv` directly in this directory:

```text
data/creditcard.csv
```

The original file has 284,807 rows and 31 columns: Time, V1–V28, Amount, and Class. Class 1 denotes fraud and Class 0 legitimate transactions. Follow the source's current dataset terms. The CSV and downloaded archives are not included in this repository.

The local dataset was obtained on 2026-09-23 and checked for ZIP integrity and the expected schema. The training script removes exact duplicates in memory; it never overwrites the original CSV. See the main README for the distinction between original EDA and deduplicated modeling data.
