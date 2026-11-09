# Dataset and local setup

Source: [BlastChar, Telco Customer Churn, Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

Filename: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

Recorded source SHA-256:
`88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`

The original raw CSV and its two full-record split copies are not included in this portfolio package. Obtain the dataset from its source under the source's applicable terms. The data dictionary is at `outputs/tables/development_data_dictionary.csv`; the definition review is here. Customer IDs in the saved manifests and prediction evidence belong to the supplied dataset, not to a live customer system.

## Restore the exact original experiment (project owner)

From the repository root, run:

```text
python scripts/restore_project_data.py "PATH_TO_YOUR_ORIGINAL_PROJECT_FOLDER"
```

Point to the folder containing the original `data`, `models`, and `notebooks` directories, not its parent. The helper copies only the three omitted CSVs, validates them against the recorded fingerprints, and refuses to replace different existing files. It does not train models or change records. These restored files are ignored by Git.

## Other readers

The notebook outputs and results can be inspected without the CSVs. To reproduce this exact archived run, you need the original split files as well as the source CSV; the fingerprints document their identities. Alternatively, download the source and conduct a new experiment in a separate working directory, generating a fresh complete split bundle with Notebook 01. Do not mix regenerated files with historical frozen records or represent a new experiment as verification of the archived one. Fully automated source-only reconstruction is not supplied by this portfolio copy.
