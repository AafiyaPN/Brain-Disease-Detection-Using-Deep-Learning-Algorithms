# Brain-Disease-Detection-Using-Deep-Learning-Algorithms

Automated early diagnosis of Alzheimer’s disease using clinical data and baseline machine learning baselines. This repository contains exploratory data analysis, preprocessing and baseline classification code implemented in the Jupyter notebook `alzhmrs.ipynb`, and a project report `Alzhmrs.pdf`.

---

## Project Summary

The goal of this research is to build reliable, reproducible baseline models for early Alzheimer's disease detection using the OASIS dataset (clinical and MRI-derived tabular measures). The project demonstrates data cleaning and imputation strategies, exploratory data analysis (EDA), feature scaling, and training/evaluation of classical machine learning models. It also outlines next steps toward deep learning on MRI volumes and multimodal fusion.

Key repository files
- `alzhmrs.ipynb` — Main Jupyter notebook: EDA, preprocessing, modeling, and evaluation (runs in Google Colab).
- `Alzhmrs.pdf` — Project report with motivation, methods, results, and conclusions.

---

## Main points from the project report (summary)
1. Motivation & objective
   - Early detection improves patient outcomes; objective is an automated diagnostic aid for early Alzheimer’s detection.

2. Dataset & features
   - Data source: OASIS clinical/imaging dataset (longitudinal CSV used in the notebook).
   - Typical features used: Age, Sex (M/F), EDUC (years of education), SES, MMSE, CDR, eTIV, nWBV, ASF.

3. Preprocessing & handling repeated measures
   - Baseline-only analysis: the notebook filters to `Visit == 1` (one row per subject) to avoid longitudinal data leakage.
   - Encoding and label mapping: M/F coded to 0/1; Group labels consolidated to binary Demented (1) vs Nondemented (0).

4. Missing values
   - SES had missing values; the notebook demonstrates two strategies: drop rows with missing values and impute SES using the median SES grouped by EDUC. The imputation strategy preserves sample size while using data-driven grouping.

5. Exploratory data analysis
   - Visualizations compare the distributions of MMSE, Age, eTIV, nWBV, ASF and EDUC across groups; scatterplots (EDUC vs SES) and KDE plots are used to understand separability and guide feature usage.

6. Modeling
   - Classical ML baselines: Logistic Regression, SVM, Decision Tree, Random Forest, AdaBoost.
   - Feature scaling (MinMax) is applied before training.
   - Evaluation metrics: accuracy, recall, confusion matrix, ROC/AUC, classification report.

7. Conclusions & next steps
   - Tabular clinical features provide useful baseline predictive power.
   - Recommended next steps: train CNNs / 3D-CNNs on MRI volumes, multimodal fusion of clinical + imaging features, stronger cross-validation and hyperparameter tuning, explainability (saliency/Grad-CAM) and fairness checks.

---

## Notebook — detailed step-by-step explanation
The `alzhmrs.ipynb` notebook implements the end-to-end baseline workflow. Steps and rationale:

1. Environment & imports
   - Imports: pandas, numpy, seaborn, matplotlib, scikit-learn.
   - Notebook is prepared to run on Google Colab (Drive mount included) so large datasets can be accessed from Google Drive.

2. Data loading
   - The notebook reads the OASIS longitudinal CSV (for example: `oasis_longitudinal.csv`). Ensure the CSV is available at the path used in the notebook or change the path.

3. Baseline filtering
   - `df = df.loc[df['Visit'] == 1]` keeps only first visits to produce one record per subject and avoid learning temporal patterns incorrectly.

4. Encoding & label mapping
   - Convert `M/F` to numeric (F→0, M→1).
   - Consolidate `Group` values: 'Converted' becomes 'Demented', then map Demented→1, Nondemented→0.
   - Drop unneeded columns: `MRI ID`, `Visit`, `Hand`.

5. Exploratory Data Analysis (EDA)
   - Use stacked bar charts, KDE plots, and scatter plots to compare distributions between classes and check for separation in features.

6. Missing-value handling
   - The notebook identifies columns with missing values (SES). Two paths are demonstrated:
     a) Drop rows with missing values (complete-case analysis).
     b) Impute SES by filling NA with the median SES within each EDUC group: `df['SES'].fillna(df.groupby('EDUC')['SES'].transform('median'), inplace=True)`.
   - After imputation or dropping, verify class counts and data integrity.

7. Feature selection and scaling
   - Features used: `['M/F','Age','EDUC','SES','MMSE','eTIV','nWBV','ASF']`.
   - Split into train/test (scikit-learn `train_test_split`) and scale numeric features using `MinMaxScaler` fitted on training data.

8. Modeling & evaluation
   - Train baseline classifiers: LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier.
   - Evaluate using metrics: Accuracy, Recall, Confusion Matrix, ROC curve and AUC, Classification Report.
   - Note: LogisticRegression may show a ConvergenceWarning — increase `max_iter` or change solver if encountered.

9. Interpret results & visualizations
   - Use the confusion matrix and ROC/AUC to compare models. In clinical contexts prioritize sensitivity (recall) when missing a positive case is costly.

---

## How to reproduce / Quickstart

Prerequisites
- Python 3.8+
- Packages: pandas, numpy, seaborn, matplotlib, scikit-learn, jupyter (or run in Google Colab)

Install (example)
```
pip install pandas numpy seaborn matplotlib scikit-learn jupyter
```

Run in Google Colab (recommended)
1. Open `alzhmrs.ipynb` in Google Colab (Colab badge in the notebook points to an editable copy).
2. Mount Google Drive and upload `oasis_longitudinal.csv` to a path accessible from Colab, or change the read_csv path inside the notebook.
3. Run cells sequentially.

Run locally
1. Clone the repository:
```
git clone https://github.com/AafiyaPN/Brain-Disease-Detection-Using-Deep-Learning-Algorithms.git
cd Brain-Disease-Detection-Using-Deep-Learning-Algorithms
```
2. Start Jupyter and open `alzhmrs.ipynb`.
3. Edit the CSV path if needed and execute cells.

---

## Notes on dataset & ethics
- Data used in the notebook is from the OASIS dataset; follow OASIS data usage and citation guidelines.
- This research is for educational/experimental use; models must be clinically validated before any diagnostic use. Consider privacy, bias, and informed consent when working with clinical data.

---

## Next steps / recommended extensions
- Train 2D/3D CNNs on raw MRI volumes after preprocessing (skull-stripping, registration, intensity normalization).
- Multimodal models that combine clinical features with learned imaging features.
- Stronger cross-validation, hyperparameter search, and per-subject splitting for imaging experiments.
- Explainability (Grad-CAM, saliency) and fairness checks across demographic groups.

---

## Contributing
Contributions are welcome. Open issues or submit pull requests to propose fixes, new pre-processing scripts, or deep-learning experiments. Please include reproducible steps and a small sample to reproduce changes.

---

## License
This repository is released under the Apache License, Version 2.0. See `LICENSE` for full license text.

---

## Contact
Repository owner: AafiyaPN — https://github.com/AafiyaPN

