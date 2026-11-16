# Student Performance Classifier

A clean, reproducible machine-learning pipeline that predicts a student's **GPA class** — Low
(`GPA ≤ 2.5`), Medium (`2.5 < GPA ≤ 3.5`), High (`GPA > 3.5`) — from study habits, and then does the
part most projects skip: a disciplined look at *which* class the model gets wrong and *why*.

The interesting result is not the accuracy, it is that the **Medium** class is where every model
struggles — it is bounded on both sides and behaviourally overlapping with its neighbours. The pipeline
is built around diagnosing that, rather than around chasing a leaderboard.

Coursework project, cleaned up for publication. Code comments are in Russian (original), README in
English.

## Dataset

`global_university_students_performance_habits_10000.csv` — 10 000 synthetic student records with
habits (study hours, sleep, attendance, AI-tool usage, etc.) and academic outcomes. It ships with the
repository, so the whole pipeline runs out of the box.

Three columns are **dropped as leakage**: `student_id`, `GPA` (the source of the label) and
`final_exam_score` (an outcome, not a predictor). Keeping them would produce a great score and a
useless model — removing them is the first honest decision in the pipeline.

## Pipeline

Six numbered stages, each a runnable script, each writing what the next one needs:

| Stage | Script | What happens |
|---|---|---|
| 1 | `src/01_preprocessing.py` | bin GPA into three classes, drop leakage columns, fill categorical gaps, `LabelEncoder` encoding, save `processed_data.pkl` |
| 2 | `src/02_baseline_models.py` | Logistic Regression, Decision Tree and Random Forest baselines, class-balanced, macro-F1 + classification report |
| 3 | `src/03_tune_random_forest.py` | `RandomizedSearchCV` over `n_estimators` / `max_depth` / `min_samples_leaf` / `max_features` / `class_weight`, 25 draws, 5-fold `StratifiedKFold`, `scoring='f1_macro'` |
| 4 | `src/04_xgboost_training.py` | XGBoost `multi:softprob`, 500 trees, depth 5, learning-rate sweep `0.03/0.05/0.07` with early stopping on a validation split, best model persisted to `best_xgb_model.pkl` |
| 5 | `src/05_feature_importance.py` | top-15 gain-based feature importances |
| 6 | `src/06_evaluation.py` | confusion-matrix heatmap plus a per-class error breakdown |

### Why the split is three-way

Stages 2–3 evaluate on a held-out test set. Stage 4 needs a validation set for early stopping, so it
splits the training portion again (85/15) and touches the test set exactly once, at the end. That is
the difference between a number and an estimate.

### The Medium-class analysis

`src/06_evaluation.py` does not stop at the confusion matrix. It reports, explicitly:

* how many true-Medium students were misclassified, and **what they were predicted as**;
* how many students were *incorrectly* predicted Medium, and **what they actually were**.

That two-way view is what turns "macro-F1 is mediocre" into "Medium is a boundary class and the errors
are symmetric" — an actionable finding rather than a metric.

## Layout

```
ml-student-performance/
├─ src/
│  ├─ 01_preprocessing.py       # cleaning, labelling, encoding
│  ├─ 02_baseline_models.py     # LR / DT / RF baselines
│  ├─ 03_tune_random_forest.py  # randomised hyperparameter search
│  ├─ 04_xgboost_training.py    # XGBoost + early stopping + persistence
│  ├─ 05_feature_importance.py  # top-15 importances
│  └─ 06_evaluation.py          # confusion matrix + per-class error analysis
├─ exploration/                 # earlier exploration scripts, kept for the trail
│  ├─ prepare_data.py
│  ├─ data_overview.py
│  └─ check_correlation.py
├─ global_university_students_performance_habits_10000.csv
└─ requirements.txt
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the stages in order **from the repository root** (the scripts read and write files by relative
path):

```bash
python src\01_preprocessing.py
python src\02_baseline_models.py
python src\03_tune_random_forest.py
python src\04_xgboost_training.py
python src\05_feature_importance.py
python src\06_evaluation.py
```

Every stage prints its metrics to stdout; stages 5 and 6 also open Matplotlib/Seaborn figures.

## Design notes

* **Macro-F1, not accuracy.** With three classes of unequal size, accuracy rewards ignoring the
  minority class. Macro-F1 treats all three as equally important, which is the right objective when a
  misclassified student is the cost.
* **Class weights everywhere.** `balanced` / `balanced_subsample` is set on every baseline; without it
  the trees simply predict the majority class.
* **Reproducibility.** `random_state=42` is threaded through splitting, model construction, the
  randomised search and the CV splitter — the numbers you see are the numbers anyone else gets.
* **No test-set peeking.** Hyperparameters are selected on validation folds, not on the test split.

## Notes and limitations

* The dataset is synthetic, so the learned relationships are properties of the generator, not of real
  students. The methodology transfers; the coefficients do not.
* Binary/categorical encodings are ordinal-encoded (`LabelEncoder`) rather than one-hot — fine for
  tree models, worth revisiting for linear ones.
* `processed_data.pkl` and `best_xgb_model.pkl` are build artefacts and are not versioned; they are
  recreated by stages 1 and 4.
* `exploration/` contains superseded versions of the preprocessing code, kept to show the analysis
  trail rather than as a maintained entry point.

## Stack

Python 3.10+ · pandas · NumPy · scikit-learn · XGBoost · Seaborn · Matplotlib · joblib
