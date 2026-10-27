# Titanic Survival Prediction: 5-Model Comparison

Comparing five machine learning classifiers on the Titanic dataset to predict whether a passenger survived.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tasleemkhan-code/model-comparison-ml/blob/main/titanic_model_comparison.ipynb)

## Models Compared

1. Support Vector Machine (SVM)
2. Logistic Regression
3. K-Nearest Neighbors (KNN)
4. Decision Tree
5. Naive Bayes

## Dataset

- Source: Titanic dataset from Seaborn (`sns.load_dataset("titanic")`)
- Features: `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`, `alone`
- Target: `survived` (0 = did not survive, 1 = survived)

## Workflow

1. Load the dataset and drop redundant columns (`alive`, `embark_town`, `adult_male`, `who`, `class`, `deck`)
2. Fill missing `age` values with the median
3. Encode categorical features (`sex`, `embarked`, `alone`) using `LabelEncoder`
4. Split data into train/test sets (80/20, `random_state=42`)
5. Train all five models on the same split
6. Evaluate using accuracy, precision, recall and F1-score
7. Compare the models side by side

## Results ✅

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **SVM** | **0.8156** | **0.8060** | 0.7297 | **0.7660** |
| Logistic Regression | 0.7989 | 0.7714 | 0.7297 | 0.7500 |
| KNN | 0.7989 | 0.7714 | 0.7297 | 0.7500 |
| Decision Tree | 0.7933 | 0.7403 | 0.7703 | 0.7550 |
| Naive Bayes | 0.7765 | 0.7125 | 0.7703 | 0.7403 |

![Model Comparison](model_comparison.png)

## Key Findings

- **SVM performed best** on accuracy (81.56%), precision (80.60%) and F1-score (76.60%).
- **Logistic Regression and KNN** gave identical scores (79.89% accuracy), a close second.
- **Decision Tree and Naive Bayes** had the highest recall (77.03%), meaning they caught more of the actual survivors, but at the cost of lower precision.
- The gap between the best and worst model is only about 4 percentage points, and the test set is small (~179 passengers), so the differences between models are modest. A larger evaluation (such as cross-validation) would give a more reliable ranking.

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## How to Run

**Google Colab:** click the "Open in Colab" badge above and run all cells.

**Locally:**
```bash
git clone https://github.com/tasleemkhan-code/model-comparison-ml.git
cd model-comparison-ml
pip install pandas numpy seaborn matplotlib scikit-learn jupyter
jupyter notebook titanic_model_comparison.ipynb
```

## 👤 Author

**Tasleem Khan**
BCA (Hons.) Student, Cloud & AI/ML
GitHub: [@tasleemkhan-code](https://github.com/tasleemkhan-code)
