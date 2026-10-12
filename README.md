# OIBSIP: Oasis Infobyte Internship Projects

- **Intern:** Zareen Sabri
- **Track:** Data Science
- **Organisation:** Oasis Infobyte (Summer Internship Program)

This repository holds my task submissions for the Oasis Infobyte Summer Internship Program. I completed 3 of the 5 Data Science tasks (the minimum required).

| Task | Folder | Dataset | Best model | Main result |
|---|---|---|---|---|
| 1. Iris Flower Classification | [DataScience-Task1-IrisClassification](DataScience-Task1-IrisClassification) | Iris (scikit-learn) | Random Forest | 96.7% accuracy (10-fold CV) |
| 4. Spam Detection | [DataScience-Task4-SpamDetection](DataScience-Task4-SpamDetection) | SMS Spam Collection | Linear SVM | 98.3% accuracy, spam F1 = 0.935 |
| 5. Sales Prediction | [DataScience-Task5-SalesPrediction](DataScience-Task5-SalesPrediction) | Advertising | Polynomial Regression (degree 2) | R² = 0.987, MAE = 0.53 |

Each folder contains the notebook, its own README with results and screenshots, the chart screenshots (png), the results table (csv), an HTML copy of the executed notebook and, where needed, the dataset.

Task 4 uses SMS messages, not email, because the task's own sourcing guideline recommends the SMS Spam Collection.

## Tools
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, NLTK (stemmer), WordCloud, Jupyter Notebook.

## How to run
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```
Open a notebook and choose Kernel > Restart & Run All. All random splits use `random_state=42`, so results can be reproduced.
