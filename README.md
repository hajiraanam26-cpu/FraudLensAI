\# FraudLens AI — Explainable Fraud Detection System



An AI-powered fraud detection dashboard built using Python, Scikit-learn, Streamlit, and SHAP. FraudLens AI predicts potentially fraudulent transactions, explains model decisions, and provides analytical tools to evaluate fraud risk and investigation costs.



\## Key Features



\- \*\*Fraud Prediction:\*\* Predicts whether a transaction is potentially fraudulent.

\- \*\*Risk Classification:\*\* Categorizes transactions into low, medium, and high risk.

\- \*\*Explainable AI:\*\* Uses SHAP to explain individual predictions and global feature importance.

\- \*\*Model Evaluation:\*\* Displays accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.

\- \*\*Model Comparison:\*\* Supports comparison of machine learning model performance.

\- \*\*Threshold Optimization:\*\* Explores how classification thresholds affect model performance.

\- \*\*Cost-Based Analysis:\*\* Estimates missed-fraud losses and investigation costs using adjustable assumptions.

\- \*\*Trust Report:\*\* Provides model performance and explainability insights.



\## Technology Stack



\- Python

\- Streamlit

\- Pandas and NumPy

\- Scikit-learn

\- SHAP

\- Plotly

\- Joblib



\## Project Structure



```text

FraudLensAI/

│

├── app.py

├── requirements.txt

├── README.md

├── .gitignore

│

├── data/

│   └── transactions.csv

│

├── models/

│   ├── fraud\_model.joblib

│   └── metadata.json

│

├── pages/

│   ├── 1\_Analytics.py

│   ├── 2\_Fraud\_Alerts.py

│   ├── 2\_Model\_Evaluation.py

│   ├── 3\_Fraud\_Analytics.py

│   ├── 4\_Explainable\_AI.py

│   ├── 5\_Model\_Performance.py

│   ├── 5\_Transaction\_History.py

│   ├── 6\_Real\_Time\_Fraud\_Alerts.py

│   ├── 7\_Model\_Comparison.py

│   ├── 8\_Threshold\_Optimization.py

│   ├── 9\_Cost\_Based\_Fraud\_Analysis.py

│   └── Model\_Trust\_Report.py

│

├── reports/

│   ├── metrics.json

│   └── test\_predictions.csv

│

├── src/

│   ├── generate\_data.py

│   ├── generate\_reports.py

│   └── train\_model.py

│

└── utils/

&#x20;   ├── helpers.py

&#x20;   └── \_\_init\_\_.py

```



\## Installation and Setup



\### 1. Clone the repository



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

cd FraudLensAI

```



\### 2. Create a virtual environment



```bash

python -m venv .venv

```



\### 3. Activate the environment



Windows PowerShell:



```powershell

.venv\\Scripts\\Activate.ps1

```



\### 4. Install dependencies



```bash

pip install -r requirements.txt

```



\### 5. Run the application



```bash

streamlit run app.py

```



\## Model Validation



The project uses a stratified training, validation, and test split, along with 5-fold cross-validation on the training data.



Latest held-out test results:



| Metric | Score |

|---|---:|

| Accuracy | 80.53% |

| Precision | 68.97% |

| Recall | 73.47% |

| F1-score | 71.15% |

| ROC-AUC | 86.44% |

| Average Precision | 74.68% |



Selected model: Logistic Regression



Mean cross-validation F1-score: 71.59%



Cross-validation standard deviation: 1.03 percentage points.



The final test set is held out from model selection.



\## Dataset



The project currently uses a synthetic demonstration dataset containing 5,000 transactions.



Features include:



\- Transaction amount

\- Transaction hour

\- Distance from usual transaction location

\- Failed transaction attempts

\- International transaction indicator



Target variable: `is\_fraud`



\*\*Disclaimer:\*\* This is an educational demonstration project using synthetic data. Its results do not establish real-world banking fraud detection performance.



\## Explainability



SHAP (SHapley Additive exPlanations) is used to understand how input features influence the model's predictions. These explanations describe model behavior and should not be interpreted as proof of actual fraud causes.



\## Future Improvements



\- Test on a suitable real-world anonymized dataset.

\- Explore additional fraud detection algorithms.

\- Add monitoring for model drift.

\- Improve deployment and automated testing.



\## Author



Developed as an academic AI and Machine Learning project.



