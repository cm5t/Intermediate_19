# 💸 Saving Urgency Predictor – Building Blocs 2025 (I19)
# By Dylan, Chandi, Kanish and Adithya
## This is a Streamlit-based web app that predicts how urgently a family needs to save based on income and expenditure data. It uses machine learning (Random Forest + Linear Regression) and provides actionable financial advice for Singaporean households.

---

## 🧠 Machine Learning Techniques Used


- **Random Forest Classifier** with 500 trees
- **Feature Importance** graph
- Visualized **Decision Tree**
- **Linear Regression** as exploratory analysis
- Trained on real Filipino household expenditure data
- Suggests subsidies and support based on Singapore government schemes like the MOE FAS and the CHAS Blue/Orange.

---

### Importing Libraries
### You need to install Python version 3.8+. Then, type 'pip install (streamlit/pandas/numpy/scikit-learn/matplotlib/joblib)' (install all of them). 

### 1. Clone the repository

```bash
git clone https://github.com/cm5t/buildingblocs25.git
cd buildingblocs25
```

Make sure Cleaned_Family_Data.csv is in the same folder as main.py, and that there is a folder called 'pages'.

# Project Structure:
```bash
├── main.py                      # Homepage for Streamlit app
├── 1-machinelearning.py        # ML model training and explanation
├── 2-douneed2save.py           # Interactive prediction UI
├── Cleaned_Family_Data.csv     # Cleaned dataset for model training
├── saving_model.pkl            # Trained Random Forest model
├── saving_scaler.pkl           # Scaler for input data
├── input_columns.pkl           # Ordered input features
└── README.md                   # Project documentation
```

## To run the project, type 'streamlit run main.py'.

You can navigate to the other pages via the sidebar:
main.py: Overview and intro
1-machinelearning.py: Data, training, and decision tree visualisation
2-douneed2save.py: Input form and saving advice