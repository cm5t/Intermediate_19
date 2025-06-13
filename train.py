import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load and preprocess the data
data = pd.read_csv("Cleaned_Family_Data.csv")

# Currency conversion
income_expense_columns = [
    'Total Household Income',
    'Total Income from Entrepreneurial Acitivites',
    'Total Food Expenditure',
    'Transportation Expenditure',
    'Housing and water Expenditure',
    'Medical Care Expenditure',
    'Education Expenditure'
]
data['Total Expenditure (USD)'] = data[['Total Food Expenditure', 'Transportation Expenditure', 'Housing and water Expenditure', 'Medical Care Expenditure', 'Education Expenditure']].div(56.14).sum(axis=1)
for col in income_expense_columns:
    data[col] = data[col] / 56.14
    data.rename(columns={col: col + " (USD)"}, inplace=True)

# Nett income
data['Nett Income (USD)'] = data['Total Household Income (USD)'] - data['Total Expenditure (USD)']

# Drop columns
toDrop = [
    'Total Food Expenditure (USD)', 'Transportation Expenditure (USD)', 'Housing and water Expenditure (USD)',
    'Medical Care Expenditure (USD)', 'Education Expenditure (USD)', 'Total Income from Entrepreneurial Acitivites (USD)',
    'Tenure Status', 'Type of Building/House', 'Household Head Marital Status',
    'Household Head Highest Grade Completed', 'Main Source of Income'
]
data = data[data['Total Household Income (USD)'] > 0]
data = data[data['Household Head Age'] >= 22]
data.drop(columns=toDrop, inplace=True)

# Add features
data['Household Head Job or Business Indicator'] = data['Household Head Job or Business Indicator'].map({'With Job/Business': 1, "No Job/Business": 0})
data['Nett income per person in household'] = data['Nett Income (USD)'] / data['Total Number of Family members']
data['Years left till retirement'] = (65 - data['Household Head Age']).clip(lower=0)
data['Saving amount'] = data['Nett Income (USD)'] * 0.65
data['Expected future Expenditure'] = data['Total Expenditure (USD)'] * ((1 + np.random.uniform(0.045, 0.05, size=len(data))) ** data['Years left till retirement'])
data['Expected income'] = data['Total Household Income (USD)'] * ((1 + np.random.uniform(0.04, 0.045, size=len(data))) ** data['Years left till retirement'])
data['Expected nett income'] = data['Expected income'] - data['Expected future Expenditure']
data['Expected saving amount'] = data['Expected nett income'] * 0.65

def savingurgency(row):
    if row['Nett Income (USD)'] > row['Expected future Expenditure']:
        return 0
    elif row['Total Household Income (USD)'] > row['Expected future Expenditure']:
        return 1
    elif (row['Expected income'] * 0.75) > row['Expected future Expenditure']:
        return 2
    elif row['Expected income'] > row['Expected future Expenditure']:
        return 3
    elif row['Total Household Income (USD)'] > row['Total Expenditure (USD)']:
        return 4
    else:
        return 5

data['Saving Urgency'] = data.apply(savingurgency, axis=1)
data = data.replace([np.inf, -np.inf], 0)

# Prepare X and y
X = data.drop(columns=['Saving Urgency'])
y = data['Saving Urgency']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=69)
model = RandomForestClassifier(n_estimators=100, random_state=69)
model.fit(X_train, y_train)
joblib.dump(list(X.columns), 'input_columns.pkl')
# Export model and scaler
joblib.dump(model, 'saving_model.pkl')
joblib.dump(scaler, 'saving_scaler.pkl')
print("✅ Model and scaler saved.")
