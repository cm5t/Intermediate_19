import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


st.title("Dataset - Filipino Family Income and Expenditure")

data = pd.read_csv("buildingblocs25/Cleaned_Family_Data.csv")
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

for col in income_expense_columns:
    data.rename(columns={col: col + " (USD)"}, inplace=True)

data['Nett Income (USD)'] = data['Total Household Income (USD)'] - data['Total Expenditure (USD)']

toDrop = [
    'Total Food Expenditure (USD)',
    'Transportation Expenditure (USD)',
    'Housing and water Expenditure (USD)',
    'Medical Care Expenditure (USD)',
    'Education Expenditure (USD)',
    'Total Income from Entrepreneurial Acitivites (USD)',
    'Tenure Status',
    'Type of Building/House',
    'Household Head Marital Status',
    'Household Head Highest Grade Completed',
    'Main Source of Income'
]

np.random.seed(69)

# to ensure that there are no insane values/outliers in the data
data = data[data['Total Household Income (USD)'] > 0]
data = data[data['Household Head Age'] >= 22]

data.drop(columns=toDrop, inplace=True)

# adding more metrics
data['Household Head Job or Business Indicator'] = data['Household Head Job or Business Indicator'].map({'With Job/Business': 1, "No Job/Business": 0})
data['Nett income per person in household'] = data['Nett Income (USD)'] / data['Total Number of Family members']
data['Years left till retirement'] = (65 - data['Household Head Age']).clip(lower=0)

# assume that you are saving 65% of nett income
data['Saving amount'] = data['Nett Income (USD)'] * 0.65

# assume inflation > salary growth not by much
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


# avoid nans
data = data.replace([np.inf, -np.inf], 0)

st.dataframe(data.head(1000))


## ML!!

scaler = StandardScaler()
X = data.drop(columns=['Saving Urgency'])
y = data['Saving Urgency']
Scaled_X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(Scaled_X, y, test_size=0.4, random_state=69)

dectree = RandomForestClassifier(n_estimators=100, random_state=69)
dectreefitted = dectree.fit(X_train, y_train)

y_pred = dectreefitted.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


plt.figure(figsize=(10, 6))
plt.bar(range(len(dectreefitted.feature_importances_)), dectreefitted.feature_importances_)
plt.title('Feature Importances for Saving Urgency Prediction')
st.pyplot(plt)

plt.figure(figsize=(20,10))
st.title("The decision tree!")
st.header("Accuracy: " + str(accuracy_score(y_test, y_pred)))
onetree = dectree.estimators_[0]
tree.plot_tree(onetree, feature_names=X.columns.tolist())
st.pyplot(plt)