import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
import streamlit as st


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

data.drop(columns=toDrop, inplace=True)
data['Household Head Job or Business Indicator'] = data['Household Head Job or Business Indicator'].map({'With Job/Business': 1, "No Job/Business": 0})


st.dataframe(data.head(100))