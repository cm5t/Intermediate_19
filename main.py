import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("saving_model.pkl")
scaler = joblib.load("saving_scaler.pkl")

st.title("💰 Should I Save or Not?")
st.write("Enter your household information below to check saving urgency and advice.")

# inputs
income = st.number_input("Total Household Income (PHP)", min_value=0.0, value=100000.0)
entrepreneur_income = st.number_input("Income from Entrepreneurial Activities (PHP)", min_value=0.0)
food = st.number_input("Food Expenditure (PHP)", min_value=0.0)
transport = st.number_input("Transportation Expenditure (PHP)", min_value=0.0)
housing = st.number_input("Housing and Water Expenditure (PHP)", min_value=0.0)
medical = st.number_input("Medical Care Expenditure (PHP)", min_value=0.0)
education = st.number_input("Education Expenditure (PHP)", min_value=0.0)

family_members = st.number_input("Total Number of Family Members", min_value=1, value=4)
members_0_4 = st.number_input("Members with age less than 5 years old", min_value=0, max_value=20, value=0)
members_5_17 = st.number_input("Members with age 5 - 17 years old", min_value=0, max_value=20, value=1)
employed_members = st.number_input("Total number of family members employed", min_value=0, max_value=20, value=2)

head_age = st.number_input("Household Head Age", min_value=18, value=40)
job_indicator = st.selectbox("Does the Household Head have a Job/Business?", ["With Job/Business", "No Job/Business"])

# when button pressed
if st.button("Predict Saving Urgency"):

    # Convert filipino money to USD
    def to_usd(php): return php / 56.14

    income_usd = to_usd(income)
    entrepreneur_income_usd = to_usd(entrepreneur_income)
    food_usd = to_usd(food)
    transport_usd = to_usd(transport)
    housing_usd = to_usd(housing)
    medical_usd = to_usd(medical)
    education_usd = to_usd(education)

    total_expenditure_usd = food_usd + transport_usd + housing_usd + medical_usd + education_usd
    nett_income = income_usd - total_expenditure_usd
    years_left = max(0, 65 - head_age)
    job_value = 1 if job_indicator == "With Job/Business" else 0

    nett_income_per_person = nett_income / family_members
    saving_amount = nett_income * 0.65
    future_exp = total_expenditure_usd * ((1 + 0.0475) ** years_left)
    expected_income = income_usd * ((1 + 0.0425) ** years_left)
    expected_nett = expected_income - future_exp
    expected_saving = expected_nett * 0.65

    # data input
    input_data = pd.DataFrame([{
        'Total Household Income (USD)': income_usd,
        'Total Income from Entrepreneurial Acitivites': entrepreneur_income_usd,
        'Total Food Expenditure': food_usd,
        'Transportation Expenditure': transport_usd,
        'Housing and water Expenditure': housing_usd,
        'Medical Care Expenditure': medical_usd,
        'Education Expenditure': education_usd,
        'Total Number of Family members': family_members,
        'Members with age less than 5 year old': members_0_4,
        'Members with age 5 - 17 years old': members_5_17,
        'Total number of family members employed': employed_members,
        'Tenure Status': 0,  # dropped later
        'Type of Building/House': 0,  # dropped later
        'Household Head Age': head_age,
        'Household Head Marital Status': 0,  # dropped later
        'Household Head Highest Grade Completed': 0,  # dropped later
        'Household Head Job or Business Indicator': job_value,
        'Main Source of Income': 0,  # dropped later
        'Nett Income (USD)': nett_income,
        'Nett income per person in household': nett_income_per_person,
        'Years left till retirement': years_left,
        'Saving amount': saving_amount,
        'Expected future Expenditure': future_exp,
        'Expected income': expected_income,
        'Expected nett income': expected_nett,
        'Expected saving amount': expected_saving,
    }])

    input_data["Total Expenditure (USD)"] = total_expenditure_usd

    # Drop unused (as done in training)
    input_data.drop(columns=[
        'Total Food Expenditure', 'Transportation Expenditure', 'Housing and water Expenditure',
        'Medical Care Expenditure', 'Education Expenditure', 'Total Income from Entrepreneurial Acitivites',
        'Tenure Status', 'Type of Building/House', 'Household Head Marital Status',
        'Household Head Highest Grade Completed', 'Main Source of Income'
    ], inplace=True)

    # Scale and predict
    expected_columns = joblib.load("input_columns.pkl")
    input_data = input_data[expected_columns]
    X_scaled = scaler.transform(input_data)
    prediction = model.predict(X_scaled)[0]

    # output
    st.subheader("🔍 Saving Urgency Level:")
    levels = {
        0: "✅ No urgency – You're doing great!",
        1: "🟢 Low urgency – Consider saving a little more.",
        2: "🟡 Moderate urgency – Watch your expenses.",
        3: "🟠 High urgency – Plan carefully.",
        4: "🔴 Very high urgency – Income just covers needs.",
        5: "🚨 Critical – You need to save urgently!"
    }
    st.write(f"**Level {prediction}:** {levels.get(prediction)}")

    # subsidies and bursaries etc.
    st.subheader("💡 Financial Advice (Singapore-based)")

    income_php = income_usd * 56.14
    if income_php < 30000:
        st.markdown("- **ComCare Assistance** – For low-income families.")
        st.markdown("- **MOE FAS** – School subsidies for kids.")
        st.markdown("- **CHAS Blue** – Medical subsidies.")
    elif income_php < 60000:
        st.markdown("- **CHAS Orange** – Healthcare help.")
        st.markdown("- **GST Vouchers, U-Save** – Utilities and cash rebates.")
    else:
        st.markdown("- You're not eligible for income-based schemes, but explore CPF reliefs and retirement supplements.")

    if head_age >= 55:
        st.markdown("- **Silver Support Scheme** – Quarterly payouts for elderly.")
    elif head_age <= 25:
        st.markdown("- **MOE Bursaries, CDC Awards** – For students in low/mid-income households.")

    st.markdown("---")
    st.caption("Subsidy advice is based on Singapore government policies as of 2025.")
