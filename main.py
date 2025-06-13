import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to the Building Blocs 2025 I19 Streamlit App!")

st.markdown(
    """
    Our group decided to create a Machine Learning Model that predicts the urgency of saving for families based on their income, expenditure and other data. The model is trained on a dataset that includes various financial metrics, such as household income, food expenditure, transportation costs, and more.
    ### We used a Random Forest Algorithm, with 100s of if-else statements to predict whether the family requires saving.
    This can be applied in the Singapore Context, and worldwide as well!
    ### The saving tiers are as follows:
    ### 0: "✅ No urgency – You're doing great!",
    ### 1: "🟢 Low urgency – Consider saving a little more.",
    ### 2: "🟡 Moderate urgency – Watch your expenses.",
    ### 3: "🟠 High urgency – Plan carefully.",
    ### 4: "🔴 Very high urgency – Income just covers needs.",
    ### 5: "🚨 Critical – You need to reduce expenses and quickly increase income!"
"""
)
