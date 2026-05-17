import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="LoanFlow - Loan Approval Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    return joblib.load('loan_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ Model file 'loan_model.pkl' not found. Please upload it to this directory.")
    st.stop()

# ============================================================================
# HEADER
# ============================================================================
st.title("💰 LoanFlow")
st.markdown("**AI-Powered Loan Approval Predictor**")
st.divider()

# ============================================================================
# SIDEBAR - APPLICATION INPUTS
# ============================================================================
with st.sidebar:
    st.header("📋 Applicant Information")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", min_value=18, max_value=75, value=35, step=1)
    with col2:
        employment_years = st.slider("Employment Years", min_value=0, max_value=50, value=5, step=1)
    
    annual_income = st.number_input(
        "Annual Income ($)",
        min_value=10000,
        max_value=1000000,
        value=75000,
        step=5000
    )
    
    loan_amount = st.number_input(
        "Requested Loan Amount ($)",
        min_value=5000,
        max_value=1000000,
        value=200000,
        step=10000
    )
    
    credit_score = st.slider(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=10
    )
    
    # Auto-calculate Debt-to-Income ratio
    estimated_monthly_debt = st.number_input(
        "Estimated Monthly Debt ($)",
        min_value=0,
        max_value=50000,
        value=2000,
        step=100
    )
    
    monthly_income = annual_income / 12
    debt_to_income = estimated_monthly_debt / monthly_income if monthly_income > 0 else 0
    
    st.info(f"📊 Debt-to-Income Ratio: {debt_to_income:.2%}")

# ============================================================================
# MAIN CONTENT - PREDICTION
# ============================================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Analysis")
    
    # Prepare input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Annual_Income': [annual_income],
        'Credit_Score': [credit_score],
        'Loan_Amount': [loan_amount],
        'Employment_Years': [employment_years],
        'Debt_to_Income': [debt_to_income]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # Display decision
    if prediction == 1:
        st.success("✅ APPROVED")
        approval_confidence = probability[1]
    else:
        st.error("❌ REJECTED")
        approval_confidence = probability[0]
    
    # Confidence meter
    st.metric(
        label="Decision Confidence",
        value=f"{max(probability)*100:.1f}%"
    )

with col2:
    st.header("⚡ Quick Stats")
    st.metric(label="Age", value=f"{age} yrs")
    st.metric(label="Credit", value=credit_score)
    st.metric(label="Income", value=f"${annual_income:,.0f}")

# ============================================================================
# RECOMMENDATION SECTION
# ============================================================================
st.divider()
st.header("💡 Key Factors")

# Risk assessment
factors = []
if credit_score < 600:
    factors.append(("❌", "Low Credit Score (< 600) - High Risk"))
else:
    factors.append(("✅", "Good Credit Score"))

if annual_income < 40000:
    factors.append(("❌", "Low Income (< $40k) - Risk Factor"))
else:
    factors.append(("✅", "Sufficient Annual Income"))

if debt_to_income > 0.5:
    factors.append(("❌", "High Debt-to-Income Ratio (> 50%) - Risk"))
else:
    factors.append(("✅", "Healthy Debt-to-Income Ratio"))

if employment_years < 2:
    factors.append(("⚠️", "Limited Employment History"))
else:
    factors.append(("✅", "Stable Employment History"))

# Display factors
for icon, factor in factors:
    st.write(f"{icon} {factor}")

# ============================================================================
# BOTTOM SECTION - MORE INFO
# ============================================================================
st.divider()

with st.expander("📈 How This Works"):
    st.write("""
    **LoanFlow** uses a **Random Forest Machine Learning Model** trained on 500 loan applications.
    
    The model analyzes:
    - **Credit Score** - Payment history and creditworthiness
    - **Annual Income** - Ability to repay the loan
    - **Debt-to-Income Ratio** - Current financial obligations
    - **Employment History** - Job stability
    - **Age & Loan Amount** - Risk profile
    
    **Note:** This is a demo. Real loan decisions require human review and compliance with banking regulations.
    """)

with st.expander("💾 Sample Test Data"):
    st.write("Try these profiles:")
    
    test_cases = {
        "✅ Strong Applicant": {
            "Age": 40, "Annual_Income": 150000, "Credit_Score": 780,
            "Loan_Amount": 200000, "Employment_Years": 15, "Debt_to_Income": 0.25
        },
        "⚠️ Medium Risk": {
            "Age": 28, "Annual_Income": 55000, "Credit_Score": 620,
            "Loan_Amount": 100000, "Employment_Years": 3, "Debt_to_Income": 0.45
        },
        "❌ High Risk": {
            "Age": 35, "Annual_Income": 35000, "Credit_Score": 520,
            "Loan_Amount": 300000, "Employment_Years": 1, "Debt_to_Income": 0.75
        }
    }
    
    for case_name, values in test_cases.items():
        st.write(f"\n**{case_name}**")
        st.json(values)

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.caption("🔒 Your data is processed locally and never stored. © 2024 LoanFlow Demo")
