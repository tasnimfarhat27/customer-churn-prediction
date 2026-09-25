import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Page Configuration & Pastel Corporate Styling
st.set_page_config(
    page_title='Customer Churn Prediction System', page_icon='📊', layout='wide'
)

st.markdown(
    """
    <style>
    /* Global Background & Font styling */
    .stApp {
        background-color: #F4F7F6;
        color: #2C3E50;
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Dashboard Header */
    .main-header {
        background: linear-gradient(135deg, #E0F2FE 0%, #F0FDF4 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #D1FAE5;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Expander Card Styling with Pale Orange Accent */
    [data-testid="stExpander"] {
        background-color: #FFF7ED !important; /* Soft pale orange background */
        border-radius: 10px;
        border: 1px solid #FFEDD5 !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }

    /* Expander Header (Solid Teal with White Text) */
    [data-testid="stExpander"] summary {
        background-color: #0D9488 !important;
        border-radius: 8px !important;
        padding: 0.6rem 1rem !important;
    }
    
    [data-testid="stExpander"] summary * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* Fix Input Field Labels inside Expanders to be dark and clearly visible */
    [data-testid="stExpander"] label p {
        color: #1F2937 !important; /* Dark charcoal color for perfect readability */
        font-weight: 600 !important;
    }

    /* High Contrast Custom Styling for the Predict Churn Button */
    div.stButton > button {
        background-color: #0D9488 !important; /* Solid Teal Background */
        color: #FFFFFF !important;            /* Crisp White Text */
        font-size: 18px !important;
        font-weight: 700 !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }

    div.stButton > button:hover {
        background-color: #0F766E !important; /* Darker Teal on hover */
        color: #FFFFFF !important;
        box-shadow: 0 6px 8px -1px rgba(0,0,0,0.15) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load Saved Artifacts from 'models/' folder
try:
  model = joblib.load('models/best_churn_model.pkl')
  scaler = joblib.load('models/scaler.pkl')
  feature_columns = joblib.load('models/feature_columns.pkl')
except Exception as e:
  st.error(
      'Model files not found! Please run `churn_project.py` in your terminal'
      ' first.'
  )
  st.stop()

# Header Section
st.markdown(
    """
    <div class="main-header">
        <h1 style="color: #065F46; margin-bottom: 5px;">🛡️ Customer Churn Prediction Dashboard</h1>
        <p style="color: #4B5563; font-size: 16px;">Click on each module below to expand, fill in the customer data, and predict churn risk.</p>
    </div>
""",
    unsafe_allow_html=True,
)

user_inputs = {}

# --- MODULE A: Customer Profile (Expander Card) ---
with st.expander(
    '👤 Module A: Customer Profile (Click to expand/collapse)', expanded=True
):
  st.markdown(
      '<p style="color: #C2410C; font-weight: 600;">Enter the personal and'
      ' demographic details of the customer:</p>',
      unsafe_allow_html=True,
  )
  col1, col2 = st.columns(2)
  with col1:
    user_inputs['CreditScore'] = st.number_input(
        'Credit Score',
        300,
        850,
        650,
        help="Numerical score reflecting the customer's creditworthiness.",
    )
    user_inputs['Age'] = st.number_input(
        'Age', 18, 100, 35, help='The age of the customer in years.'
    )
    user_inputs['Tenure'] = st.number_input(
        'Tenure (Years)',
        0,
        10,
        3,
        help='Number of years the customer has been with the institution.',
    )
  with col2:
    user_inputs['Geography'] = st.selectbox(
        'Geography',
        [0, 1, 2],
        format_func=lambda x: ['France', 'Germany', 'Spain'][x],
        help='Country or region of residence.',
    )
    user_inputs['Gender'] = st.selectbox(
        'Gender',
        [0, 1],
        format_func=lambda x: 'Female' if x == 0 else 'Male',
        help='Gender of the customer.',
    )

# --- MODULE B: Behavior & Finances (Expander Card) ---
with st.expander(
    '⚙️ Module B: Behavior & Finances (Click to expand/collapse)', expanded=True
):
  st.markdown(
      '<p style="color: #C2410C; font-weight: 600;">Enter account balances,'
      ' product subscriptions, and financial activity:</p>',
      unsafe_allow_html=True,
  )
  col3, col4 = st.columns(2)
  with col3:
    user_inputs['Balance'] = st.number_input(
        'Account Balance',
        0.0,
        250000.0,
        50000.0,
        help='Amount of money deposited in primary account.',
    )
    user_inputs['NumOfProducts'] = st.number_input(
        'Number of Products',
        1,
        4,
        2,
        help='Total banking products subscribed to.',
    )
    user_inputs['HasCrCard'] = st.selectbox(
        'Has Credit Card?',
        [0, 1],
        format_func=lambda x: 'Yes' if x == 1 else 'No',
        help='Holds a credit card with the company.',
    )
  with col4:
    user_inputs['IsActiveMember'] = st.selectbox(
        'Is Active Member?',
        [0, 1],
        format_func=lambda x: 'Yes' if x == 1 else 'No',
        help='Regular activity and engagement level.',
    )
    user_inputs['EstimatedSalary'] = st.number_input(
        'Estimated Salary',
        0.0,
        200000.0,
        60000.0,
        help='Estimated annual salary.',
    )

# --- MODULE C: Satisfaction & Feedback (Expander Card) ---
with st.expander(
    '📊 Module C: Satisfaction & Feedback (Click to expand/collapse)',
    expanded=True,
):
  st.markdown(
      '<p style="color: #C2410C; font-weight: 600;">Enter satisfaction scores,'
      ' complaints, and loyalty card metrics:</p>',
      unsafe_allow_html=True,
  )
  col5, col6 = st.columns(2)
  with col5:
    user_inputs['Complain'] = st.selectbox(
        'Recent Complaints',
        [0, 1],
        format_func=lambda x: 'No' if x == 0 else 'Yes',
        help='Logged formal complaints recently.',
    )
    user_inputs['Satisfaction Score'] = st.slider(
        'Satisfaction Score', 1, 5, 3, help='Customer satisfaction rating (1-5).'
    )
  with col6:
    user_inputs['Card Type'] = st.selectbox(
        'Card Type',
        [0, 1, 2, 3],
        format_func=lambda x: ['DIAMOND', 'GOLD', 'PLATINUM', 'SILVER'][x],
        help='Tier of loyalty or credit card.',
    )
    user_inputs['Point Earned'] = st.number_input(
        'Points Earned', 0, 1000, 500, help='Accumulated loyalty reward points.'
    )

# Fill any remaining schema feature columns if missing from UI layout
for col in feature_columns:
  if col not in user_inputs:
    user_inputs[col] = 0

st.divider()

# Center Action Button & Output Section
col_space1, col_btn, col_space2 = st.columns([1, 2, 1])

with col_btn:
  predict_clicked = st.button('[ PREDICT CHURN ]', use_container_width=True)

if predict_clicked:
  input_df = pd.DataFrame([user_inputs])
  input_df = input_df[feature_columns]  # Reorder columns to match training schema

  # Predict
  prob = model.predict_proba(input_df)[0][1]
  prediction = 1 if prob >= 0.5 else 0

  # Risk Assessment
  if prob >= 0.7:
    risk_level, risk_color = 'HIGH', 'red'
  elif prob >= 0.4:
    risk_level, risk_color = 'MEDIUM', 'orange'
  else:
    risk_level, risk_color = 'LOW', 'green'

  # UI Output Container
  st.markdown(
      """
        <div style="background-color: #FFFFFF; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-top: 20px;">
            <h2 style="color: #1F2937; text-align: center; margin-top:0;">🎯 Live Prediction Results</h2>
        </div>
        """,
      unsafe_allow_html=True,
  )

  out_col1, out_col2, out_col3 = st.columns(3)

  with out_col1:
    if prediction == 1:
      st.error('**Prediction:** Likely to Churn')
    else:
      st.success('**Prediction:** Likely to Stay')

  with out_col2:
    st.metric(label='Churn Probability', value=f'{prob * 100:.2f}%')

  with out_col3:
    st.markdown(f'**Risk Level:** :{risk_color}[{risk_level}]')
    st.progress(prob)

  # Actionable Recommendations
  st.markdown('### 💡 Proactive Retention Strategy')
  if prediction == 1:
    st.warning(
        '⚠️ **Action Required:** Offer personalized retention incentives, fee'
        ' waivers, or a dedicated support call immediately.'
    )
  else:
    st.info(
        '✅ **Status Stable:** Customer is at low risk. Maintain standard'
        ' engagement cadence.'
    )