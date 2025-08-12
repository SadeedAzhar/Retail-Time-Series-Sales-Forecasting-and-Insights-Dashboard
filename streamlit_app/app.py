import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load data for dropdown options
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

df = load_data()

# Load saved model and preprocessors 
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("Sales Forecasting App")

# User Input
store_options = sorted(df['store_nbr'].dropna().unique().tolist())
store = st.selectbox("Select Store Number", store_options)


# Get family options 
family_values = df['family'].dropna().unique()  
family_options = [str(f) for f in family_values]  
family = st.selectbox("family", sorted(family_options))
onpromotion = st.number_input("Number of Promotions", min_value=0, step=1)
date_input = st.date_input("Forecast Date", value=datetime.today())

# Feature Engineering 
df_input = pd.DataFrame({
    'store_nbr': [store],
    'family': [family],
    'onpromotion': [onpromotion],
    'Year': [date_input.year],
    'Month': [date_input.month],
    'Day': [date_input.day],
    'Weekday': [date_input.weekday()]
})


# Preprocessing
# Encode family using saved LabelEncoder
try:
    df_input['family'] = label_encoder.transform(df_input['family'])
except ValueError:
    st.error("Selected family was not seen during training. Please select a valid one.")
    st.stop()

try:
    df_processed = preprocessor.transform(df_input)
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()

# Prediction 
if st.button("Forecast Sales"):
    try:
        prediction = model.predict(df_processed)[0]
        prediction = max(0, prediction)  # Ensure no negative sales
        st.success(f"📈 Predicted Sales: {round(prediction, 2)}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
