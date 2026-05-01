import streamlit as st
import pickle
import numpy as np
from streamlit_lottie import st_lottie
import requests

# Page configuration
st.set_page_config(
    page_title="Healthcare Insurance Cost Predictor",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

# Load Lottie animation
@st.cache_data
def load_lottie(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return None

model = load_model()
lottie_health = load_lottie(
    "https://assets2.lottiefiles.com/packages/lf20_5njp3vgg.json"
)

# Title
st.title("Healthcare Insurance Cost Predictor")
st.markdown("Enter your details below to get an estimated insurance cost.")
st.markdown("---")

# Input section
col1, col2, col3 = st.columns(3)

with col1:
    # Age with Lottie animation side by side
    age_col, anim_col = st.columns([2, 1])
    with age_col:
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
    with anim_col:
        if lottie_health:
            st_lottie(lottie_health, height=80, key="age_anim")

    sex = st.selectbox("Sex", ["male", "female"])

with col2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

with col3:
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

st.markdown("---")

# Encode inputs
sex_encoded = 1 if sex == "male" else 0
smoker_encoded = 1 if smoker == "yes" else 0
region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
region_encoded = region_map[region]

# Predict
if st.button("Predict Insurance Cost"):
    input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
    prediction = model.predict(input_data)[0]

    st.markdown("### Predicted Insurance Cost")
    st.success(f"${prediction:,.2f}")

    st.markdown("---")
    st.markdown("**Note:** This is an estimate based on a Linear Regression model trained on historical data.")
