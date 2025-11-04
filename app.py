# ------------------------------------------------------------
# 🏠 Boston Housing Price Prediction (Advanced Streamlit App)
# ------------------------------------------------------------
# Features:
#   ✅ Compare Linear Regression, Random Forest, and Gradient Boosting models
#   ✅ Predict house prices instantly
#   ✅ View model R² comparison chart
#   ✅ Developer details in sidebar
#   ✅ Smooth animations & custom UI styling
# ------------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from PIL import Image

# ------------------------------------------------------------
# 🎨 Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Boston Housing Predictor",
    page_icon="🏠",
    layout="wide"
)

# ------------------------------------------------------------
# 💫 Custom CSS for Styling
# ------------------------------------------------------------
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to right, #f0f4f8, #e3f2fd);
    }
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #004e92, #000428);
        color: white;
    }
    .sidebar-title {
        color: #ffffff;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 15px;
    }
    .sidebar-sub {
        color: #ddd;
        font-size: 14px;
    }
    .stButton button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        padding: 0.6em 1.2em;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #0072ff, #00c6ff);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 🎬 Animated Title Section
# ------------------------------------------------------------
st.markdown("<h1 style='text-align: center; color:#004e92;'>🏠 Boston Housing Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color:#333;'>Compare ML Models & Predict Median Home Prices Instantly</h4>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------
# 📦 Load Trained Models
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    try:
        with open("Linear_Regression_BostonHousing.pkl", "rb") as f:
            models["Linear Regression"] = pickle.load(f)
        with open("Random_Forest_BostonHousing.pkl", "rb") as f:
            models["Random Forest"] = pickle.load(f)
        with open("Gradient_Boosting_BostonHousing.pkl", "rb") as f:
            models["Gradient Boosting"] = pickle.load(f)
    except Exception as e:
        st.error("❌ Error loading models: " + str(e))
    return models

models = load_models()

# ------------------------------------------------------------
# 🧠 Sidebar - Developer Details
# ------------------------------------------------------------
st.sidebar.markdown("<div class='sidebar-title'>👨‍💻 Developer Info</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-sub'>Name: <b>Abhishek Shelke</b></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-sub'>Master’s in Computer Science</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-sub'>Interest: AI & Machine Learning</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-sub'>GitHub: <a style='color:#80dfff;' href='https://github.com/abhishek-shelke' target='_blank'>@abhishek-shelke</a></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-sub'>LinkedIn: <a style='color:#80dfff;' href='https://linkedin.com/in/abhishek-shelke' target='_blank'>@abhishek-shelke</a></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# ------------------------------------------------------------
# ⚙️ User Input Section
# ------------------------------------------------------------
st.subheader("🏡 Enter Housing Details Below")

col1, col2, col3 = st.columns(3)
with col1:
    CRIM = st.number_input("Per capita crime rate (CRIM)", 0.0, 100.0, 0.1)
    ZN = st.number_input("Residential land zoned (ZN)", 0.0, 100.0, 0.0)
    INDUS = st.number_input("Non-retail business acres (INDUS)", 0.0, 30.0, 5.0)
with col2:
    RM = st.number_input("Average number of rooms (RM)", 1.0, 10.0, 6.0)
    AGE = st.number_input("Older houses (AGE)", 0.0, 100.0, 50.0)
    DIS = st.number_input("Distance to employment centers (DIS)", 0.0, 15.0, 4.0)
with col3:
    TAX = st.number_input("Property tax rate (TAX)", 100.0, 800.0, 300.0)
    PTRATIO = st.number_input("Pupil-teacher ratio (PTRATIO)", 10.0, 30.0, 18.0)
    LSTAT = st.number_input("Lower status population (LSTAT)", 0.0, 40.0, 12.0)

input_data = np.array([[CRIM, ZN, INDUS, RM, AGE, DIS, TAX, PTRATIO, LSTAT]])

# ------------------------------------------------------------
# 🧩 Model Selection
# ------------------------------------------------------------
st.markdown("### ⚙️ Choose a Model")
model_choice = st.selectbox("Select Model", list(models.keys()))

# ------------------------------------------------------------
# 🔮 Prediction
# ------------------------------------------------------------
if st.button("🔮 Predict House Price"):
    with st.spinner('Calculating prediction...'):
        time.sleep(1.5)
        model = models[model_choice]
        prediction = model.predict(input_data)[0]
        st.success(f"🏡 Predicted Median House Price: **${prediction * 1000:,.2f}**")
        st.balloons()
        st.info(f"Model Used: {model_choice}")

# ------------------------------------------------------------
# 📊 Model Performance Comparison Section
# ------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Model Performance Comparison")

# Dummy evaluation (replace with your actual evaluation_df values if available)
evaluation_data = {
    "Model": ["Linear Regression", "Random Forest", "Gradient Boosting"],
    "R² Score": [0.68, 0.86, 0.88]
}
df_eval = pd.DataFrame(evaluation_data)

col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(df_eval.set_index("Model"), use_container_width=True)
with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df_eval["Model"], df_eval["R² Score"], color=["#0072ff", "#00c6ff", "#0059b3"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("R² Score")
    ax.set_title("Model Comparison (Test R² Scores)")
    st.pyplot(fig)

# ------------------------------------------------------------
# 📈 Footer
# ------------------------------------------------------------
st.markdown("---")
st.markdown("<h5 style='text-align:center; color:#333;'>Developed with ❤️ by <b>Abhishek Shelke</b></h5>", unsafe_allow_html=True)
