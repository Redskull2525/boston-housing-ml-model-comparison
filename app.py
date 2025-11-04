# ------------------------------------------------------------
# 🏠 Boston Housing Price Prediction (Advanced Streamlit App)
# ------------------------------------------------------------
# This Streamlit app allows users to:
#   ✅ Choose between Linear Regression, Random Forest, and Gradient Boosting models
#   ✅ Input housing features
#   ✅ Get predicted house prices instantly
#   ✅ View model performance scores
# ------------------------------------------------------------

import streamlit as st
import numpy as np
import pickle
import time
from PIL import Image
from sklearn.metrics import r2_score

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
    /* Gradient background */
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
st.markdown("<h4 style='text-align: center; color:#333;'>Compare multiple ML models and predict median home values instantly</h4>", unsafe_allow_html=True)
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
# 📈 Footer
# ------------------------------------------------------------
st.markdown("---")
st.markdown("<h5 style='text-align:center; color:#333;'>Developed with ❤️ by <b>Abhishek Shelke</b></h5>", unsafe_allow_html=True)
