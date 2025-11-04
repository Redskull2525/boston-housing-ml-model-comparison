# ------------------------------------------------------------
# 🏠 Boston Housing Price Prediction (Dark Themed Streamlit App)
# ------------------------------------------------------------
# Features:
#   ✅ Compare Linear Regression, Random Forest, and Gradient Boosting models
#   ✅ Predict house prices interactively
#   ✅ Model R² comparison chart
#   ✅ Beautiful permanent sidebar with developer info
#   ✅ Sleek dark mode design with neon blue highlights
# ------------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ------------------------------------------------------------
# 🎨 Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Boston Housing Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"   # Sidebar always open
)

# ------------------------------------------------------------
# 🌙 Custom Dark Theme CSS
# ------------------------------------------------------------
st.markdown("""
    <style>
    /* --- MAIN APP BACKGROUND --- */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #0d0d0d, #1a1a1a);
        color: #FAFAFA;
    }

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #000428, #004e92);
        padding: 20px 15px;
        box-shadow: 4px 0px 15px rgba(0, 0, 0, 0.5);
    }

    .sidebar-title {
        color: #00c6ff;
        font-size: 24px;
        font-weight: 800;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 20px;
    }

    .sidebar-sub {
        color: #cce7ff;
        font-size: 15px;
        margin: 8px 0;
    }

    .sidebar-link a {
        color: #66e0ff;
        text-decoration: none;
        font-weight: bold;
    }

    .sidebar-link a:hover {
        color: #00ffff;
        text-decoration: underline;
    }

    /* --- BUTTONS --- */
    .stButton button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        padding: 0.7em 1.3em;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
        box-shadow: 0 0 12px rgba(0, 198, 255, 0.4);
    }

    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(0, 198, 255, 0.8);
    }

    /* --- TITLES & HEADERS --- */
    h1, h2, h3, h4 {
        color: #00c6ff;
        text-shadow: 0px 0px 8px #00c6ff;
    }

    /* --- FOOTER --- */
    .footer {
        text-align: center;
        color: #999;
        margin-top: 30px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 🎬 App Header
# ------------------------------------------------------------
st.markdown("<h1 style='text-align: center;'>🏠 Boston Housing Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #cccccc;'>Compare ML Models & Predict Median Home Prices Instantly</h4>", unsafe_allow_html=True)
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
# 🧠 Sidebar - Developer Info (Permanent)
# ------------------------------------------------------------
st.sidebar.markdown("<div class='sidebar-title'>👨‍💻 Developer Info</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-sub'><b>Name:</b> Abhishek Shelke</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-sub'><b>Degree:</b> Master’s in Computer Science</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-sub'><b>Interest:</b> Artificial Intelligence & Machine Learning</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-sub sidebar-link'><b>GitHub:</b> <a href='https://github.com/Redskull2525' target='_blank'>@abhishek-shelke</a></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-sub sidebar-link'><b>LinkedIn:</b> <a href='https://linkedin.com/in/abhishek-s-b98895249' target='_blank'>@abhishek-shelke</a></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("<div style='font-size:13px; color:#ccc;'>📍 Pune, India</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# ⚙️ User Input Section
# ------------------------------------------------------------
st.subheader("🏡 Enter Housing Details")

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
# 📊 Model Comparison Chart
# ------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Model Performance Comparison")

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
    ax.bar(df_eval["Model"], df_eval["R² Score"], color=["#00c6ff", "#0072ff", "#33ccff"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("R² Score", color="white")
    ax.set_title("Model Comparison (Test R² Scores)", color="#00c6ff")
    ax.tick_params(colors="white")
    st.pyplot(fig)

# ------------------------------------------------------------
# 📈 Footer
# ------------------------------------------------------------
st.markdown("<div class='footer'>Developed by <b>Abhishek Shelke</b></div>", unsafe_allow_html=True)
