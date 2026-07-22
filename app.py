import streamlit as st
import pickle
import numpy as np

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://benchmarkwines.com.sg/cdn/shop/articles/12-different-types-of-red-wine-2945079.jpg?v=1765390078&width=480");
        background-size: cover;
        background-position: center;     
        background-repeat: no-repeat;    
        background-attachment: fixed;    
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Load the model
with open("model_RF.pkl", "rb") as f:
    model = pickle.load(f)


# Load scaler if used
try:
    with open("scalar.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None
    
st.markdown(
    """
    <div style="
        display:flex;
        justify-content:center;
        align-items:center;
        height:30vh;
    ">
        <div style="
            background: rgba(255, 255, 255, 0.6);
            padding: 30px 60px;
            border-radius: 20px;
            text-align: center;
            color: black;
            font-size: 36px;
            font-weight: 700;
            color: #5A2D0C;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    ">
        <span style="
                font-size: 44px;
                font-weight: 800;
                background: linear-gradient(90deg, #8B0000, #C0392B, #E74C3C);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
        ">
            Wine Quality Prediction🍷
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* Predict button container */
    div.stButton > button {
        background: rgba(255, 255, 255, 0.6);
        color: #8B0000;
        font-size: 20px;
        font-weight: 700;
        padding: 12px 40px;
        border-radius: 30px;
        border: none;
        box-shadow: 0 8px 25px rgba(0,0,0,0.25);
        transition: all 0.3s ease-in-out;
    }

    /* Hover effect */
    div.stButton > button:hover {
        background: linear-gradient(90deg, #8B0000, #C0392B, #E74C3C);
        color: white;
        transform: scale(1.05);
        box-shadow: 0 12px 30px rgba(0,0,0,0.35);
    }

    /* Click effect */
    div.stButton > button:active {
        transform: scale(0.97);
    }
    </style>
    """,
    unsafe_allow_html=True
)
import streamlit as st

st.markdown(
    """
    <style>
    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .result-card {
        background: rgba(255, 255, 255, 0.6);
        padding: 25px 50px;
        border-radius: 25px;
        text-align: center;
        font-size: 26px;
        font-weight: 800;
        color: #8B0000;
        box-shadow: 0 12px 35px rgba(0,0,0,0.3);
        animation: fadeUp 0.8s ease-in-out;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("img.jpg",  use_container_width=True)

st.markdown(
    "<h4 style='text-align:color: #8B0000;' center;'>Enter the details of the wine to predict its quality.</h4>",
    unsafe_allow_html=True)


#user inputs
age = st.number_input("Fixed Acidity")
sex = st.number_input("Volatile Acidity")
cp = st.number_input("Citric Acid")
trestbps = st.number_input("Residual Sugar")
chol = st.number_input("Chlorides")
fbs = st.number_input("Free Sulfur Dioxide")
restecg = st.number_input("Total Sulfur Dioxide")
thalach = st.number_input("Density")
oldpeak = st.number_input("pH")
slope = st.number_input("Sulphates")    
ca = st.number_input("Alcohol")


import streamlit as st

st.markdown(
    """
    <style>
    /* LABEL text (above input) */
    label,
    div[data-testid="stWidgetLabel"] {
        color: white !important;
        font-weight: 600;
        font-size: 16px;
    }

    /* FULL input box background */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-radius: 12px !important;
    }

    /* Text inside input */
    div[data-baseweb="input"] input {
        color: black !important;
        font-weight: 500;
        font-size: 15px;
    }

    /* Placeholder text */
    div[data-baseweb="input"] input::placeholder {
        color: #ffffff !important;
    }
    </style> 
    """,
    unsafe_allow_html=True
)


# Make predictions
if st.button("Predict Quality"):
    if scaler:
        inputs = scaler.transform(np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, oldpeak, slope, ca, thal]]))
    else:
        inputs = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, oldpeak, slope, ca]])
    prediction = model.predict(inputs)
    st.write(f"The predicted quality of the wine is📈: {prediction[0]}")
    