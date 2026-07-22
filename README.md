# wine_quality

🍷 Wine Quality Prediction using Machine Learning

📌 Project Overview

The Wine Quality Prediction project is a Streamlit-based Machine Learning web application that predicts the quality of red wine based on its physicochemical properties. The application uses a trained Random Forest model to classify wine quality from user-provided input features. It features an attractive, interactive interface with a custom background, styled input fields, and real-time prediction capabilities.

🚀 Features
Predicts wine quality using a trained Random Forest model
Interactive and user-friendly Streamlit interface
Real-time quality prediction
Accepts 11 wine characteristics as input
Supports feature scaling (if scaler is available)
Modern UI with custom CSS styling and background image
🛠️ Technologies Used
Python
Streamlit
Scikit-learn
NumPy
Pickle
HTML/CSS (Streamlit Styling)
📂 Project Structure
Wine-Quality-Prediction/
│
├── app.py
├── model_RF.pkl
├── scalar.pkl
├── img.jpg
├── requirements.txt
└── README.md
📊 Dataset

This project uses the Red Wine Quality Dataset, which contains physicochemical properties of wine samples.

Input Features
Fixed Acidity
Volatile Acidity
Citric Acid
Residual Sugar
Chlorides
Free Sulfur Dioxide
Total Sulfur Dioxide
Density
pH
Sulphates
Alcohol

The model predicts the quality score of the wine based on these features.

🤖 Machine Learning Workflow
Load the trained Random Forest model.
Load the scaler (if available).
Accept user inputs through the Streamlit interface.
Scale the input features.
Predict the wine quality.
Display the predicted quality instantly.
▶️ Installation
Clone the repository
git clone https://github.com/yourusername/Wine-Quality-Prediction.git
Navigate to the project folder
cd Wine-Quality-Prediction
Install dependencies
pip install -r requirements.txt
▶️ Run the Application
streamlit run app.py

The application will open automatically in your default web browser.

📦 Requirements
streamlit
numpy
scikit-learn
pickle-mixin

Install manually if needed:

pip install streamlit numpy scikit-learn
🖥️ Application Workflow
User Enters Wine Features
            │
            ▼
   Feature Scaling (Optional)
            │
            ▼
   Random Forest Model
            │
            ▼
   Predict Wine Quality
            │
            ▼
   Display Prediction
🎯 Learning Outcomes
Data preprocessing
Feature scaling
Machine Learning model deployment
Random Forest algorithm
Building interactive web applications with Streamlit
Model serialization using Pickle
Creating responsive user interfaces with custom CSS
📈 Future Enhancements
Display wine quality category (Poor, Average, Good, Excellent)
Add probability/confidence score for predictions
Support multiple machine learning algorithms for comparison
Visualize feature importance
Deploy the application on Streamlit Community Cloud, Render, or Hugging Face Spaces
Store prediction history for users
👨‍💻 Author

M. Navya Sahithi

B.Tech – Artificial Intelligence & Data Science

Aspiring Data Analyst | Machine Learning Enthusiast

📄 License

This project is developed for educational and learning purposes. Feel free to use, modify, and enhance it for academic or personal projects.
