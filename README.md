# Heart Disease Prediction Project

This project is an end-to-end machine learning application that predicts the likelihood of heart disease using the UCI Heart Disease dataset. The entire pipeline, from data cleaning and model tuning to web deployment, is included.

## Features
- **Data Preprocessing:** Cleans and prepares the raw, multi-source UCI dataset.
- **Modeling:** Trains and evaluates four classification models (Logistic Regression, Decision Tree, Random Forest, SVM).
- **Hyperparameter Tuning:** Optimizes the best-performing model (Random Forest) using GridSearchCV.
- **Interactive UI:** A user-friendly web application built with Streamlit for real-time predictions.

## Live Demo
**You can access the live application here:** [https://700fe89dd9ec.ngrok-free.app/]

## How to Run Locally
1. Clone the repository: `git clone <https://github.com/omar-elsaghir/Heart_Disease/edit/main/README.md>`
2. Navigate into the project directory.
3. Install the required libraries: `pip install -r requirements.txt`
4. Run the Streamlit app: `streamlit run ui/app.py`
