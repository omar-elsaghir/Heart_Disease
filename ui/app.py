import streamlit as st
import pandas as pd
import joblib

# --- Load the Model and Scaler ---
try:
    model = joblib.load('models/final_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
except FileNotFoundError:
    st.error("Error: Model or scaler files not found. Make sure you are running the app from the main project folder.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# --- Main Title ---
st.title('Heart Disease Prediction App ❤️')
st.write(
    "This app uses a Random Forest model to predict the likelihood of heart disease based on patient data. Please enter the patient's details in the sidebar.")

# --- Sidebar for User Input ---
st.sidebar.header('Enter Patient Data')

def get_user_input():
    """
    Creates sidebar widgets to collect user input and returns a dictionary.
    """
    # Using number_input for numerical features to allow typing
    age = st.sidebar.number_input('Age', min_value=29, max_value=77, value=54)
    trestbps = st.sidebar.number_input('Resting Blood Pressure (trestbps)', min_value=94, max_value=200, value=132)
    chol = st.sidebar.number_input('Serum Cholestoral (chol)', min_value=126, max_value=564, value=246)
    thalch = st.sidebar.number_input('Maximum Heart Rate Achieved (thalch)', min_value=71, max_value=202, value=150)
    oldpeak = st.sidebar.number_input('ST Depression (oldpeak)', min_value=0.0, max_value=6.2, value=1.0, step=0.1)

    # Using selectbox for categorical features
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    cp = st.sidebar.selectbox('Chest Pain Type (cp)',
                              ('Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'))
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', ('True', 'False'))
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results (restecg)',
                                   ('Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'))
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', ('Yes', 'No'))
    slope = st.sidebar.selectbox('Slope of the peak exercise ST segment', ('Upsloping', 'Flat', 'Downsloping'))
    thal = st.sidebar.selectbox('Thalassemia (thal)', ('Normal', 'Fixed defect', 'Reversible defect'))
    ca = st.sidebar.number_input('Number of Major Vessels (ca)', 0, 4, 0)
    dataset = 'Cleveland'

    user_data = {
        'age': age, 'sex': sex, 'dataset': dataset, 'cp': cp,
        'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 'restecg': restecg,
        'thalch': thalch, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
        'ca': ca, 'thal': thal
    }
    return user_data

# --- Preprocessing the User Input ---
# This list must match the columns the model was trained on, in the same order.
expected_columns = [
    'age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'sex_Male',
    'dataset_Hungary', 'dataset_Switzerland', 'dataset_VA Long Beach',
    'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
    'fbs_True', 'restecg_normal', 'restecg_st-t abnormality',
    'exang_True', 'slope_flat', 'slope_upsloping', 'thal_normal',
    'thal_reversable defect'  # <-- TYPO FIXED HERE
]

def preprocess_input(user_data):
    data_for_df = {
        'age': user_data['age'], 'trestbps': user_data['trestbps'],
        'chol': user_data['chol'], 'thalch': user_data['thalch'],
        'oldpeak': user_data['oldpeak'], 'ca': user_data['ca'],
        'sex_Male': 1 if user_data['sex'] == 'Male' else 0,
        'dataset_Hungary': 1 if user_data['dataset'] == 'Hungary' else 0,
        'dataset_Switzerland': 1 if user_data['dataset'] == 'Switzerland' else 0,
        'dataset_VA Long Beach': 1 if user_data['dataset'] == 'VA Long Beach' else 0,
        'cp_atypical angina': 1 if user_data['cp'] == 'Atypical Angina' else 0,
        'cp_non-anginal': 1 if user_data['cp'] == 'Non-anginal Pain' else 0,
        'cp_typical angina': 1 if user_data['cp'] == 'Typical Angina' else 0,
        'fbs_True': 1 if user_data['fbs'] == 'True' else 0,
        'restecg_normal': 1 if user_data['restecg'] == 'Normal' else 0,
        'restecg_st-t abnormality': 1 if user_data['restecg'] == 'ST-T wave abnormality' else 0,
        'exang_True': 1 if user_data['exang'] == 'Yes' else 0,
        'slope_flat': 1 if user_data['slope'] == 'Flat' else 0,
        'slope_upsloping': 1 if user_data['slope'] == 'Upsloping' else 0,
        'thal_normal': 1 if user_data['thal'] == 'Normal' else 0,
        'thal_reversable defect': 1 if user_data['thal'] == 'Reversible defect' else 0 # <-- TYPO FIXED HERE
    }
    input_df = pd.DataFrame([data_for_df])
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)
    cols_to_scale = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
    return input_df

# Get the user's input from the sidebar.
user_input_raw = get_user_input()

# Create the button. When it's clicked, use the input to predict.
if st.sidebar.button('Predict'):
    processed_input = preprocess_input(user_input_raw)
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)

    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.warning(f"The model predicts a **HIGH** risk of heart disease.")
    else:
        st.success(f"The model predicts a **LOW** risk of heart disease.")

    st.subheader('Prediction Probability')
    st.write(f"**Probability of having Heart Disease:** {prediction_proba[0][1] * 100:.2f}%")
    st.write(f"**Probability of NOT having Heart Disease:** {prediction_proba[0][0] * 100:.2f}%")