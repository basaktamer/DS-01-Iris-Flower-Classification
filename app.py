import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Iris Classifier (No-Pickle)", page_icon="🌸")

# --- 1. Load Data & Train Model ---
@st.cache_data # This keeps the model in memory so it doesn't retrain on every slider move
def train_model():
    # Load your CSV file
    df = pd.read_csv('Iris.csv')
    
    # Preprocessing (matching what you did in your notebook)
    # Drop Id if it exists
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    # Map species to numbers
    d = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    df['Species'] = df['Species'].map(d)
    
    # Split features and target
    X = df.drop('Species', axis=1)
    y = df['Species']
    
    # Train the model
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    return clf, X.columns

# Initialize the model
model, feature_names = train_model()
species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# --- 2. Streamlit UI ---
st.title("🌸 Iris Species Predictor")
st.info("This app trains a Random Forest model instantly using the Iris.csv file.")

st.sidebar.header("Input Features")

def user_input_features():
    # We use the feature names from the CSV to make sure sliders match exactly
    sepal_l = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.4)
    sepal_w = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.4)
    petal_l = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 1.3)
    petal_w = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 0.2)
    
    data = {
        'SepalLengthCm': sepal_l,
        'SepalWidthCm': sepal_w,
        'PetalLengthCm': petal_l,
        'PetalWidthCm': petal_w
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader('Your Input')
st.write(input_df)

# --- 3. Prediction ---
prediction = model.predict(input_df)
probs = model.predict_proba(input_df)

st.subheader('Result')
st.success(f"Predicted Species: **{species_map[prediction[0]]}**")

# Probability Chart
st.subheader('Certainty')
proba_df = pd.DataFrame(probs, columns=['Setosa', 'Versicolor', 'Virginica'])
st.bar_chart(proba_df.T)