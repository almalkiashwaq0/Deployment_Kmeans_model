import streamlit as st
import requests
import numpy as np

st.title("Football Player Clustering")
st.write("Use the sliders to input player statistics and predict the player's cluster.")

appearance = st.slider('Appearance', 0.0, 100.0, 10.0)
goals = st.slider('Goals', 0.0, 1.0, 0.1)
award = st.slider('Award (0 or 1)', min_value=0, max_value=1, step=1)
height = st.slider('Height (cm)', 150.0, 210.0, 175.0)

if st.button('Predict Cluster'):
    payload = {
        "appearance": appearance,
        "goals": goals,
        "award": award,
        "height": height
    }
    response = requests.post("https://deployment-ml-model.onrender.com/predict", json=payload)
    if response.status_code == 200:
        result = response.json()
        st.success(f'The player belongs to cluster: {result["cluster"]}')
    else:
        st.error("Error in prediction.")

