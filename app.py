import streamlit as st
import requests
import json  # Import json module for JSON serialization

st.title("Football Player Clustering")
st.write("ML model to cluster players based on similarity")

def get_prediction_from_api(url, input_data):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=json.dumps(input_data))  # Use json.dumps to serialize input_data
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to get a response from the API"}

# Input fields using sliders
appearance = st.slider('Appearance', 0.0, 100.0, 10.0)
goals = st.slider('Goals', 0.0, 1.0, 0.1)
award = st.slider('Award (0 or 1)', min_value=0, max_value=1, step=1)
height = st.slider('Height (cm)', 150.0, 210.0, 175.0)

# API URL
api_url = "https://deployment-ml-model.onrender.com/predict"

# Predict button
if st.button('Predict Cluster'):
    # Prepare the payload
    payload = {
        "appearance": appearance,
        "goals": goals,
        "award": award,
        "height": height
    }

    # Call the prediction API
    result = get_prediction_from_api(api_url, payload)  # Corrected the variable name from input_data to payload
    
     # Display the result
    if "pred" in result:
        st.success(f"The player belongs to cluster: {result['pred']}")
    else:
        st.error(f"Error: {result.get('error', 'Unknown error')}")



