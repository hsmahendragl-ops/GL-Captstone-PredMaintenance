import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="HSMahendraGL/predictive_maintenance", filename="best_maintainance_predictor_model_v1.joblib")
model = joblib.load(model_path)

import streamlit as st
import pandas as pd

# Streamlit UI for Machine Failure Prediction
st.title("Prediction Maintenance App")
st.write("""
This application predicts the condition of the engine and likelihood of Engine needing repair.
Prediction is based on Engine and Oil parameters.
Please enter the profile data below to get a prediction.
""")

# User input using sliders and floating point values
EngineRPM = st.slider("Engine rpm", min_value=50, max_value=2500, value=1000, step=1)

# Sliders configured directly with float values (divided by 1000 beforehand)
Luboilpressure = st.slider("Lub oil pressure", min_value=0.001, max_value=8.0, value=1.5, step=0.001, format="%.3f")
Fuelpressure = st.slider("Fuel pressure", min_value=0.001, max_value=20.0, value=1.5, step=0.001, format="%.3f")
Coolantpressure = st.slider("Coolant pressure", min_value=0.001, max_value=8.0, value=1.5, step=0.001, format="%.3f")

luboiltemp = st.slider("lub oil temp", min_value=65.0, max_value=95.0, value=70.0, step=0.1, format="%.1f")
Coolanttemp = st.slider("Coolant temp", min_value=50.0, max_value=99.0, value=65.0, step=0.1, format="%.1f")

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Engine rpm': EngineRPM,
    'Lub oil pressure': Luboilpressure,
    'Fuel pressure': Fuelpressure,
    'Coolant pressure': Coolantpressure,
    'lub oil temp': luboiltemp,
    'Coolant temp': Coolanttemp,
}])

if st.button("Predict Engine Condition"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")

    if prediction == 0:
        result = "Engine is running OK"
        st.success(f"The model predicts: **{result}**")
    else:
        result = "Engine may need Service shortly"
        st.error(f"The model predicts: **{result}**")
