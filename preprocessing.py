import pandas as pd

def preprocess_input(age, asa_n, anesthesia_nc, smoking, hypertension, dialysis, steroids, diabetes):
    # Mappings for categorical values
    asa_mapping = {"ASA I": 0, "ASA II": 1, "ASA III": 2, "ASA IV": 3}
    anesthesia_mapping = {"General": "General", "Regional": "Regional", "Spinal": "Spinal", "MAC": "MAC"}

    # Create DataFrame with correct column names
    processed_data = pd.DataFrame({
        "Age": [age],
        "ASA_n": [asa_mapping[asa_n]],
        "anesthesia_nc_" + anesthesia_mapping[anesthesia_nc]: [1],  # One-hot encoding
        "Smoke_n_Yes": [1 if smoking == "Yes" else 0],
        "Hypertension_n_Yes": [1 if hypertension == "Yes" else 0],
        "Dialysis_n_Yes": [1 if dialysis == "Yes" else 0],
        "Chornic_Steroid_n_Yes": [1 if steroids == "Yes" else 0],
        "Diabetes_nc_Yes": [1 if diabetes == "Yes" else 0]
    })

    # Ensure all expected columns exist
    expected_columns = [
        "Age", "ASA_n", "anesthesia_nc_General", "anesthesia_nc_Regional",
        "anesthesia_nc_Spinal", "anesthesia_nc_MAC", "Smoke_n_Yes",
        "Hypertension_n_Yes", "Dialysis_n_Yes", "Chornic_Steroid_n_Yes", "Diabetes_nc_Yes"
    ]
    
    for col in expected_columns:
        if col not in processed_data.columns:
            processed_data[col] = 0  # Fill missing columns with 0

    return processed_data[expected_columns]  # Ensure correct column order
