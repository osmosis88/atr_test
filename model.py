import joblib

def load_models():
    """Load trained ML models."""
    models = {
        "DVT_nc": joblib.load("DVT_nc_gbc_model.joblib"),
        "surgical_site_infection": joblib.load("surgical_site_infection_gbc_model.joblib"),
        "syst_major_complications": joblib.load("syst_major_complications_gbc_model.joblib")
    }
    return models

def predict(models, input_data):
    """Make predictions using trained models."""
    probabilities = {target: models[target].predict_proba(input_data)[0, 1] for target in models}
    return probabilities
