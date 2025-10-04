import joblib

def predict_data(X):
    """
    Predict the class labels for the input data.
    """
    model = joblib.load("../model/breast_cancer_model.pkl")
    scaler = joblib.load("../model/scaler.pkl")
    
    X_scaled = scaler.transform(X)
    
    y_pred = model.predict(X_scaled)
    return y_pred

def predict_proba(X):
    """
    Predict the probability of each class for the input data.
    """
    model = joblib.load("../model/breast_cancer_model.pkl")
    scaler = joblib.load("../model/scaler.pkl")
    
    X_scaled = scaler.transform(X)
    y_proba = model.predict_proba(X_scaled)
    return y_proba