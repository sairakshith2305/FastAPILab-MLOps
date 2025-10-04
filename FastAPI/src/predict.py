import joblib

def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (list or numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels (0=malignant, 1=benign).
    """
    # Load both model and scaler
    model = joblib.load("../model/breast_cancer_model.pkl")
    scaler = joblib.load("../model/scaler.pkl")
    
    # Scale the input features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    y_pred = model.predict(X_scaled)
    return y_pred

def predict_proba(X):
    """
    Predict the probability of each class for the input data.
    Args:
        X (list or numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_proba (numpy.ndarray): Predicted probabilities for each class.
    """
    model = joblib.load("../model/breast_cancer_model.pkl")
    scaler = joblib.load("../model/scaler.pkl")
    
    X_scaled = scaler.transform(X)
    y_proba = model.predict_proba(X_scaled)
    return y_proba