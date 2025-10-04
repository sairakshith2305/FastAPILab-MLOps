import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    """
    Load the Breast Cancer dataset and return the features and target values.
    Returns:
        X (numpy.ndarray): The features of the Breast Cancer dataset.
        y (numpy.ndarray): The target values (0=malignant, 1=benign).
    """
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
    return X, y

def split_data(X, y):
    """
    Split the data into training and testing sets and scale the features.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test, scaler (tuple): The split dataset and fitted scaler.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler