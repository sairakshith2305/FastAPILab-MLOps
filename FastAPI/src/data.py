import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    """
    Load the Breast Cancer dataset and return the features and target values.
    """
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])
    return X, y

def split_data(X, y):
    """
    Split the data into training and testing sets and scale the features.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler