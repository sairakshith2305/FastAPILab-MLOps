from sklearn.linear_model import LogisticRegression
import joblib
from data import load_data, split_data

def fit_model(X_train, y_train, scaler):
    """
    Train a Logistic Regression Classifier and saving both model and scaler to files.
    """
    lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
    lr_classifier.fit(X_train, y_train)
    
    joblib.dump(lr_classifier, "../model/breast_cancer_model.pkl")
    joblib.dump(scaler, "../model/scaler.pkl")
    
    print("Model and scaler saved successfully!")
    print(f"Training accuracy: {lr_classifier.score(X_train, y_train):.4f}")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = split_data(X, y)
    fit_model(X_train, y_train, scaler)
    model = joblib.load("../model/breast_cancer_model.pkl")
    test_accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")