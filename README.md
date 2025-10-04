# Breast Cancer Prediction API

A FastAPI-based machine learning application that predicts whether a breast tumor is **malignant(=0)** or **benign(=1)** using Logistic Regression.

## Dataset

**Wisconsin Breast Cancer Dataset** from scikit-learn
- 569 samples
- 30 features
- Binary classification: 0 = Malignant, 1 = Benign


## API Endpoints

### Health Check
```
GET /
```
Returns API health status.

### Predict (Basic)
```
POST /predict
```
Returns prediction and diagnosis.

### Predict with Probability
```
POST /predict-proba
```
Returns prediction, diagnosis, and probability scores of malignant and beingn.


## Model Details

- **Model:** Logistic Regression
- **Preprocessing:** StandardScaler (feature scaling)
- **Train/Test Split:** 80/20
- **Training Time:** < 1 minute on CPU
- **Expected Accuracy:** ~95-97%

## Notes

- Feature scaling is applied using StandardScaler (required for Logistic Regression)
- Both the model and scaler are saved and loaded during predictison
- The model predicts: 0 = Malignant (cancerous), 1 = Benign (non-cancerous)