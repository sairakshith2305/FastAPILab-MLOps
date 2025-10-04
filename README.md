# Breast Cancer Prediction API

A FastAPI-based machine learning application that predicts whether a breast tumor is **malignant** or **benign** using Logistic Regression.

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

