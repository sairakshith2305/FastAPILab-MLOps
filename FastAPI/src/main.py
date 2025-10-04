from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data, predict_proba


app = FastAPI()

class BreastCancerData(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

class PredictionResponse(BaseModel):
    prediction: int
    diagnosis: str

class ProbabilityResponse(BaseModel):
    prediction: int
    diagnosis: str
    malignant_probability: float
    benign_probability: float

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_breast_cancer(features: BreastCancerData):
    try:
        feature_list = [[
            features.mean_radius, features.mean_texture, features.mean_perimeter,
            features.mean_area, features.mean_smoothness, features.mean_compactness,
            features.mean_concavity, features.mean_concave_points, features.mean_symmetry,
            features.mean_fractal_dimension, features.radius_error, features.texture_error,
            features.perimeter_error, features.area_error, features.smoothness_error,
            features.compactness_error, features.concavity_error, features.concave_points_error,
            features.symmetry_error, features.fractal_dimension_error, features.worst_radius,
            features.worst_texture, features.worst_perimeter, features.worst_area,
            features.worst_smoothness, features.worst_compactness, features.worst_concavity,
            features.worst_concave_points, features.worst_symmetry, features.worst_fractal_dimension
        ]]

        prediction = predict_data(feature_list)
        diagnosis = "Benign" if prediction[0] == 1 else "Malignant"
        
        return PredictionResponse(
            prediction=int(prediction[0]),
            diagnosis=diagnosis
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-proba", response_model=ProbabilityResponse)
async def predict_with_probability(features: BreastCancerData):
    try:
        feature_list = [[
            features.mean_radius, features.mean_texture, features.mean_perimeter,
            features.mean_area, features.mean_smoothness, features.mean_compactness,
            features.mean_concavity, features.mean_concave_points, features.mean_symmetry,
            features.mean_fractal_dimension, features.radius_error, features.texture_error,
            features.perimeter_error, features.area_error, features.smoothness_error,
            features.compactness_error, features.concavity_error, features.concave_points_error,
            features.symmetry_error, features.fractal_dimension_error, features.worst_radius,
            features.worst_texture, features.worst_perimeter, features.worst_area,
            features.worst_smoothness, features.worst_compactness, features.worst_concavity,
            features.worst_concave_points, features.worst_symmetry, features.worst_fractal_dimension
        ]]

        prediction = predict_data(feature_list)
        probabilities = predict_proba(feature_list)
        diagnosis = "Benign" if prediction[0] == 1 else "Malignant"
        
        return ProbabilityResponse(
            prediction=int(prediction[0]),
            diagnosis=diagnosis,
            malignant_probability=float(probabilities[0][0]),
            benign_probability=float(probabilities[0][1])
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))