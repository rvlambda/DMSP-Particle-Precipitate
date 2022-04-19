import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    result = 0.0
    result = model.predict(data)
    if result > 0.5:
        result = 1.0
    
    return result