import joblib
import pandas as pd

# Load the saved model
model = joblib.load("house_price_model.pkl")

# Example input (ensure this matches the feature order and preprocessing)
example = pd.DataFrame({
    'LotArea': [9500],
    'OverallQual': [7],
    'OverallCond': [5],
    'YearBuilt': [2005],
    'GrLivArea': [2000],
    'FullBath': [2],
    'GarageCars': [2]
})

# Predict
prediction = model.predict(example)
print(f"Predicted house price: ${prediction[0]:,.0f}")
