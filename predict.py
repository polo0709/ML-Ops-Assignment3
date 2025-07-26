import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load data
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Load model
model = joblib.load("model.joblib")

# Make a prediction
sample_pred = model.predict(X_test[:1])
print("Sample Prediction:", sample_pred)
