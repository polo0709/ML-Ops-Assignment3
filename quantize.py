import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load trained sklearn model
model = joblib.load("model.joblib")
coef = model.coef_
intercept = model.intercept_

# Store unquantized parameters
unquant_params = {
    "coef": coef,
    "intercept": intercept
}
joblib.dump(unquant_params, "unquant_params.joblib")

# Quantization to int8 (supports negative values)
def quantize(arr):
    scale = np.max(np.abs(arr)) / 127.0 if np.max(np.abs(arr)) != 0 else 1.0
    quant = np.round(arr / scale).astype(np.int8)
    return quant, scale

def dequantize(quant, scale):
    return quant.astype(np.float32) * scale

# Quantize model weights and bias
q_coef, scale_coef = quantize(coef)
q_intercept, scale_intercept = quantize(np.array([intercept]))

# Save quantized parameters
quant_params = {
    "coef": q_coef,
    "intercept": q_intercept,
    "scale_coef": scale_coef,
    "scale_intercept": scale_intercept
}
joblib.dump(quant_params, "quant_params.joblib")

# Define equivalent PyTorch Linear model
class QuantizedLinearModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# Load data
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Create model and load dequantized weights
model_torch = QuantizedLinearModel(input_dim=X_train.shape[1])
with torch.no_grad():
    model_torch.linear.weight.copy_(torch.tensor(dequantize(q_coef, scale_coef).reshape(1, -1)))
    model_torch.linear.bias.copy_(torch.tensor(dequantize(q_intercept, scale_intercept)))

# Evaluate model
model_torch.eval()
with torch.no_grad():
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    preds = model_torch(X_tensor).squeeze().numpy()
    print("Sample prediction:", preds[:5])
    print("Ground truth:", y_test[:5])
    r2 = r2_score(y_test, preds)
    print("R2 Score (Quantized PyTorch):", r2)
