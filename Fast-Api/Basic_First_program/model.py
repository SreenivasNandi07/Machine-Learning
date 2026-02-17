# model.py
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Train once → Save model → Load → Predict → Respond


# Sample data
X = np.array([[500], [800], [1000], [1500], [2000]])
y = np.array([1000000, 1600000, 2000000, 3000000, 4000000])

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
