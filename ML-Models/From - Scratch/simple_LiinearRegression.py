"""
Simple Linear Regression — Normal Equation Only
================================================
No gradient descent, no learning rate, no complexity.
Just the math: θ = (XᵀX)⁻¹ · Xᵀ · y
"""

import numpy as np
import matplotlib.pyplot as plt


# ── 1. THE MODEL ──────────────────────────────────────────────

class LinearRegression:

    def fit(self, X, y):
        """
        Learn the best weights using the Normal Equation.
        We add a column of 1s to X so the bias term is handled automatically.
        """
        # Add bias column (column of 1s) to X
        X_b = np.hstack([np.ones((len(X), 1)), X])

        # Normal Equation: θ = (XᵀX)⁻¹ · Xᵀ · y
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        self.bias    = theta[0]    # intercept (b)
        self.weights = theta[1:]   # feature weights (w)

    def predict(self, X):
        """ ŷ = X·w + b """
        return X @ self.weights + self.bias

    def score(self, X, y):
        """ Compute MAE, RMSE, and R² """
        y_pred = self.predict(X)

        mae  = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        r2   = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

        print(f"\n  MAE  : {mae:.4f}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  R²   : {r2:.4f}  → model explains {r2*100:.1f}% of the variance")


# ── 2. GENERATE SOME DATA ─────────────────────────────────────

np.random.seed(0)
X = 2 * np.random.rand(100, 1)         # 100 samples, 1 feature
y = 4 + 3 * X[:, 0] + np.random.randn(100)  # true: y = 3x + 4 + noise


# ── 3. TRAIN ──────────────────────────────────────────────────

model = LinearRegression()
model.fit(X, y)

print(f"Learned weight (slope)     : {model.weights[0]:.4f}  (true: 3.0)")
print(f"Learned bias  (intercept)  : {model.bias:.4f}        (true: 4.0)")


# ── 4. EVALUATE ───────────────────────────────────────────────

model.score(X, y)


# ── 5. PLOT ───────────────────────────────────────────────────

x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='steelblue', alpha=0.6, label='Data points')
plt.plot(x_line, y_line, color='red', linewidth=2, label='Regression line')
plt.title("Linear Regression — Normal Equation", fontsize=13, fontweight='bold')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("represent.png", dpi=150)
plt.show()
print("\nPlot saved!")


"""
OUTPUT:
Learned weight (slope)     : 2.9685  (true: 3.0)
Learned bias  (intercept)  : 4.2222        (true: 4.0)

  MAE  : 0.8493
  RMSE : 0.9962
  R²   : 0.7470  → model explains 74.7% of the variance



"""