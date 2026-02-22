"""
============================================================
  LINEAR REGRESSION FROM SCRATCH
  
  Covers:
    - Simple & Multiple Linear Regression
    - Gradient Descent (Batch)
    - Normal Equation (Closed-Form)
    - Feature Scaling (Standardization)
    - Performance Metrics: MAE, MSE, RMSE, R²
    - Loss curve visualization
    - Prediction on new data
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SECTION 1: THE LINEAR REGRESSION CLASS
# ============================================================

class LinearRegressionScratch:
    """
    Linear Regression implemented from scratch using NumPy.

    Supports two solving methods:
      - 'gradient_descent': iterative, scales to large datasets
      - 'normal_equation':  closed-form, exact solution in one step

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent (alpha). Default: 0.01
    n_iterations : int
        Number of gradient descent steps. Default: 1000
    method : str
        'gradient_descent' or 'normal_equation'
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, method='gradient_descent'):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method

        # These will be learned during training
        self.weights = None   # w (feature weights)
        self.bias = None      # b (intercept)
        self.loss_history = []  # track MSE loss per iteration

    # ----------------------------------------------------------
    # FEATURE SCALING — Standardization (Z-score)
    # ----------------------------------------------------------
    def _standardize(self, X):
        """
        Standardize features to mean=0, std=1.
        Formula: x_scaled = (x - mean) / std

        Why? Gradient descent converges much faster when all
        features are on the same scale. Without this, features
        with large ranges dominate the gradient updates.
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_  = np.std(X, axis=0)

        # Avoid division by zero for constant features
        self.std_[self.std_ == 0] = 1

        return (X - self.mean_) / self.std_
    
    def _scale_input(self, X):
        """Apply the stored mean/std from training to new data."""
        return (X - self.mean_) / self.std_

    # ----------------------------------------------------------
    # FORWARD PASS — Compute Predictions
    # ----------------------------------------------------------
    def _predict_raw(self, X):
        """
        Linear equation: ŷ = X·w + b
        Matrix multiplication gives us predictions for all
        m samples at once — fast and clean.
        """
        return np.dot(X, self.weights) + self.bias
    
    # ----------------------------------------------------------
    # LOSS FUNCTION — Mean Squared Error
    # ----------------------------------------------------------
    def _compute_loss(self, y_pred, y_true):
        """
        MSE Loss: J(θ) = (1/2m) · Σ(ŷᵢ - yᵢ)²

        The 1/2 factor is a math convenience — it cancels with
        the exponent's 2 when we differentiate, giving a cleaner
        gradient formula.
        """
        m = len(y_true)
        return (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)
    
    # ----------------------------------------------------------
    # GRADIENTS — Partial Derivatives of Loss w.r.t. θ
    # ----------------------------------------------------------
    def _compute_gradients(self, X, y_pred, y_true):
        """
        Gradient of MSE w.r.t. weights:  ∂J/∂w = (1/m) · Xᵀ · (ŷ - y)
        Gradient of MSE w.r.t. bias:     ∂J/∂b = (1/m) · Σ(ŷ - y)

        Derivation (for weight gradient):
          J = (1/2m) · Σ(ŷ - y)²
          ŷ = Xw + b
          ∂J/∂w = (1/m) · Xᵀ(ŷ - y)   ← chain rule

        The gradient vector points in the direction of steepest
        *increase* of loss, so we subtract it to descend.
        """
        m = len(y_true)
        error = y_pred - y_true              # shape: (m,)
        dw = (1 / m) * np.dot(X.T, error)   # shape: (n_features,)
        db = (1 / m) * np.sum(error)         # scalar
        return dw, db
    

    # ----------------------------------------------------------
    # METHOD A: GRADIENT DESCENT
    # ----------------------------------------------------------
    def _fit_gradient_descent(self, X, y):
        """
        Iteratively updates θ using:
          w := w - α · ∂J/∂w
          b := b - α · ∂J/∂b

        Each iteration = one full pass over the dataset (Batch GD).
        """
        m, n = X.shape

        # Initialize weights to zeros (or small random values)
        self.weights = np.zeros(n)
        self.bias = 0.0

        print(f"\n{'='*50}")
        print(f"  Training via Gradient Descent")
        print(f"  Learning Rate : {self.learning_rate}")
        print(f"  Iterations    : {self.n_iterations}")
        print(f"{'='*50}")

        for i in range(self.n_iterations):
            # Step 1: Forward pass
            y_pred = self._predict_raw(X)

            # Step 2: Compute loss
            loss = self._compute_loss(y_pred, y)
            self.loss_history.append(loss)

            # Step 3: Compute gradients
            dw, db = self._compute_gradients(X, y_pred, y)

            # Step 4: Update parameters (the actual "learning")
            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"  Iteration {i+1:5d} | Loss: {loss:.6f}")

        print(f"\n  ✅ Training Complete!")
        print(f"  Final Loss : {self.loss_history[-1]:.6f}")

    # ----------------------------------------------------------
    # METHOD B: NORMAL EQUATION (Closed-Form)
    # ----------------------------------------------------------
    def _fit_normal_equation(self, X, y):
        """
        Closed-form solution:
          θ = (XᵀX)⁻¹ · Xᵀ · y

        We augment X with a column of 1s to absorb the bias term b
        into the weight vector θ = [b, w₁, w₂, ..., wₙ].

        Advantage : Exact answer, no iterations needed.
        Drawback  : O(n³) complexity for matrix inversion —
                    gets slow when n_features > ~10,000.
        """
        print(f"\n{'='*50}")
        print(f"  Training via Normal Equation (Closed-Form)")
        print(f"{'='*50}")

        m = len(y)

        # Add bias column (column of 1s) to X
        X_b = np.hstack([np.ones((m, 1)), X])   # shape: (m, n+1)

        # θ = (XᵀX)⁻¹ · Xᵀ · y
        # np.linalg.pinv = pseudo-inverse (handles singular matrices gracefully)
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        self.bias = theta[0]          # first element is the bias
        self.weights = theta[1:]      # rest are feature weights

        print(f"  ✅ Solution found in one step (no iterations needed)!")
        print(f"  Bias    : {self.bias:.4f}")
        print(f"  Weights : {self.weights}")


        # ----------------------------------------------------------
    # PUBLIC API: FIT
    # ----------------------------------------------------------
    def fit(self, X, y):
        """
        Train the model on data X and labels y.

        Steps:
          1. Standardize features
          2. Solve for optimal weights
        """
        # Always scale features first
        X_scaled = self._standardize(X)

        if self.method == 'gradient_descent':
            self._fit_gradient_descent(X_scaled, y)
        elif self.method == 'normal_equation':
            self._fit_normal_equation(X_scaled, y)
        else:
            raise ValueError("method must be 'gradient_descent' or 'normal_equation'")

        return self
    
    # ----------------------------------------------------------
    # PUBLIC API: PREDICT
    # ----------------------------------------------------------
    def predict(self, X):
        """
        Generate predictions for new data X.
        Apply the same scaling learned during training.
        """
        X_scaled = self._scale_input(X)
        return self._predict_raw(X_scaled)
    




    # ----------------------------------------------------------
    # PUBLIC API: SCORE — All Metrics
    # ----------------------------------------------------------
    def score(self, X, y):
        """
        Compute and display all performance metrics:
          - MAE  : Mean Absolute Error
          - MSE  : Mean Squared Error
          - RMSE : Root Mean Squared Error
          - R²   : Coefficient of Determination
        """
        y_pred = self.predict(X)
        m = len(y)

        # --- MAE: (1/m) · Σ|yᵢ - ŷᵢ|
        mae = np.mean(np.abs(y - y_pred))

        # --- MSE: (1/m) · Σ(yᵢ - ŷᵢ)²
        mse = np.mean((y - y_pred) ** 2)

        # --- RMSE: √MSE
        rmse = np.sqrt(mse)

        # --- R²: 1 - (SS_res / SS_tot)
        ss_res = np.sum((y - y_pred) ** 2)          # residual sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)      # total sum of squares
        r2 = 1 - (ss_res / ss_tot)

        print(f"\n{'='*50}")
        print(f"  📊 PERFORMANCE METRICS")
        print(f"{'='*50}")
        print(f"  MAE  (Mean Absolute Error)      : {mae:.4f}")
        print(f"  MSE  (Mean Squared Error)       : {mse:.4f}")
        print(f"  RMSE (Root Mean Squared Error)  : {rmse:.4f}")
        print(f"  R²   (Coefficient of Det.)      : {r2:.4f}")
        print(f"\n  📌 Interpretation:")
        print(f"     → Predictions are off by ±{rmse:.2f} units on average (RMSE)")
        print(f"     → Model explains {r2*100:.1f}% of variance in the data (R²)")

        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    # ----------------------------------------------------------
    # PLOT: Loss Curve (Gradient Descent only)
    # ----------------------------------------------------------
    def plot_loss(self):
        """Visualize how the loss decreases over training iterations."""
        if not self.loss_history:
            print("No loss history — use gradient_descent method to see this.")
            return

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(self.loss_history, color='#e74c3c', linewidth=2)
        ax.set_title("Loss Curve (MSE over Gradient Descent Iterations)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("MSE Loss")
        ax.fill_between(range(len(self.loss_history)), self.loss_history, alpha=0.1, color='#e74c3c')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("/mnt/user-data/outputs/loss_curve.png", dpi=150)
        plt.show()
        print("  📈 Loss curve saved to loss_curve.png")

    # ----------------------------------------------------------
    # PLOT: Predictions vs Actuals (for simple regression)
    # ----------------------------------------------------------
    def plot_predictions(self, X, y, feature_name="Feature", target_name="Target"):
        """Scatter plot of actual vs predicted values (best for 1 feature)."""
        y_pred = self.predict(X)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # --- Left: Regression Line Plot (only works for 1 feature)
        if X.shape[1] == 1:
            x_sorted = np.sort(X[:, 0])
            y_line = self.predict(x_sorted.reshape(-1, 1))
            axes[0].scatter(X[:, 0], y, color='#3498db', alpha=0.6, label='Actual', s=40)
            axes[0].plot(x_sorted, y_line, color='#e74c3c', linewidth=2, label='Regression Line')
            axes[0].set_xlabel(feature_name)
            axes[0].set_ylabel(target_name)
            axes[0].set_title("Data & Regression Line", fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, "Regression line plot\nonly for 1 feature",
                        ha='center', va='center', transform=axes[0].transAxes, fontsize=12)

        # --- Right: Actual vs Predicted
        axes[1].scatter(y, y_pred, color='#2ecc71', alpha=0.6, s=40)
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
        axes[1].set_xlabel("Actual Values")
        axes[1].set_ylabel("Predicted Values")
        axes[1].set_title("Actual vs Predicted", fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle("Linear Regression — Model Evaluation", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig("/mnt/user-data/outputs/predictions_plot.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("  📊 Prediction plots saved to predictions_plot.png")


# ============================================================
# SECTION 2: HELPER — Train/Test Split
# ============================================================

def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    test_size = fraction of data used for testing (e.g., 0.2 = 20%)
    """
    np.random.seed(random_state)
    m = len(y)
    indices = np.random.permutation(m)
    test_count = int(m * test_size)

    test_idx  = indices[:test_count]
    train_idx = indices[test_count:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ============================================================
# SECTION 3: DEMO — Simple Linear Regression (1 Feature)
# ============================================================

def demo_simple_regression():
    print("\n" + "="*60)
    print("  DEMO 1: Simple Linear Regression (1 Feature)")
    print("="*60)

    # Generate synthetic data: y = 3.5x + 7 + noise
    np.random.seed(0)
    X = 2 * np.random.rand(150, 1)
    y = 3.5 * X[:, 0] + 7 + np.random.randn(150) * 0.8

    print(f"\n  Dataset: {X.shape[0]} samples, {X.shape[1]} feature")
    print(f"  True relationship: y = 3.5·x + 7 + noise")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train with Gradient Descent
    model = LinearRegressionScratch(learning_rate=0.1, n_iterations=500, method='gradient_descent')
    model.fit(X_train, y_train)

    # Evaluate
    print("\n  --- Training Set ---")
    model.score(X_train, y_train)

    print("\n  --- Test Set ---")
    model.score(X_test, y_test)

    # Visualize
    model.plot_loss()
    model.plot_predictions(X_test, y_test, feature_name="x", target_name="y")

    # Show what the model learned (in original scale context)
    print(f"\n  Learned Weights : {model.weights}")
    print(f"  Learned Bias    : {model.bias:.4f}")

    return model


# ============================================================
# SECTION 4: DEMO — Multiple Linear Regression (Multi-Feature)
# ============================================================

def demo_multiple_regression():
    print("\n" + "="*60)
    print("  DEMO 2: Multiple Linear Regression (3 Features)")
    print("="*60)

    # Generate synthetic data: y = 2x₁ + (-1.5)x₂ + 4x₃ + 10 + noise
    np.random.seed(42)
    m = 300
    X = np.random.randn(m, 3)
    true_weights = np.array([2.0, -1.5, 4.0])
    true_bias = 10.0
    y = X @ true_weights + true_bias + np.random.randn(m) * 1.5

    print(f"\n  Dataset: {m} samples, 3 features")
    print(f"  True weights: {true_weights}")
    print(f"  True bias   : {true_bias}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # --- Method A: Gradient Descent
    print("\n  [Method A] Gradient Descent")
    gd_model = LinearRegressionScratch(learning_rate=0.1, n_iterations=500, method='gradient_descent')
    gd_model.fit(X_train, y_train)
    print("\n  Test Set Results:")
    gd_model.score(X_test, y_test)

    # --- Method B: Normal Equation
    print("\n  [Method B] Normal Equation")
    ne_model = LinearRegressionScratch(method='normal_equation')
    ne_model.fit(X_train, y_train)
    print("\n  Test Set Results:")
    ne_model.score(X_test, y_test)

    # Compare recovered weights vs true weights
    print(f"\n  {'='*50}")
    print(f"  WEIGHT RECOVERY COMPARISON")
    print(f"  {'='*50}")
    print(f"  True weights          : {true_weights}")
    print(f"  GD recovered (approx) : both methods should closely match true weights")

    return gd_model, ne_model


# ============================================================
# SECTION 5: LEARNING RATE EXPERIMENT
# ============================================================

def demo_learning_rates():
    """
    Show how different learning rates affect convergence.
    Too small = slow. Too large = diverges. Just right = fast & stable.
    """
    print("\n" + "="*60)
    print("  DEMO 3: Learning Rate Experiment")
    print("="*60)

    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 3 * X[:, 0] + 5 + np.random.randn(100) * 0.5

    learning_rates = [0.001, 0.01, 0.1, 0.5]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    fig, ax = plt.subplots(figsize=(10, 5))

    for lr, color in zip(learning_rates, colors):
        model = LinearRegressionScratch(learning_rate=lr, n_iterations=200, method='gradient_descent')
        # Suppress prints for this demo
        import io, sys
        suppress = io.StringIO()
        sys.stdout = suppress
        model.fit(X, y)
        sys.stdout = sys.__stdout__

        if model.loss_history and not any(np.isnan(model.loss_history)):
            ax.plot(model.loss_history, label=f"α = {lr}", color=color, linewidth=2)
        else:
            print(f"  ⚠️  α={lr} diverged (loss exploded — learning rate too large)")

    ax.set_title("Effect of Learning Rate on Convergence", fontsize=13, fontweight='bold')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/learning_rates.png", dpi=150)
    plt.show()
    print("  📈 Learning rate comparison saved to learning_rates.png")


# ============================================================
# MAIN — Run All Demos
# ============================================================

if __name__ == "__main__":
    print("\n" + "🔷"*30)
    print("  LINEAR REGRESSION FROM SCRATCH")
    print("🔷"*30)

    # Run demos
    simple_model   = demo_simple_regression()
    gd_model, ne_model = demo_multiple_regression()
    demo_learning_rates()

    print("\n" + "="*60)
    print("  ✅ ALL DEMOS COMPLETE")
    print("  Files saved:")
    print("    → loss_curve.png")
    print("    → predictions_plot.png")
    print("    → learning_rates.png")
    print("="*60)



     # -------------------------------------------------------
    # QUICK REFERENCE: Using the model on YOUR OWN DATA
    # -------------------------------------------------------
    print("""
  HOW TO USE ON YOUR OWN DATA:
  ──────────────────────────────
  import numpy as np

  # 1. Load your data (X = features matrix, y = target vector)
  X = np.array([[...]])   # shape: (n_samples, n_features)
  y = np.array([...])     # shape: (n_samples,)

  # 2. Create and train the model
  model = LinearRegressionScratch(
      learning_rate=0.01,
      n_iterations=1000,
      method='gradient_descent'   # or 'normal_equation'
  )
  model.fit(X_train, y_train)

  # 3. Predict
  y_pred = model.predict(X_test)

  # 4. Evaluate
  metrics = model.score(X_test, y_test)

  # 5. Visualize
  model.plot_loss()
  model.plot_predictions(X_test, y_test)
""")