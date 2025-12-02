
import numpy as np
import pandas as pd
import time
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# PART 1: NAIVE BAYES CLASS (HARDCODED)
# ==========================================
class HardcodedGaussianNB:
    """
    Hardcoded Gaussian Naive Bayes Classifier.
    Implementation from scratch using NumPy.
    """
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Initialize mean, variance, and priors for each class
        self._mean = np.zeros((n_classes, n_features))
        self._var = np.zeros((n_classes, n_features))
        self._priors = np.zeros(n_classes)

        for idx, c in enumerate(self._classes):
            # Filter data for the specific class
            X_c = X[y == c]
            
            # Calculate mean and variance for features given the class
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            
            # Calculate prior probability P(y)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        posteriors = []

        # Calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            # Prior: log(P(y))
            prior = np.log(self._priors[idx])
            
            # Likelihood: sum(log(P(x_i|y)))
            # We add a small epsilon (1e-9) to variance to prevent division by zero
            class_var = self._var[idx] + 1e-9
            class_mean = self._mean[idx]
            
            numerator = np.exp(- (x - class_mean)**2 / (2 * class_var))
            denominator = np.sqrt(2 * np.pi * class_var)
            
            # Probability density function (add epsilon to avoid log(0))
            pdf = numerator / denominator
            
            # Sum of logs (to prevent underflow)
            posterior = prior + np.sum(np.log(pdf + 1e-9)) 
            posteriors.append(posterior)
        
        # Return class with highest posterior probability
        return self._classes[np.argmax(posteriors)]


# ==========================================
# PART 2: EXECUTION & METRICS
# ==========================================
print("Fetching dataset for Naive Bayes...")
try:
    student_performance = fetch_ucirepo(id=320)
    X_raw = student_performance.data.features
    y_raw = student_performance.data.targets
except ImportError:
    print("ERROR: 'ucimlrepo' not found. Please run '!pip install ucimlrepo' first.")
    raise

# Preprocessing
target_map = {'Fail': 0, 'Pass': 1} 
# Define target based on G3
y_target_int = y_raw['G3'].apply(lambda g: 'Pass' if g >= 10 else 'Fail').map(target_map).astype(int)
# Drop G3 to avoid data leakage and one-hot encode
X_processed = pd.get_dummies(X_raw.drop(columns=['G3'], errors='ignore'), drop_first=True)

# Split (Same random_state as Decision Tree for fair comparison)
# FIX: Explicitly cast values to float to prevent object dtype issues (which caused the TypeError)
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_processed.values.astype(float), y_target_int.values, test_size=0.20, random_state=42, stratify=y_target_int
)

# Run
print("\n--- Training Hardcoded Naive Bayes ---")
model = HardcodedGaussianNB()

# Measure Training Time
start = time.perf_counter_ns()
model.fit(X_train_df, y_train)
train_time = (time.perf_counter_ns() - start) / 1000

print("Training Complete. Predicting...")

# Measure Testing Time
start = time.perf_counter_ns()
y_pred = model.predict(X_test_df)
test_time = (time.perf_counter_ns() - start) / 1000

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

print("\n" + "="*40)
print(f"{'NAIVE BAYES RESULTS':^40}")
print("="*40)
print(f"Accuracy:      {acc:.4f}")
print(f"Precision:     {prec:.4f}")
print(f"Recall:        {rec:.4f}")
print(f"F1-Score:      {f1:.4f}")
print("-" * 40)
print(f"Training Time: {train_time:.2f} µs")
print(f"Testing Time:  {test_time:.2f} µs")
print("="*40)
