#Python file for algorithm
from gettext import install
import pip
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import time
 
# fetch dataset 
student_performance = fetch_ucirepo(id=320) 
  
# data (as pandas dataframes) 
X = student_performance.data.features 
y = student_performance.data.targets 
  
# metadata 
print(student_performance.metadata) 
  
# variable information 
print(student_performance.variables) 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Import the hardcoded DecisionTree class
from decision_tree_classifier import DecisionTree 

# --- 1. DATA FETCHING ---

print("Fetching dataset...")
# Fetch dataset (Student Performance)
student_performance = fetch_ucirepo(id=320)

X_raw = student_performance.data.features
y_raw = student_performance.data.targets


# --- 2. PREPROCESSING AND TARGET DEFINITION ---

def preprocess_data(X_in, y_in):
    """
    Cleans data, converts categorical features to numeric, and defines the binary target.
    Pass/Fail Threshold: G3 >= 10 is 'Pass', G3 < 10 is 'Fail'.
    """
    print("Preprocessing data and defining target...")

    # Create the binary target variable (y)
    # Define the mapping for the custom DecisionTree class (0 and 1 integers)
    target_map = {'Fail': 0, 'Pass': 1} 
    y_target_str = y_in['G3'].apply(lambda g: 'Pass' if g >= 10 else 'Fail')
    # Convert string labels to integers (crucial for the DecisionTree class)
    y_target_int = y_target_str.map(target_map).astype(int)
    
    # Drop the original grade columns from the features (X)
    X_features = X_in.drop(columns=['G3'], errors='ignore')
    
    # Convert all remaining categorical features to numerical using One-Hot Encoding
    X_processed = pd.get_dummies(X_features, drop_first=True)
    
    return X_processed, y_target_int, target_map['Pass']

X, y_int, PASS_LABEL_INT = preprocess_data(X_raw, y_raw)

# Split data into training (80%) and testing (20%) - as per proposal
# Need to use y_int here for stratify
X_train_df, X_test_df, y_train_int, y_test_int = train_test_split(
    X, y_int, test_size=0.20, random_state=42, stratify=y_int
)

# Convert Pandas objects to NumPy arrays for the hardcoded class
X_train = X_train_df.values.astype(float)
X_test = X_test_df.values.astype(float)
y_train = y_train_int.values
y_test = y_test_int.values

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print(f"Number of Features (m): {X_train.shape[1]}")
print(f"Target Labels mapped to: 0 (Fail), 1 (Pass)")


# --- 3. MODEL TRAINING, TESTING, AND ANALYSIS ---

# Set max depth for analysis 
MAX_DEPTH = 5 

print("\n--- Training Hardcoded Decision Tree (Entropy/IG) ---")

# Initialize and train the model
model = DecisionTree(max_depth=MAX_DEPTH)

start_time_ns = time.perf_counter_ns()
model.train(X_train, y_train)
training_runtime_us = (time.perf_counter_ns() - start_time_ns) / 1000

print(f"Training Complete. Max Depth: {MAX_DEPTH}")


print("\n--- Testing and Metrics Calculation ---")

# Predict using the model
start_time_ns = time.perf_counter_ns()
y_pred = model.predict(X_test)
testing_runtime_us = (time.perf_counter_ns() - start_time_ns) / 1000


# Calculate Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
# Use PASS_LABEL_INT (1) as the positive class for Precision, Recall, F1-Score
precision = precision_score(y_test, y_pred, pos_label=PASS_LABEL_INT, zero_division=0)
recall = recall_score(y_test, y_pred, pos_label=PASS_LABEL_INT, zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label=PASS_LABEL_INT, zero_division=0)


# --- 4. REPORTING RESULTS ---
print("---------------------------------------------------------")
print(f"Hardcoded Decision Tree (Greedy) Algorithm Results (Entropy/IG)")
print("---------------------------------------------------------")
print(f"1. Accuracy: {accuracy:.4f}")
print(f"2. Precision (Pass): {precision:.4f}")
print(f"3. Recall (Pass): {recall:.4f}")
print(f"4. F1-Score (Pass): {f1:.4f}")
print(f"5. Experimental Training Runtime: {training_runtime_us:.2f} µs (microseconds)")
print(f"6. Experimental Testing Runtime: {testing_runtime_us:.2f} µs (microseconds)")
print("\nTop 5 Feature Importances (Index: Value):")
# Print top 5 features by importance
feature_names = X_train_df.columns
sorted_importances = sorted(model.feature_importances.items(), key=lambda item: item[1], reverse=True)
for idx, importance in sorted_importances[:5]:
    print(f"   {feature_names[idx]}: {importance:.4f}")

# print("\n--- Tree Structure (First few levels) ---")
# model.print_tree()

