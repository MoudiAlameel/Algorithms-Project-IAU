import csv
import math
import time
import random

# ===========================================================
# 1. Load CSV (UCI student-mat.csv with ; separator) - LOCAL
# ===========================================================

def load_csv_local(filename="student-mat.csv"):
    data = []
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=";")
        header = next(reader)        # column names
        for row in reader:
            if row:
                data.append(row)
    print("Header columns:", header)
    return data, header


# ===========================================================
# 2. Encode Dataset (Numeric + Pass/Fail target)
# ===========================================================

def encode_dataset(data, header):
    processed = []

    if "G3" not in header:
        raise ValueError(f'"G3" not found in header: {header}')

    g3_index = header.index("G3")

    for row in data:
        new_row = []

        for i, value in enumerate(row):
            # skip G3 (it is the target)
            if i == g3_index:
                continue

            # try numeric
            try:
                new_row.append(float(value))
            except:
                # simple numeric encoding for categorical values
                new_row.append(hash(value) % 500)

        # target: Pass/Fail based on G3
        g3_value = float(row[g3_index])
        target = 1 if g3_value >= 10 else 0   # 1 = Pass, 0 = Fail
        new_row.append(target)
        processed.append(new_row)

    return processed


# ===========================================================
# 3. Manual Train/Test Split
# ===========================================================

def train_test_split_manual(data, test_ratio=0.2):
    random.shuffle(data)
    cut = int(len(data) * (1 - test_ratio))
    return data[:cut], data[cut:]


# ===========================================================
# 4. Hardcoded Gaussian Naive Bayes
# ===========================================================

def mean(col):
    return sum(col) / len(col)

def std(col):
    m = mean(col)
    variance = sum((x - m) ** 2 for x in col) / (len(col) - 1)
    return math.sqrt(variance) if variance > 0 else 1e-6  # avoid zero

def gaussian(x, mu, sigma):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def separate_by_class(data):
    separated = {}
    for row in data:
        label = row[-1]
        if label not in separated:
            separated[label] = []
        separated[label].append(row)
    return separated

def train_naive_bayes(training_data):
    separated = separate_by_class(training_data)
    model = {}

    for c, rows in separated.items():
        summaries = []
        cols = list(zip(*rows))
        # all columns except the last one (label)
        for col in cols[:-1]:
            col = list(map(float, col))
            summaries.append((mean(col), std(col)))
        model[c] = summaries

    return model, separated

def calculate_probabilities(model, separated, row):
    probs = {}
    total = sum(len(separated[c]) for c in separated)

    for c in model:
        prior = len(separated[c]) / total
        probs[c] = prior

        for i in range(len(model[c])):
            mu, sigma = model[c][i]
            probs[c] *= gaussian(row[i], mu, sigma)

    return probs

def predict(model, separated, row):
    probs = calculate_probabilities(model, separated, row)
    return max(probs, key=probs.get)

def evaluate(model, separated, test):
    correct = 0
    for row in test:
        y_true = row[-1]
        y_pred = predict(model, separated, row)
        if y_true == y_pred:
            correct += 1
    return correct / len(test)


# ===========================================================
# 4.1 Feature Importance for Naive Bayes (Statistical Proxy)
# ===========================================================

def compute_feature_importance_nb(model, feature_names):
    """
    Approximate feature importance:
    |mean_pass - mean_fail| for each feature.
    """
    importance = []

    # assume model[0] = Fail, model[1] = Pass
    for i in range(len(model[0])):
        mu_fail, std_fail = model[0][i]
        mu_pass, std_pass = model[1][i]

        score = abs(mu_pass - mu_fail)
        feature_name = feature_names[i]
        importance.append((feature_name, score))

    importance.sort(key=lambda x: x[1], reverse=True)
    return importance


# ===========================================================
# 5. MAIN (VS Code / Local Python)
# ===========================================================

if __name__ == "__main__":
    print("➡ Loading dataset (student-mat.csv, ; separated)...")
    raw_data, header = load_csv_local("student-mat.csv")

    # feature names without the target G3 (to align with encoded data)
    feature_names = [h for h in header if h != "G3"]

    print("➡ Encoding dataset (features + Pass/Fail based on G3)...")
    dataset = encode_dataset(raw_data, header)
    print(f"Total samples: {len(dataset)}")

    print("➡ Splitting into train/test...")
    train, test = train_test_split_manual(dataset, test_ratio=0.2)
    print(f"Train size: {len(train)}, Test size: {len(test)}")

    print("➡ Training Hardcoded Naive Bayes...")
    start = time.time()
    model, separated = train_naive_bayes(train)
    train_time_ms = (time.time() - start) * 1000
    print(f"⏱ Training Time: {train_time_ms:.3f} ms")

    print("➡ Evaluating on test set...")
    accuracy = evaluate(model, separated, test)
    print(f" Naive Bayes (Hardcoded) Accuracy: {accuracy:.4f}")

    # =======================================================
    # 6. Feature Importance Output (Top 5)
    # =======================================================
    feature_importance = compute_feature_importance_nb(model, feature_names)

    print("\nTop 5 Feature Importances (Index: Value):")
    for feature, score in feature_importance[:5]:
        print(f"{feature}: {score:.4f}")
