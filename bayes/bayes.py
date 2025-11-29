import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Load the dataset
data = pd.read_csv("student-mat.csv", sep=";") 

#Create target label: Pass / Fail based on final grade G3
PASS_THRESHOLD = 10
data["Pass"] = (data["G3"] >= PASS_THRESHOLD).astype(int)  # 1 = Pass, 0 = Fail

#Define features (X) and label (y)
X = data.drop(columns=["G3", "Pass"])
y = data["Pass"]

#Identify categorical and numerical columns
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

#Preprocessing

numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

#Build the full pipeline: preprocessing + Gaussian Naive Bayes
nb_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", GaussianNB())
])

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Train the model
nb_model.fit(X_train, y_train)

#Evaluate on test set
y_pred = nb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Naive Bayes Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
