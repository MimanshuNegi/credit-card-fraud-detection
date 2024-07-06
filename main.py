import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import zipfile

# Load the dataset
print("Loading dataset...")

# Path to the zip file
zip_file_path = 'Dataset/creditcard.zip'

# Open the zip file and read the CSV
with zipfile.ZipFile(zip_file_path, 'r') as z:
    csv_file_name = z.namelist()[0]
    with z.open(csv_file_name) as f:
        data = pd.read_csv(f)
print("Dataset loaded successfully.")

# Explore the dataset
print("First five rows of the dataset:")
print(data.head())

# Handle missing values (if any)
print("Handling missing values...")
data = data.dropna()
print("Missing values handled.")

# Separate features and target variable
print("Separating features and target variable...")
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target variable (fraudulent or legitimate)
print("Features and target variable separated.")

# Normalize the features
print("Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features normalized.")

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print("Data split into training and testing sets.")

# Initialize the models
log_reg = LogisticRegression()
dec_tree = DecisionTreeClassifier()

# Train the models
print("Training Logistic Regression model...")
log_reg.fit(X_train, y_train)
print("Logistic Regression model trained.")

print("Training Decision Tree model...")
dec_tree.fit(X_train, y_train)
print("Decision Tree model trained.")

# Function to evaluate a model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# Evaluate the Logistic Regression model
print("Evaluating Logistic Regression model...")
log_reg_metrics = evaluate_model(log_reg, X_test, y_test)
print(f"Logistic Regression - Accuracy: {log_reg_metrics[0]}, Precision: {log_reg_metrics[1]}, Recall: {log_reg_metrics[2]}, F1 Score: {log_reg_metrics[3]}")

# Evaluate the Decision Tree model
print("Evaluating Decision Tree model...")
dec_tree_metrics = evaluate_model(dec_tree, X_test, y_test)
print(f"Decision Tree - Accuracy: {dec_tree_metrics[0]}, Precision: {dec_tree_metrics[1]}, Recall: {dec_tree_metrics[2]}, F1 Score: {dec_tree_metrics[3]}")

# Choose the best model based on F1 Score
best_model = log_reg if log_reg_metrics[3] >= dec_tree_metrics[3] else dec_tree

print(f"Best model: {'Logistic Regression' if best_model == log_reg else 'Decision Tree'}")

# Save the best model and scaler
print("Saving the best model and scaler...")
joblib.dump(best_model, 'model/best_fraud_detection_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
print("Model and scaler saved successfully.")
