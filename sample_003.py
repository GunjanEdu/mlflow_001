import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

# Set the tracking URI to the local server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create a new experiment or set the existing one
experiment_name = "demo_mlflow"
mlflow.set_experiment(experiment_name)

# Load data
df = pd.read_csv("data.csv")
#Dropping columns that are not needed
df = df.drop(columns=['id', 'Unnamed: 32'])

#Map the target to binary values: 'M' to 1 (malignant), 'B' to 0 (benign)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and target datasets
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=222)

# Start an MLflow run
with mlflow.start_run():
    # Create and train a model
    model = RandomForestClassifier(n_estimators=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Log parameters
    mlflow.log_param("n_estimators", 1000)
    mlflow.log_param("random_state", 42)
    
    # Log metrics
    accuracy = accuracy_score(y_test, predictions)
    mlflow.log_metric("accuracy", accuracy)
    
    rec_2 = '22222'
    mlflow.log_metric("recall_1", rec_2)
    
    
    rec_1 = '33333'
    mlflow.log_metric("recall_2", rec_1)
    # Log the model
    mlflow.sklearn.log_model(model, "model")

    print(f"Model accuracy: {accuracy}")

# Print out the experiment ID
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
print(f"Experiment ID: {experiment_id}")
