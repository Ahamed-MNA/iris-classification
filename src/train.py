import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def train_model():
    # Get the project root directory (one level up from 'src')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define absolute paths for data and models directories
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")

    # 1. Ensure required directories exist explicitly
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("Loading Iris dataset...")
    # 2. Load the Iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")

    # 3. Split into training and test sets (80/20)
    print("Splitting data into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Save the train/test splits as CSV files inside the data/ folder
    print("Saving dataset splits to data/ directory...")
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)

    # 5. Initialize and train the RandomForestClassifier
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6. Evaluate and print test accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Training Complete.")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # 7. Save the trained model to disk using joblib
    model_path = os.path.join(models_dir, "random_forest_iris.joblib")
    joblib.dump(model, model_path)
    print(f"Trained model successfully saved to '{model_path}'\n")

if __name__ == "__main__":
    train_model()