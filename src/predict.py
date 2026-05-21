import os
import joblib
import pandas as pd
from sklearn.datasets import load_iris

def run_prediction():
    model_path = "models/random_forest_iris.joblib"
    
    # 1. Check if the model exists before loading
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'.")
        print("Please run 'src/train.py' first to train and save the model.")
        return

    # 2. Load the trained model and target/species names
    print(f"Loading trained model from '{model_path}'...")
    model = joblib.load(model_path)
    
    iris = load_iris()
    species_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']
    feature_names = iris.feature_names

    # 3. Define sample data for prediction
    # You can read from data/test.csv, or define explicit test features here:
    print("\nSetting up sample test instances...")
    samples = [
        [5.1, 3.5, 1.4, 0.2],  # Expected: setosa
        [6.7, 3.0, 5.2, 2.3]   # Expected: virginica
    ]
    
    # Convert samples to a DataFrame to maintain valid feature names matching train.py
    X_samples = pd.DataFrame(samples, columns=feature_names)

    # 4. Generate predictions and probability/confidence scores
    predictions = model.predict(X_samples)
    probabilities = model.predict_proba(X_samples)

    # 5. Output results sequentially
    print("\n--- Inference Results ---")
    for i, (pred_idx, prob_scores) in enumerate(zip(predictions, probabilities)):
        predicted_species = species_names[pred_idx]
        
        print(f"\nSample {i + 1} Features: {samples[i]}")
        print(f"Predicted Class: {predicted_species.upper()}")
        print("Confidence Scores per Species:")
        
        for species, score in zip(species_names, prob_scores):
            print(f"  - {species.capitalize()}: {score * 100:.2f}%")

if __name__ == "__main__":
    run_prediction()