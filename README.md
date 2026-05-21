# Iris Classification ML Pipeline Containerization

This repository contains a modular, containerized end-to-end machine learning pipeline that trains a `RandomForestClassifier` on the Iris dataset, evaluates its performance, and validates inference outputs. The system is fully containerized using Docker and `uv` for seamless execution.

## Setup and Execution Guide

### Prerequisites
* **Docker Desktop** must be installed and running on your system.

### Step 1: Build the Docker Image
Open your terminal in the root directory of the project (where the `Dockerfile` is located) and run the following command to compile the container image:

```bash
docker build -t iris-classification .
```

*Note: If you need to force a clean build without using Docker's cached layers, use:*
```bash
docker build --no-cache -t iris-classification .
```

### Step 2: Run the Pipeline Container
Execute the compiled image. The `--rm` flag automatically cleans up the container instance after the execution completes:

```bash
docker run --rm iris-classification
```

## Expected Output

When the container launches, the central orchestrator (`main.py`) will run the training and prediction scripts sequentially. You should expect the following exact printout in your terminal:

```text
==================================================
   STARTING MACHINE LEARNING CONTAINER PIPELINE   
==================================================

[STEP 1/2] Running model training workflow...
Loading Iris dataset...
Splitting data into train and test sets (80/20)...
Saving dataset splits to data/ directory...
Training RandomForestClassifier...

Model Training Complete.
Test Accuracy: 90.00%
Trained model successfully saved to '/app/models/random_forest_iris.joblib'

[+] Step 1 completed successfully.
--------------------------------------------------

[STEP 2/2] Running prediction and validation workflow...
Loading trained model from 'models/random_forest_iris.joblib'...

Setting up sample test instances...

--- Inference Results ---

Sample 1 Features: [5.1, 3.5, 1.4, 0.2]
Predicted Class: SETOSA
Confidence Scores per Species:
  - Setosa: 100.00%
  - Versicolor: 0.00%
  - Virginica: 0.00%

Sample 2 Features: [6.7, 3.0, 5.2, 2.3]
Predicted Class: VIRGINICA
Confidence Scores per Species:
  - Setosa: 0.00%
  - Versicolor: 2.00%
  - Virginica: 98.00%

[+] Step 2 completed successfully.

==================================================
   PIPELINE EXECUTION COMPLETED END-TO-END!      
==================================================
```