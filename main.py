# main.py
import subprocess
import sys  # <--- Make sure to import sys

print("=== Starting ML Pipeline ===")

# 1. Train the model using the active environment's interpreter
# Using sys.executable passes down the uv environment automatically
subprocess.run([sys.executable, "src/train.py"], check=True)

# 2. Run prediction evaluation
subprocess.run([sys.executable, "src/predict.py"], check=True)

print("=== Pipeline Execution Finished Successfully ===")