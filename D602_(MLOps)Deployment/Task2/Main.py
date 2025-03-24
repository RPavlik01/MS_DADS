import argparse
import subprocess
import sys


def run_command(command):
    result = subprocess.run(command, shell=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    # Set up argument parser to receive parameters from MLFlow
    parser = argparse.ArgumentParser(description="Run the full pipeline")
    parser.add_argument("data", type=str, help="Path to the dataset file")
    parser.add_argument("num_alphas", type=int, help="Number of alpha increments")
    args = parser.parse_args()

    # Run data import/formatting with the data parameter
    run_command(["python", "Steps/StepB_Import_Format.py", "--data", args.data])

    # Run data filtering and cleaning (assuming it doesn't require extra parameters)
    run_command(["python", "Steps/StepC_Filter_Clean.py"])

    # Run model training with the num_alphas parameter
    run_command(["python", "Steps/poly_regressor_Python_1.0.0.py", '--num_alphas', str(args.num_alphas)])
