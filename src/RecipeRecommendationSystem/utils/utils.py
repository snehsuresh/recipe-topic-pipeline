import os
import joblib
import pandas as pd


def save_model(model, model_name):
    # Define the folder to save models
    models_folder = "models"
    os.makedirs(models_folder, exist_ok=True)  # Create the folder if it doesn't exist

    # Save the model
    model_path = os.path.join(models_folder, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Model {model_name} saved to {model_path}")


def load_model(model_name):
    # Define the folder where models are saved
    models_folder = "models"
    model_path = os.path.join(models_folder, f"{model_name}.joblib")

    # Check if the model file exists
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Model {model_name} loaded from {model_path}")
        return model
    else:
        print(f"Model {model_name} does not exist at {model_path}")
        return None


def save_dataframe(df, filename):
    # Define the folder to save the DataFrame
    save_folder = "data/processed"
    os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist

    # Save the DataFrame as a CSV file
    save_path = os.path.join(save_folder, filename)
    df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"DataFrame saved to {save_path}")


def load_dataframe(filename):
    # Define the folder where the DataFrame was saved
    save_folder = "data/processed"
    save_path = os.path.join(save_folder, filename)

    # Check if the file exists before loading
    if os.path.exists(save_path):
        df = pd.read_csv(save_path, encoding="utf-8")
        print(f"DataFrame loaded from {save_path}")
        return df
    else:
        print(f"File {filename} does not exist in {save_folder}")
        return None
