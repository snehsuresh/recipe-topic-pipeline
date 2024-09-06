import boto3
import joblib
import pandas as pd
from io import BytesIO
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import os

s3_client = boto3.client(
    "s3",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
# Initialize S3 client
s3_client = boto3.client("s3")
BUCKET_NAME = "recipe-recommender-data"


def save_model(model, model_name):
    model_bytes = BytesIO()
    joblib.dump(model, model_bytes)
    model_bytes.seek(0)

    try:
        s3_client.upload_fileobj(
            model_bytes, BUCKET_NAME, f"models/{model_name}.joblib"
        )
        print(f"Model {model_name} saved to S3 bucket {BUCKET_NAME} in 'models' folder")
    except (NoCredentialsError, PartialCredentialsError) as e:
        print(f"Credentials error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def load_model(model_name):
    try:
        # Download the model from S3 to a BytesIO object
        model_bytes = BytesIO()
        s3_client.download_fileobj(
            BUCKET_NAME, f"models/{model_name}.joblib", model_bytes
        )
        model_bytes.seek(0)

        # Load the model
        model = joblib.load(model_bytes)
        print(
            f"Model {model_name} loaded from S3 bucket {BUCKET_NAME} in 'models' folder"
        )
        return model
    except (NoCredentialsError, PartialCredentialsError) as e:
        print(f"Credentials error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None


def save_dataframe(df, filename):
    # Save the DataFrame to a CSV in a BytesIO object
    csv_bytes = BytesIO()
    df.to_csv(csv_bytes, index=False, encoding="utf-8")
    csv_bytes.seek(0)

    # Upload to S3
    try:
        s3_client.upload_fileobj(csv_bytes, BUCKET_NAME, f"data/processed/{filename}")
        print(f"DataFrame saved to S3 bucket {BUCKET_NAME} in 'data/processed' folder")
    except (NoCredentialsError, PartialCredentialsError) as e:
        print(f"Credentials error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def load_dataframe(filename):
    try:
        # Download the CSV from S3 to a BytesIO object
        csv_bytes = BytesIO()
        s3_client.download_fileobj(BUCKET_NAME, f"data/processed/{filename}", csv_bytes)
        csv_bytes.seek(0)

        # Load the DataFrame
        df = pd.read_csv(csv_bytes, encoding="utf-8")
        print(
            f"DataFrame loaded from S3 bucket {BUCKET_NAME} in 'data/processed' folder"
        )
        return df
    except (NoCredentialsError, PartialCredentialsError) as e:
        print(f"Credentials error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None
