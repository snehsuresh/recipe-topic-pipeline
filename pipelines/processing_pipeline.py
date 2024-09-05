import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent  # Goes up one level to the root
sys.path.insert(0, str(project_root))


from src.RecipeRecommendationSystem.components.data_extraction import Extract
from src.RecipeRecommendationSystem.components.data_processing import DataProcessing
from src.RecipeRecommendationSystem.utils.utils import save_dataframe


def start_processing():
    # Extract data
    extractor = Extract()
    df_copy = extractor.extract_data()

    # Process data
    processor = DataProcessing()
    processed_df = processor.process_data(df_copy)
    save_dataframe(processed_df, "cleaned_dataframe.csv")

    # Optionally, you can save or further utilize processed_df
    print("Processing complete.")


if __name__ == "__main__":
    start_processing()
