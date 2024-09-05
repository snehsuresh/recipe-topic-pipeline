import pandas as pd
from pathlib import Path
from src.RecipeRecommendationSystem.utils.utils import save_dataframe


class Extract:
    def extract_data(self):
        csv_files = Path("data/output").glob("*.csv")

        df_list = [
            pd.read_csv(file, encoding="utf-8", delimiter=",") for file in csv_files
        ]
        recipes_df = pd.concat(df_list, ignore_index=True)

        # count non-NaN values in the 'directions' column
        non_nan_directions_count = recipes_df["directions"].notna().sum()

        print(f"Number of non-NaN values in 'directions': {non_nan_directions_count}")

        df_copy = recipes_df.copy()

        df_copy = df_copy.dropna(subset=["directions", "ingredients"])

        # Remove all rows with any NaN values
        df_copy = df_copy.dropna()

        df_copy["combined_text"] = df_copy.apply(
            lambda row: f"{row['directions']} {' '.join(row['ingredients'].values()) if isinstance(row['ingredients'], dict) else ''}",
            axis=1,
        )
        return df_copy


if __name__ == "__main__":
    extract = Extract()
    extract.extract_data()
