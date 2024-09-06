from src.RecipeRecommendationSystem.components.data_processing import DataProcessing
from src.RecipeRecommendationSystem.components.model_trainer import Trainer
from src.RecipeRecommendationSystem.utils.utils import load_model, load_dataframe
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Recommendation:
    def recommend(self, ingredients, model):
        processor = DataProcessing()
        trainer = Trainer()
        user_input_processed = processor.preprocess_text(ingredients)
        user_input_dtm = trainer.generate_vectors(user_input_processed, 1)
        dtm = load_model("document_term_matrix")
        df = load_dataframe("cleaned_dataframe.csv")
        if model == "lda":
            # LDA
            lda_model = load_model("lda_model")
            document_lda_topic_features = lda_model.transform(dtm)
            user_input_lda_features = lda_model.transform(user_input_dtm)
            similarities = cosine_similarity(
                user_input_lda_features, document_lda_topic_features
            )
            top_indices = similarities[0].argsort()[-5:][::-1]
            # Get top 5 recommended recipes
            recommended_recipes = df.iloc[top_indices]

            recommended_recipes_list = recommended_recipes[
                ["title", "directions"]
            ].to_dict(orient="records")

            suggested_recipes = {
                "message": f"Recipes suggested based on ingredients: {ingredients}",
                "recommended_recipes": recommended_recipes_list,
            }
            return suggested_recipes

        elif model == "nmf":
            nmf_model = load_model("nmf_model")
            document_nmf_topic_features = nmf_model.transform(dtm)
            user_input_nmf_features = nmf_model.transform(user_input_dtm)
            similarities = cosine_similarity(
                user_input_nmf_features, document_nmf_topic_features
            )
            top_indices = similarities[0].argsort()[-5:][::-1]
            # Get top 5 recommended recipes
            recommended_recipes = df.iloc[top_indices]

            recommended_recipes_list = recommended_recipes[
                ["title", "directions"]
            ].to_dict(orient="records")

            suggested_recipes = {
                "message": f"Recipes suggested based on ingredients: {ingredients}",
                "recommended_recipes": recommended_recipes_list,
            }
            return suggested_recipes
            # NMF
        elif model == "ensemble":
            lda_model = load_model("lda_model")
            nmf_model = load_model("nmf_model")
            document_lda_topic_features = lda_model.transform(dtm)
            document_nmf_topic_features = nmf_model.transform(dtm)
            combined_topic_features = np.hstack(
                (document_lda_topic_features, document_nmf_topic_features)
            )
            user_input_lda_features = lda_model.transform(user_input_dtm)
            user_input_nmf_features = nmf_model.transform(user_input_dtm)

            # Combine topic features from both models
            user_input_combined_features = np.hstack(
                (user_input_lda_features, user_input_nmf_features)
            )
            similarities = cosine_similarity(
                user_input_combined_features, combined_topic_features
            )
            # Print the similarity scores for debugging
            print("Similarity Scores:", similarities)

            df = load_dataframe("cleaned_dataframe.csv")
            top_indices = similarities[0].argsort()[-5:][
                ::-1
            ]  # Get top 5 recommended recipes
            recommended_recipes = df.iloc[top_indices]

            recommended_recipes_list = recommended_recipes[
                ["title", "directions"]
            ].to_dict(orient="records")

            suggested_recipes = {
                "message": f"Recipes suggested based on ingredients: {ingredients}",
                "recommended_recipes": recommended_recipes_list,
            }
            return suggested_recipes
        elif model == "bert":
            data = load_dataframe("data_with_topics.csv")
            bert_model = load_model("bert_model")
            user_topics, user_probs = bert_model.transform(user_input_processed)
            predicted_topic = user_topics[0]
            filtered_recipes = data[data["predicted_topic"] == predicted_topic]
            filtered_recipes["probability"] = filtered_recipes[
                "topic_probabilities"
            ]  # No need to index

            # Sort recipes by probability
            sorted_recipes = filtered_recipes.sort_values(
                by="probability", ascending=False
            )

            # top 5 recipes
            top_recipes = sorted_recipes.head(5)
            recommended_recipes_list = top_recipes[["title", "directions"]].to_dict(
                orient="records"
            )
            suggested_recipes = {
                "message": f"Recipes suggested based on ingredients: {ingredients}",
                "recommended_recipes": recommended_recipes_list,
            }
            return suggested_recipes
