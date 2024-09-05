from flask import Flask, render_template, request, jsonify
from pipelines.processing_pipeline import start_processing
from pipelines.training_pipeline import start_training
from src.RecipeRecommendationSystem.components.data_processing import DataProcessing
from src.RecipeRecommendationSystem.components.model_trainer import Trainer
from src.RecipeRecommendationSystem.utils.utils import load_model, load_dataframe
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)


# Home route
@app.route("/")
def home():
    return render_template("index.html")


# API route for suggesting recipes
@app.route("/initialize_model", methods=["POST"])
def initialize_model():
    print("Initializing Model")
    start_processing()
    start_training()
    response = {"message": "Model initialized successfully!"}
    return jsonify(response)


# API route for suggesting recipes
@app.route("/suggest-recipe", methods=["POST"])
def suggest_recipe():
    processor = DataProcessing()
    trainer = Trainer()
    ingredients = request.form.get("ingredients", "")
    user_input_processed = processor.preprocess_text(ingredients)

    user_input_dtm = trainer.generate_vectors(user_input_processed, 1)

    lda_model = load_model("lda_model")
    nmf_model = load_model("nmf_model")
    dtm = load_model("document_term_matrix")

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
        user_input_lda_features, document_lda_topic_features
    )
    # Print the similarity scores for debugging
    print("Similarity Scores:", similarities)

    df = load_dataframe("cleaned_dataframe.csv")
    top_indices = similarities[0].argsort()[-5:][::-1]  # Get top 5 recommended recipes
    recommended_recipes = df.iloc[top_indices]

    recommended_recipes_list = recommended_recipes[["title", "directions"]].to_dict(
        orient="records"
    )

    suggested_recipes = {
        "message": f"Recipes suggested based on ingredients: {ingredients}",
        "recommended_recipes": recommended_recipes_list,
    }
    return jsonify(suggested_recipes)


if __name__ == "__main__":
    app.run(debug=True)
