from flask import Flask, render_template, request, jsonify
from pipelines.processing_pipeline import start_processing
from pipelines.training_pipeline import start_training
from src.RecipeRecommendationSystem.components.recommendation import Recommendation

app = Flask(__name__)
app.secret_key = "snehpillai"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/train_model", methods=["POST"])
def initialize_lda():
    model_type = request.json.get("model_type")
    mode = None

    if model_type == "lda":
        mode = 0
        print("Initializing LDA Model")
        response = {"message": "LDA Model initialized successfully!"}
    elif model_type == "nmf":
        mode = 1
        print("Initializing NMF Model")
        response = {"message": "NMF Model initialized successfully!"}
    elif model_type == "ensemble":
        mode = 2
        print("Initializing Ensemble Model")
        response = {"message": "LDA + NMF initialized successfully!"}
    elif model_type == "bert":
        mode = 3
        print("Initializing BERT Model")
        response = {"message": "BERT Model initialized successfully!"}
    else:
        response = {"message": "Invalid model type specified."}

    # start_processing()
    start_training(mode)
    return jsonify(response)


# API route for suggesting recipes
@app.route("/suggest-recipe", methods=["POST"])
def suggest_recipe():
    # Get the model type from the form data
    selected_model = request.form.get("model")
    # lda, nmf, bert, ensemble
    recommender = Recommendation()

    ingredients = request.form.get("ingredients", "")
    suggested_recipes = recommender.recommend(ingredients, selected_model)

    return jsonify(suggested_recipes)


if __name__ == "__main__":
    app.run(debug=True)
