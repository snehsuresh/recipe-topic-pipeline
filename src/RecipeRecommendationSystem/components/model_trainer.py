import mlflow
import mlflow.sklearn
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import joblib
from src.RecipeRecommendationSystem.utils.utils import save_model, load_model
from bertopic import BERTopic


class Trainer:
    def generate_vectors(self, data, user):
        print("Generating Vectors")
        vectorizer_path = "vectorizer"
        # Create a document-term matrix for ingredients and directions combined

        if user == 0:
            vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
            dtm = vectorizer.fit_transform(data["cleaned_text"])
            save_model(vectorizer, vectorizer_path)
        else:
            vectorizer = load_model(vectorizer_path)
            dtm = vectorizer.transform([data])
        return dtm

    def fit_models(self, dtm, training_mode, data=None):
        print("Fitting Model")
        if training_mode == 0:
            lda = LatentDirichletAllocation(n_components=100, random_state=42)
            lda.fit(dtm)
            return lda
        elif training_mode == 1:
            nmf = NMF(n_components=100, random_state=42)
            nmf.fit(dtm)
            return nmf
        elif training_mode == 3:
            topic_model = BERTopic()
            texts = data["cleaned_text"].tolist()
            topics, probs = topic_model.fit_transform(texts)
            data["predicted_topic"] = topics
            data["topic_probabilities"] = probs
            return data, topic_model
            # return bert
        elif training_mode == 2:
            lda = LatentDirichletAllocation(
                n_components=100, random_state=42
            )  # Adjust number of topics as needed
            lda.fit(dtm)
            nmf = NMF(n_components=100, random_state=42)
            nmf.fit(dtm)
            return lda, nmf
        else:
            print("Training mode error")
