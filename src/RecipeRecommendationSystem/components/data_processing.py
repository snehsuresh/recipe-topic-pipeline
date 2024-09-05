# data_processing.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("wordnet")


class DataProcessing:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):
        # Clean and tokenize the text
        text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in text.split()
            if word not in self.stop_words
        ]
        return " ".join(tokens)

    def ingredients_to_string(self, ingredients_dict):
        # Convert ingredients dictionary to a string
        return " ".join([f"{value}" for value in ingredients_dict.values()])

    def process_data(self, df_copy):
        # Combine ingredients and directions into a single text field
        df_copy["ingredients_str"] = df_copy["ingredients"].apply(
            lambda x: self.ingredients_to_string(eval(x)) if pd.notna(x) else ""
        )

        df_copy["text"] = df_copy["ingredients_str"] + " " + df_copy["directions"]

        # Clean the combined text
        df_copy["cleaned_text"] = df_copy["text"].apply(self.preprocess_text)

        print(df_copy)
        return df_copy
