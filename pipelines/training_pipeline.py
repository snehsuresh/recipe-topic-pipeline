from src.RecipeRecommendationSystem.components.model_trainer import Trainer
from src.RecipeRecommendationSystem.utils.utils import save_model
from src.RecipeRecommendationSystem.utils.utils import load_dataframe


def start_training():
    print("Starting Training")
    trainer = Trainer()
    df = load_dataframe("cleaned_dataframe.csv")
    document_term_matrix = trainer.generate_vectors(df, 0)
    print("DTM Generated")
    lda, nmf = trainer.fit_models(document_term_matrix)
    print("Training Completed")
    # return lda, nmf

    # Save the document-term matrix
    save_model(document_term_matrix, "document_term_matrix")
    save_model(lda, "lda_model")
    save_model(nmf, "nmf_model")
