from src.RecipeRecommendationSystem.components.model_trainer import Trainer
from src.RecipeRecommendationSystem.utils.utils import save_model
from src.RecipeRecommendationSystem.utils.utils import load_dataframe, save_dataframe


def start_training(training_mode):
    print("Starting Training")
    trainer = Trainer()
    df = load_dataframe("cleaned_dataframe.csv")
    document_term_matrix = trainer.generate_vectors(df, 0)
    print("DTM Generated")
    if training_mode == 0:
        lda = trainer.fit_models(document_term_matrix, training_mode)
        save_model(lda, "lda_model")
    elif training_mode == 1:
        nmf = trainer.fit_models(document_term_matrix, training_mode)
        save_model(nmf, "nmf_model")
    elif training_mode == 3:
        data_with_topics, bert = trainer.fit_models(
            document_term_matrix, training_mode, df
        )
        save_model(bert, "bert_model")
        save_dataframe(data_with_topics, "data_with_topics.csv")
    elif training_mode == 2:
        lda, nmf = trainer.fit_models(document_term_matrix, training_mode)
        save_model(lda, "lda_model")
        save_model(nmf, "nmf_model")
    print("Training Completed")

    save_model(document_term_matrix, "document_term_matrix")
