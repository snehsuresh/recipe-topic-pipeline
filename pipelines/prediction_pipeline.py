# # Preprocess user input
# user_input = "flour baking powder cocoa powder sugar eggs"
# user_input_processed = preprocess_text(user_input)
# user_input_dtm = vectorizer.transform([user_input_processed])

# # Transform user input using both LDA and NMF
# user_input_lda_features = lda.transform(user_input_dtm)
# user_input_nmf_features = nmf.transform(user_input_dtm)

# # Combine topic features from both models
# user_input_combined_features = np.hstack(
#     (user_input_lda_features, user_input_nmf_features)
# )

# # Combine topic features from both models
# combined_topic_features = np.hstack(
#     (user_input_lda_features, user_input_nmf_features)
# )

# similarities = cosine_similarity(
#     user_input_combined_features, combined_topic_features
# )

# top_indices = similarities[0].argsort()[-10:][
#     ::-1
# ]  # Get top 10 recommended recipes
# recommended_recipes = df_copy.iloc[top_indices]
# print(recommended_recipes[["title", "directions"]])
