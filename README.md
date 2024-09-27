# Recipe Recommender System Pipeline

## Overview

This project is a comprehensive recipe recommender system designed to fetch, process, and analyze recipes from various sources. It leverages web scraping, data extraction, and natural language processing (NLP) techniques to build a robust recommendation engine. The system includes multiple components for scraping recipe data, extracting details, and analyzing the recipes to provide recommendations.

## Project Components

1. **Recipe Data Fetching**: Fetches recipe data from an API and stores it in CSV files.
2. **Recipe Details Extraction**: Uses Selenium to scrape detailed recipe information from web pages.
3. **Data Processing and Cleaning**: Preprocesses the recipe data, including handling missing values and combining text fields for NLP.
4. **Topic Modeling**: Uses Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorization (NMF) to identify topics from recipe descriptions.
5. **Recommendation System**: Provides recipe recommendations based on user input using a combination of LDA, NMF, and BERTopic models.

## Getting Started

### Prerequisites

Make sure you have Python installed. You will also need to install the required libraries. You can do this by running:

```bash
pip install -r requirements.txt
python app.py
```
### Running the Project
## Detailed Instructions

### 1. Fetch Recipe Data

The data collection script scrapes recipe data from specified URLs and saves the data into CSV files. Modify the category_urls list if you want to scrape recipes from different categories.

### 2. Extract Recipe Details

The script uses Selenium to extract detailed directions and ingredient lists for each recipe. Ensure that you have the ChromeDriver installed and the path specified in the script.

### 3. Process CSV Files

The script processes the CSV files to extract directions and ingredients. It cleans the text data and prepares it for topic modeling. Adjust the input and output directory paths according to your environment.

### 4. Analyze and Recommend Recipes

The analysis scripts use LDA and NMF to identify topics from recipe descriptions and provide recommendations based on user input. The final recommendations are displayed based on the highest similarity scores.

### 5. Setting Up Environment Variables

You will need to set up an .env file to save and load dataframes and models. Create a file named .env in the root directory of your project and add the following variables
1.AWS_ACCESS_KEY_ID=
2.AWS_SECRET_ACCESS_KEY=
3.AWS_REGION=''


## Troubleshooting

API Rate Limits: If you encounter API rate limit errors, you may need to implement rate-limiting or retry logic in your fetch scripts.
Missing Data: Ensure that your data sources are up and that the CSV files are correctly formatted.
Web Scraping Issues: Web scraping might fail if the website layout changes. Adjust the Selenium selectors if needed.