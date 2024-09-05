from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import csv
import requests

import random


def write_recipes_to_csv(recipes, page_number):
    filename = f"recipes_page_{page_number}.csv"
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "title",
                "url",
                "image_url",
                "author",
                "rating_percent",
                "cook_time",
            ],
        )
        writer.writeheader()
        writer.writerows(recipes)
    print(f"Data written to {filename}")


def fetch_recipe_data(page_number, timeout=60):
    url = f"https://api.food.com/services/mobile/fdc/search/sectionfront?pn={page_number}&recordType=Recipe&sortBy=trending&collectionId=17"

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        if "API rate limit" in str(e):  # Check if it's an API rate limit error
            raise
        return {}


def scrape_recipes(category_urls, target_unique_titles=82212):
    unique_titles = set()  # Use a set to track unique titles
    recipes = []
    last_page_number = 17891

    for category_url in category_urls:
        page_number = 17891
        while len(unique_titles) < target_unique_titles:
            print(f"Fetching URL: {category_url}?page={page_number}")
            try:
                data = fetch_recipe_data(page_number)
            except requests.exceptions.ReadTimeout as e:
                print(f"Read timeout error: {e}")
                # Save state and return if read timeout error
                return recipes, last_page_number
            except Exception as e:
                print(f"Error fetching data: {e}")
                if "API rate limit" in str(e):
                    # Save state and return if rate limit error
                    print("API rate limit reached. Saving progress and exiting.")
                    return recipes, last_page_number

                # Random sleep to avoid hitting API rate limits
                time.sleep(random.uniform(1, 5))
                continue

            if not data.get("response") or not data.get("response").get("results"):
                print("No results found or no response, ending.")
                break

            for recipe in data.get("response").get("results", []):
                title = recipe.get("main_title", "").strip()
                if title in unique_titles:
                    continue  # Skip duplicate titles

                recipe_url = recipe.get("record_url")
                image_url = recipe.get("recipe_photo_url")
                author = recipe.get("main_username")
                rating_percent = recipe.get("main_rating")
                cook_time = recipe.get("recipe_totaltime")

                recipes.append(
                    {
                        "title": title,
                        "url": recipe_url,
                        "image_url": image_url,
                        "author": author,
                        "rating_percent": rating_percent,
                        "cook_time": cook_time,
                    }
                )

                unique_titles.add(title)  # Add title to the set

            print(f"Current count of unique recipes: {len(unique_titles)}")

            if len(unique_titles) >= target_unique_titles:
                break  # Stop if the target number of unique titles is reached

            # Write data to CSV every 100 pages
            if page_number % 100 == 0:
                write_recipes_to_csv(recipes, page_number)
                recipes = []  # Clear recipes list after saving to CSV

            print("Moving to next page")
            page_number += 1  # Move to the next page
            last_page_number = page_number

    # Write remaining data to CSV if needed
    if recipes:
        write_recipes_to_csv(recipes, last_page_number)

    return recipes, last_page_number


import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Counter for errors
error_count = 0


def extract_recipe_details(recipe_url, cook_time, recipe_rating):
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # headless mode

    service = Service("/opt/homebrew/bin/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get(recipe_url)

        # Wait for directions and ingredients to be present
        wait = WebDriverWait(driver, 3)
        directions_list = wait.until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".direction-list li"))
        )
        ingredient_elements = wait.until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, ".ingredient-list li")
            )
        )

        # Extract directions
        directions = "\n".join([li.text.strip() for li in directions_list])

        # Extract ingredients
        ingredients_dict = {}
        for element in ingredient_elements:
            quantity = element.find_element(
                By.CSS_SELECTOR, ".ingredient-quantity"
            ).text.strip()
            text = element.find_element(
                By.CSS_SELECTOR, ".ingredient-text"
            ).text.strip()
            ingredients_dict[quantity] = text

        return {
            "directions": directions,
            "ingredients": ingredients_dict,
            "cook_time": cook_time,
            "recipe_rating": recipe_rating,
        }

    except Exception as e:
        global error_count
        error_count += 1
        return {
            "directions": "N/A",
            "ingredients": {},
            "cook_time": cook_time,
            "recipe_rating": recipe_rating,
        }

    finally:
        driver.quit()


def process_csv_file(input_file, output_file):
    global error_count
    print(f"Processing file: {input_file}")

    # Read the CSV file
    recipes = pd.read_csv(input_file)

    # Prepare to collect results
    batch_size = 10
    all_recipes = []

    # Open the output file in append mode
    with open(output_file, "a") as f:
        # Write the header if the file is empty
        if os.stat(output_file).st_size == 0:
            pd.DataFrame(
                columns=[
                    "title",
                    "url",
                    "image_url",
                    "author",
                    "rating_percent",
                    "cook_time",
                    "directions",
                    "ingredients",
                ]
            ).to_csv(f, index=False)

    # Process recipes in batches
    for i, (_, recipe) in enumerate(recipes.iterrows()):
        try:
            details = extract_recipe_details(
                recipe["url"], recipe["cook_time"], recipe["rating_percent"]
            )
            all_recipes.append(
                {
                    "title": recipe["title"],
                    "url": recipe["url"],
                    "image_url": recipe["image_url"],
                    "author": recipe["author"],
                    "rating_percent": recipe["rating_percent"],
                    "cook_time": recipe["cook_time"],
                    "directions": details["directions"],
                    "ingredients": details["ingredients"],
                }
            )

            # Write to CSV every 10 recipes
            if (i + 1) % batch_size == 0:
                df = pd.DataFrame(all_recipes)
                df.to_csv(output_file, mode="a", header=False, index=False)
                all_recipes = []  # Reset list for the next batch
                print(f"Processed and saved batch of {batch_size} recipes.")

        except Exception:
            # Log general error message
            print(f"Failed to extract details for {recipe['url']}")

    # Process any remaining recipes that didn't fill a complete batch
    if all_recipes:
        df = pd.DataFrame(all_recipes)
        df.to_csv(output_file, mode="a", header=False, index=False)
        print(f"Processed and saved final batch of {len(all_recipes)} recipes.")

    # Print the total error count
    print(f"Total errors encountered: {error_count}")

    # Delete the input file once done
    os.remove(input_file)
