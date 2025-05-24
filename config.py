import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "password",  # Replace with your actual password
    "database": "movie_mania" # Replace with your actual db name
}

# Load movie and actor lists
def load_entity_lists():
    movies_list = []
    actors_list = []
    
    if os.path.exists('backend/data/movies_list.json'):
        with open('backend/data/movies_list.json', 'r', encoding='utf-8') as f:
            movies_list = json.load(f)
    else:
        print("Warning: movies_list.json not found. Using empty list.")

    if os.path.exists('backend/data/actors_list.json'):
        with open('backend/data/actors_list.json', 'r', encoding='utf-8') as f:
            actors_list = json.load(f)
    else:
        print("Warning: actors_list.json not found. Using empty list.")
        
    return movies_list, actors_list

MOVIES_LIST, ACTORS_LIST = load_entity_lists()
