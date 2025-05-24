from google import genai
from config import GEMINI_API_KEY
from models import MovieInfo

# Initialize Google Generative AI client
client = genai.Client(api_key=GEMINI_API_KEY)

def extract_movie_info(user_query):
    """
    Extract structured movie information from a user query using Gemini.
    
    Args:
        user_query: The natural language query from the user
        
    Returns:
        MovieInfo object containing extracted entities and task
    """
    print(f"🔍 Extracting movie information from query: '{user_query}'")
    
    # Enhanced prompt with clear instructions
    prompt = f"""
Extract structured movie-related information from the following user query. Follow the guidelines strictly and use your knowledge and reasoning to infer details accurately.

User Query: "{user_query}"

Extraction Guidelines:

- **Title**: Identify and extract only the actual movie titles mentioned in the query.

- **Genre**: Extract any movie genres explicitly stated (e.g., action, drama, thriller, comedy, etc.).

- **Year**: Capture any specific years or time periods mentioned (e.g., 2022, 1990s). Only include numeric or decade-based references.

- **Actors**: Extract names of actors referenced in the query.
    - Use full names wherever possible.
    - Correct spelling errors and expand abbreviations or nicknames.
    - Interpret popular aliases (e.g., "King Khan" → "Shah Rukh Khan", "Akki" → "Akshay Kumar") using general knowledge.

- **IMDb Rating**: Extract any rating values or filters mentioned (e.g., "above 7.5", "at least 8", etc.).

- **Task**: Summarize the user's overall intent—such as filtering movies, finding recommendations, counting results, etc.

If any field is missing or not clearly stated in the query, return an empty list or value for that field.
"""
    
    # Generate response from Gemini with schema
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": MovieInfo,
        },
    )
    
    # Return the parsed Pydantic object
    extracted_info = response.parsed
    print(f"✅ Extraction complete. Found: {len(extracted_info.Title)} titles, {len(extracted_info.Actors)} actors")
    return extracted_info