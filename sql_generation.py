from typing import List, Optional, Dict, Any
from google import genai
from google.genai import types
from config import GEMINI_API_KEY
from models import SQLResponse

# Initialize Google Generative AI client
client = genai.Client(api_key=GEMINI_API_KEY)

# System instruction for Gemini model
SYSTEM_INSTRUCTION_SQL = """You are a specialized SQL query generator for a movie database. Your task is to convert natural language questions into correct PostgreSQL queries.

DATABASE SCHEMA:
- movies (id, title, year, imdb_rating, plot)
- actors (id, name)
- genres (id, name) 
- languages (id, name)
- movie_actors (movie_id, actor_id)
- movie_genres (movie_id, genre_id)
- movie_languages (movie_id, language_id)

KEY RELATIONSHIPS:
- movies have many actors through movie_actors
- movies have many genres through movie_genres
- movies have many languages through movie_languages

DATA SPECIFICATIONS:
- ALL TEXT in database is stored in lowercase (titles, actor names, genres, plot, languages)
- 'year' and 'imdb_rating' are stored as TEXT
- Missing data is stored as "N/A" (imdb_rating, year) or "n/a" (other fields)
- For numeric comparisons, cast imdb_rating to NUMERIC using NULLIF(m.imdb_rating, 'N/A')::NUMERIC

ADVANCED PROCESSING GUIDELINES:
1. Previous Answers: If the answer is available from past conversation, include that in the reason field and set is_completed to True without generating new queries.

2. Direct Plot Questions: For questions about "what happens in [specific movie]", fetch the exact plot with SQL. This directly answers the question (is_completed = true).

3. Plot Analysis Questions: For questions requiring analysis of plot content (e.g., character identification, scene descriptions), fetch the relevant movie plots with SQL, but set is_completed = false as further processing of these plots is needed.

4. Multi-part Questions: If a query contains multiple questions, generate SQL queries for all parts that can be answered with SQL, and clearly indicate in the reason field which parts require RAG processing.

5. RAG-Specific Cases: RAG is ONLY needed for:
   - Similarity-based recommendations (e.g., "Find movies similar to Inception")
   - General content searches not specific to named movies (e.g., "Which movies contain a bank robbery scene?")
   - Comparison requests (e.g., "Which movies have themes like The Matrix?")
   For these cases, set is_completed = false and indicate RAG is needed in the reason field without generating SQL queries.

OUTPUT FORMAT:
You must respond with a SQLResponse object containing:
1. sql_queries: An array containing SQL queries to retrieve relevant data
2. reason: A brief explanation of what the queries retrieve and whether further processing or RAG is needed
3. is_completed: Set to True only if SQL directly and completely answers the question. Set to False if further processing or RAG is needed.

RESPONSE REQUIREMENTS:
1. For factual movie database questions (ratings, years, counts), provide SQL queries that directly answer the question
2. Always convert search terms to lowercase in your queries
3. For direct plot questions about specific movies, fetch the exact plot and set is_completed = true
4. For character or scene questions, fetch relevant plots but set is_completed = false (requires further processing)
5. For similarity/recommendation questions, indicate RAG is needed in the reason field
6. Use LIKE operators for plot searches when possible to narrow results
7. Only respond to movie-related queries (greetings and farewells are acceptable)
8. Never give insert, update, delete, drop, or truncate queries
9. If in past conversations, you can find direct answer to user query, include that in the reason field and set is_completed to True
"""

async def get_sql_from_gemini(question: str, 
                             extracted_movies: Optional[List[str]] = None, 
                             extracted_actors: Optional[List[str]] = None, 
                             task: Optional[str] = None, 
                             conversation_history: List = None) -> Dict[str, Any]:
    """
    Convert natural language question to SQL using Gemini API,
    with additional context from extracted entities and task.
    
    Args:
        question: The original natural language question
        extracted_movies: List of corrected movie names after fuzzy matching
        extracted_actors: List of corrected actor names after fuzzy matching
        task: Extracted user intent/task
        conversation_history: List of conversation messages
        
    Returns:
        SQL response object as a dictionary
    """
    print(f"🔍 Generating SQL for query: '{question}'")
    
    try:
        if conversation_history is None:
            conversation_history = []
            
        # Create a structured prompt that includes the extracted entities
        enhanced_prompt = question
        
        # Add context section if we have any extracted information
        if any([extracted_movies, extracted_actors, task]):
            enhanced_prompt += "\n\n### Additional Context ###\n"
            
            if extracted_movies and len(extracted_movies) > 0:
                movie_list = ", ".join([f"'{movie}'" for movie in extracted_movies if movie])
                enhanced_prompt += f"Extracted Movies: {movie_list}\n"
                
            if extracted_actors and len(extracted_actors) > 0:
                actor_list = ", ".join([f"'{actor}'" for actor in extracted_actors if actor])
                enhanced_prompt += f"Extracted Actors: {actor_list}\n"
                
            if task:
                enhanced_prompt += f"User Intent: {task}\n"
                
            enhanced_prompt += "\nNote: These are fuzzy-matched and corrected names. Please use these names in your SQL query where applicable, as they match the database records."
        
        user_message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=enhanced_prompt)],
        )
        conversation_history.append(user_message)

        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION_SQL,
                temperature=0.1,
                response_schema=SQLResponse,
                response_mime_type="application/json"),
            contents=conversation_history
        )
        
        # Extract SQL from response
        sql_object = response.parsed
        print(f"✅ SQL generated successfully")
        return sql_object.model_dump()
    
    except Exception as e:
        print(f"❌ Error generating SQL: {str(e)}")
        return {"sql_queries": [], "reason": f"Error: {str(e)}", "is_completed": False}