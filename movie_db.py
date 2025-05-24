import asyncio
from typing import Dict, List, Any, Optional
from google.genai import types
from entity_extraction import extract_movie_info
from fuzzy_matching import fuzzy_match_entities
from sql_generation import get_sql_from_gemini
from db_connector import connect_to_db, execute_query
from answer_validation import validate_movie_query_response
from rag_search import search_rag_movies
from config import GEMINI_API_KEY
from google import genai

# Initialize Google Generative AI client
client = genai.Client(api_key=GEMINI_API_KEY)

async def query_movies_db(question: str, 
                         extracted_movies: Optional[List[str]] = None, 
                         extracted_actors: Optional[List[str]] = None, 
                         task: Optional[str] = None, 
                         conversation_history: List = None) -> Dict[str, Any]:
    """
    Process natural language query and return database results.
    
    Args:
        question: The user's query
        extracted_movies: List of movies extracted from the query
        extracted_actors: List of actors extracted from the query
        task: The task extracted from the query
        conversation_history: Conversation history
        
    Returns:
        Dict containing SQL response and data
    """
    # Get SQL query from Gemini
    sql_object = await get_sql_from_gemini(question, extracted_movies, extracted_actors, task, conversation_history)
    
    # Print SQL reasoning
    if "reason" in sql_object:
        print(f"📝 SQL reasoning: {sql_object['reason']}")
    
    # Create DB connection pool
    pool = await connect_to_db()
    if not pool:
        return {
            "sql_tool_response": sql_object,
            "sql_data": "Failed to connect to database",
            "note": "Database connection failed"
        }
    
    try:
        # Execute the query
        data = await execute_query(pool, sql_object)
        
        # Add the SQL query to the result for reference
        result_dict = {
            "sql_tool_response": sql_object,
            "sql_data": data,
            "note": "each sql data correspond to query in sql_tool_response -> sql_queries"
        }
        # Close the pool
        await pool.close()
        
        return result_dict
    
    except Exception as e:
        print(f"❌ Error during database query: {str(e)}")
        await pool.close()
        return {
            "sql_tool_response": sql_object,
            "sql_data": "Failed to execute query",
            "note": f"Error: {str(e)}"
        }

async def process_user_query(user_query: str, conversation_history: List) -> str:
    """
    Process a user query and generate a response.
    
    Args:
        user_query: The user's natural language query
        conversation_history: List of previous conversation messages
        
    Returns:
        Final answer to the user's query
    """
    print("\n" + "=" * 50)
    print(f"📝 New query: {user_query}")
    print("=" * 50)
    
    # Step 1: Extract movie information from the query
    extracted_info = extract_movie_info(user_query)
    
    # Step 2: Perform fuzzy matching on extracted entities
    corrected_actors, corrected_movies = fuzzy_match_entities(
        extracted_info.Actors, extracted_info.Title
    )
    
    # Step 3: Query the database
    db_result = await query_movies_db(
        user_query, 
        corrected_movies, 
        corrected_actors, 
        extracted_info.Task,
        conversation_history
    )
    
    # Add the database result to the conversation history
    model_message = types.Content(
        role="model",
        parts=[types.Part.from_text(text=str(db_result))],
    )
    conversation_history.append(model_message)
    
    # Step 4: Validate if the SQL results answer the query or if RAG is needed
    validation_result = validate_movie_query_response(conversation_history)
    
    # Step 5: Generate the final answer
    final_answer = None
    
    if validation_result.further_search:
        if not validation_result.rag_prompt:
            print("❓ No RAG prompts available, using direct answer")
            final_answer = validation_result.direct_answer
            conversation_history.append(types.Content(
                role="model",
                parts=[types.Part.from_text(text=str(final_answer))],
            ))
        else:
            # Perform RAG search
            print(f"🔍 Performing RAG search with prompts: {validation_result.rag_prompt}")
            validation_json = validation_result.model_dump()
            documents_rag = {}
            
            # Search for each RAG prompt
            for query in validation_result.rag_prompt:
                rag_results = search_rag_movies(query, validation_result.rag_filter)
                documents_rag[query] = rag_results
            
            # Add RAG results to the validation data
            validation_json.update({"rag_documents": documents_rag})
            
            # Add RAG results to conversation history
            conversation_history.append(types.Content(
                role="model",
                parts=[types.Part.from_text(text=str(validation_json))],
            ))
            
            # Generate final answer using RAG results
            print("🧠 Generating final answer using RAG results...")
            final_response = client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                config=types.GenerateContentConfig(
                    system_instruction="Based on the provided RAG documents, answer the user's recent question. Try to be flexible and brainstorm what user is asking and give satisfactory answer. If the answer cannot be found in the RAG documents, answer \"I'm sorry, I don't know the answer to that question.\"",
                    temperature=0.1,
                ),
                contents=conversation_history
            )
            
            final_answer = final_response.text
            
            # Add final answer to conversation history
            conversation_history.append(types.Content(
                role="model",
                parts=[types.Part.from_text(text=str(final_answer))],
            ))
    else:
        # Use the direct answer from SQL validation
        print("✅ Using direct answer from SQL results")
        final_answer = validation_result.direct_answer
        conversation_history.append(types.Content(
            role="model",
            parts=[types.Part.from_text(text=str(final_answer))],
        ))
    
    return final_answer