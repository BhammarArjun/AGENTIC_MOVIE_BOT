from google import genai
from google.genai import types
from config import GEMINI_API_KEY
from models import ValidateAnswer

# Initialize Google Generative AI client
client = genai.Client(api_key=GEMINI_API_KEY)

# System instruction for the validator
SYSTEM_INSTRUCTION_VALIDATOR = """Based on the user and model interaction, determine if the question can be answered directly from SQL results or if RAG-based search is required.

INPUT FORMAT:
You will receive a result_dict object with these components:
- sql_tool_response: Contains Json object with fields:
  - sql_queries: List of SQL queries executed
  - reason: Explanation from the SQL tool
  - is_completed: Whether SQL alone answered the question
- sql_data: Array of data results, each corresponding to a query in sql_queries
- note: Additional context about the SQL data

RAG USAGE TYPES:
1. Similarity Search: When user asks for movies similar to a specific title
   - Include movie titles in rag_prompt (e.g., ["Inception"])
   
2. General Plot Search: When user asks for movies with specific plot elements
   - Include optimized search phrases in rag_prompt (e.g., ["student returns from USA to India"])
   - DO NOT include words like "movie", "film", or "show" in these search phrases

DECISION PROCESS:

1. If sql_data contains information that can answer the question (fully or partially):
   - Use the model's intelligence to analyze the data (whether factual or plot content)
   - Format a clear direct_answer based on this analysis
   - Set sql_query to the primary query used
   - Set further_search = False
   - Set rag_prompt and rag_filter to empty/None
   - Explain in reason what data was found and how it answers the query

2. If user is requesting movie recommendations or similarity-based results:
   - Set direct_answer to None (or partial answer if available)
   - Set further_search = TRUE
   - Populate rag_prompt with appropriate movie titles for similarity search
   - Set appropriate rag_filter if needed
   - Explain in reason why similarity search is needed

3. If user is asking about specific plot elements without naming a movie:
   - If SQL provided some results but they're insufficient:
     - Include any partial answer in direct_answer
     - Set further_search = TRUE
     - Create optimized search phrases in rag_prompt (focus on plot elements, not "movie" terms)
     - Set appropriate rag_filter if needed
     - Explain in reason why general plot search is needed
   - If SQL provided no results:
     - Set direct_answer to a message indicating the need for plot search
     - Set further_search = TRUE
     - Create optimized search phrases in rag_prompt
     - Explain in reason why general plot search is needed

4. If sql_data is empty or NULL and it's not a RAG search case:
   - Set direct_answer to a polite denial message (e.g., "I couldn't find any information about that in our database.")
   - Set further_search = False
   - Set sql_query to the primary query used
   - Explain in reason that no data was found for the query

EXAMPLES:

1. Direct Answer from Factual Data:
   - Query: "What are the highest-rated action movies from 2020?"
   - SQL returns: List of movies with ratings
   - Response: {direct_answer: [formatted list of movies], further_search: false}

2. Direct Answer from Plot Analysis:
   - Query: "In which Shah Rukh Khan movie does he play a character named Raj?"
   - SQL returns: Plots from all SRK movies
   - Response: {direct_answer: "Shah Rukh Khan plays a character named Raj in DDLJ and Kuch Kuch Hota Hai.", further_search: false, reason: "Found character information by analyzing the plots of SRK movies."}

3. No Data Available:
   - Query: "What happens in Avatar 5?"
   - SQL returns: Empty result
   - Response: {direct_answer: "I couldn't find any information about Avatar 5 in our database.", further_search: false, reason: "No data was found for this movie."}

4. Similarity RAG Search:
   - Query: "Recommend movies like Inception"
   - SQL returns: Plot of Inception
   - Response: {direct_answer: null, further_search: true, rag_prompt: ["Inception"], reason: "Need similarity search to find movies like Inception"}

5. General Plot Search:
   - Query: "Which movie shows a hero going to USA to study and then returning to India?"
   - SQL returns: Limited or no results from LIKE query
   - Response: {direct_answer: null, further_search: true, rag_prompt: ["hero studies in USA returns to India"], reason: "Need to search movie plots for this specific narrative element"}

6. General Plot Search with Genre Filter:
   - Query: "Which comedy movie has a scene with twins separated at birth?"
   - SQL returns: Limited results
   - Response: {direct_answer: null, further_search: true, rag_prompt: ["twins separated at birth reunite"], rag_filter: {"Genre": ["comedy"]}, reason: "Need to search comedy movie plots for separated twins storyline"}

IMPORTANT NOTES:
1. RAG is needed for TWO cases:
   - Similarity/recommendation queries (use movie titles in rag_prompt)
   - General plot element searches (use optimized search phrases in rag_prompt)
2. When creating optimized search phrases for plot elements:
   - Focus on the narrative elements, characters, or plot points
   - DO NOT include terms like "movie," "film," or "show"
   - Keep phrases concise and descriptive
3. Always provide a clear denial in direct_answer when no data is found and RAG isn't applicable
4. Use rag_filter appropriately when the user specifies genres, years, actors, or ratings
"""

def validate_movie_query_response(conversation_history):
    """
    Validate if SQL results properly answer the user's query or if RAG search is needed.
    
    Args:
        conversation_history: Conversation history between user and model
        
    Returns:
        ValidateAnswer object with validation results
    """
    print("🔍 Validating if SQL results are sufficient or if RAG search is needed")
    
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION_VALIDATOR,
        response_mime_type="application/json", 
        response_schema=ValidateAnswer,
        temperature=0.1
    )
    
    # Make the API call to Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        config=config,
        contents=conversation_history
    )
    
    # The response.parsed will automatically convert to the Pydantic model
    validation_result = response.parsed
    
    if validation_result.further_search:
        print(f"🔄 Need RAG search: {validation_result.reason}")
    else:
        print(f"✅ SQL results are sufficient")
        if validation_result.direct_answer:
            print(f"📝 Direct answer: {validation_result.direct_answer[:100]}...")
        
    return validation_result