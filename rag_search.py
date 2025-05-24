import requests
import warnings
import numpy as np
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from config import MOVIES_LIST

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize Qdrant client
qdrant_client = QdrantClient("http://localhost:6333/dashboard")
COLLECTION_NAME = "rag_movies"

def get_embedding(text: str) -> np.ndarray:
    """
    Get embeddings for a text string using local MX Bai server.
    
    Args:
        text: Text to embed
        
    Returns:
        Normalized embedding vector
    """
    print(f"🧠 Generating embedding for: '{text[:50]}...'")
    
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "mxbai-embed-large:latest", "prompt": text}
    )
    vector = np.array(response.json()['embedding'])
    vector = vector / np.linalg.norm(vector)
    return vector

def get_embedding_by_title(title: str) -> Optional[List[float]]:
    """
    Get embedding vector for a movie by its title.
    
    Args:
        title: Movie title to find embedding for
        
    Returns:
        Embedding vector if found, None otherwise
    """
    print(f"🔍 Looking up embedding for movie: '{title}'")
    
    query_filter = {
        "must": [
            {
                "key": "Title",
                "match": {
                    "value": title
                }
            }
        ]
    }

    # Perform search with a dummy query_vector (zero vector)
    dummy_vector = [0.0] * 1024

    response = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=dummy_vector,
        query_filter=query_filter,
        limit=1,
        with_payload=True,
        with_vectors=True
    )

    if response:
        print(f"✅ Found embedding for movie: '{title}'")
        return response[0].vector
    else:
        print(f"❌ No embedding found for movie: '{title}'")
        return None

def search_rag_movies(query: str, filter=None) -> List[Dict[str, Any]]:
    """
    Search for movies in the RAG database.
    
    Args:
        query: Search query (movie title or description)
        filter: Optional filter to narrow search results
        
    Returns:
        List of movie data matching the query
    """
    print(f"🔍 Performing RAG search for: '{query}'")
    
    # Convert query to lowercase
    query = query.lower()
    is_movie = False
    query_vector = None
    movie_plot = []

    # Check if query is a known movie title
    if query in set(MOVIES_LIST):
        print(f"✅ Query matches known movie title: '{query}'")
        query_vector = get_embedding_by_title(query)
        is_movie = True
        
        # Get the movie's plot summary
        if query_vector:
            movie_plot_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=1,
                with_payload=True
            )
            
            if movie_plot_results:
                movie_plot = [x.payload for x in movie_plot_results]
                print(f"✅ Retrieved plot for movie: '{query}'")
    
    # If not a known movie or couldn't get embedding, generate from query text
    if not query_vector:
        print("🧠 Generating embedding from query text")
        query_vector = get_embedding(query)
    
    # Initialize query_filter as None
    query_filter = None
    
    # If filter is provided, create a dynamic Qdrant Filter object
    if filter:
        print(f"🔍 Applying filters to RAG search")
        must_conditions = []
        
        # Process all filters
        for filter_type in ['Title', 'Genre', 'Year', 'Actors', 'ImdbRating']:
            filter_values = getattr(filter, filter_type, None)
            if filter_values:
                print(f"  - {filter_type} filter: {', '.join(filter_values)}")
                must_conditions.append(
                    FieldCondition(
                        key=filter_type,
                        match=MatchAny(any=[value.lower() for value in filter_values])
                    )
                )
        
        # Only create a query_filter if we have conditions
        if must_conditions:
            query_filter = Filter(must=must_conditions)
    
    # Execute the search with the constructed filter
    print("🔍 Executing vector search")
    response = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=10,
        with_payload=True
    )
    
    results = [x.payload for x in response] if response else []
    print(f"✅ RAG search found {len(results)} results")
    
    # If the query was a movie title, prepend its plot to the results
    if is_movie and movie_plot:
        final_results = movie_plot + results
        print(f"✅ Final results: {len(final_results)} items (including movie plot)")
        return final_results
    
    return results