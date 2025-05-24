from typing import List, Tuple
from rapidfuzz import process, fuzz
from config import MOVIES_LIST, ACTORS_LIST

def fuzzy_match_entities(user_actors: List[str] = [], 
                         user_movies: List[str] = [], 
                         threshold: int = 70) -> Tuple[List[str], List[str]]:
    """
    Perform fuzzy matching against known actor and movie lists to correct
    misspelled or abbreviated names.
    
    Args:
        user_actors: List of actor names provided by the user
        user_movies: List of movie names provided by the user
        threshold: Minimum score (0-100) to consider a match valid
        
    Returns:
        tuple: (corrected_actors, corrected_movies) lists
    """
    print(f"🔄 Performing fuzzy matching on {len(user_movies)} movies and {len(user_actors)} actors")
    
    # Initialize output lists
    corrected_actors = []
    corrected_movies = []
    
    # Process actors using weighted ratio for better matching
    for user_actor in user_actors:
        # Skip empty strings
        if not user_actor.strip():
            corrected_actors.append("")
            continue
            
        # Find the best match in actor_list
        match_result = process.extractOne(
            user_actor, 
            ACTORS_LIST, 
            scorer=fuzz.WRatio
        )

        if match_result and match_result[1] >= threshold:
            print(f"  Actor match: '{user_actor}' → '{match_result[0]}' (score: {match_result[1]})")
            corrected_actors.append(match_result[0])
        else:
            print(f"  No good match found for actor: '{user_actor}'")
            corrected_actors.append(user_actor)
    
    # Process movies using weighted ratio for better matching
    for user_movie in user_movies:
        # Skip empty strings
        if not user_movie.strip():
            corrected_movies.append("")
            continue
            
        # Find the best match in movies_list
        match_result = process.extractOne(
            user_movie, 
            MOVIES_LIST, 
            scorer=fuzz.WRatio
        )

        if match_result and match_result[1] >= threshold:
            print(f"  Movie match: '{user_movie}' → '{match_result[0]}' (score: {match_result[1]})")
            corrected_movies.append(match_result[0])
        else:
            print(f"  No good match found for movie: '{user_movie}'")
            corrected_movies.append(user_movie)
    
    return corrected_actors, corrected_movies