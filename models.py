from pydantic import BaseModel, Field
from typing import List, Optional

class MovieInfo(BaseModel):
    Title: List[str] = Field(
        default_factory=list,
        description="Movie titles explicitly mentioned in the user query. Only include actual movie names, not descriptions or other text."
    )
    Genre: List[str] = Field(
        default_factory=list,
        description="Movie genres mentioned in the user query (e.g., action, comedy, thriller, drama, horror)."
    )
    Year: List[str] = Field(
        default_factory=list,
        description="Release years of movies mentioned in the user query. Extract only specific years (e.g., 2022)."
    )
    Actors: List[str] = Field(
        default_factory=list,
        description="Actor names mentioned in the user query. Use your brain to include correct names. Include full names when available."
    )
    ImdbRating: List[str] = Field(
        default_factory=list,
        description="IMDb ratings mentioned in the user query. Include only numeric values or ranges (e.g., 7.5, 'above 8')."
    )
    Task: str = Field(
        default="",
        description="A brief summary of what the user is trying to accomplish with their query (e.g., finding movies, counting results, comparing ratings)."
    )

class SQLResponse(BaseModel):
    sql_queries: List[str] = Field(
        default_factory=list,
        description="List of SQL queries needed to retrieve relevant movie data. Include multiple queries only if necessary."
    )
    reason: str = Field(
        default="",
        description="Explanation of what the SQL queries retrieve and whether further processing or RAG is needed."
    )
    is_completed: bool = Field(
        default=False,
        description="True if SQL queries directly and completely answer the question. False if further processing or RAG is needed."
    )

class RAGFilter(BaseModel):
    Title: Optional[List[str]] = None
    Genre: Optional[List[str]] = None
    Year: Optional[List[str]] = None
    Actors: Optional[List[str]] = None
    ImdbRating: Optional[List[str]] = None

class ValidateAnswer(BaseModel):
    direct_answer: Optional[str] = Field(
        default=None,
        description="The direct answer to the user's query based on SQL results. The model should analyze and interpret the data to provide the best possible answer."
    )
    sql_query: Optional[str] = Field(
        default=None,
        description="The primary SQL query used to retrieve data for the user's query."
    )
    rag_prompt: List[str] = Field(
        default_factory=list,
        description="List of movie titles or optimized search queries. For similarity searches, include movie titles. For general plot searches, include concise descriptive phrases."
    )
    rag_filter: Optional[RAGFilter] = Field(
        default=None,
        description="Filters to constrain RAG search. Allowed keys: Title, Genre, Year, Actors, ImdbRating. All values should be arrays of strings."
    )
    reason: str = Field(
        default="",
        description="A concise explanation of how the model's response addressed the user's query or why RAG is needed."
    )
    further_search: bool = Field(
        default=False,
        description="Whether RAG-based search is required. True for similarity/recommendation tasks OR for general plot element searches."
    )