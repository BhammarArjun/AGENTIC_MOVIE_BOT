import asyncio
from google.genai import types
from movie_db import process_user_query

async def main():
    """
    Main function to run the movie database interaction system.
    """
    # Initialize conversation history
    conversation_history = []
    
    print("\n" + "=" * 50)
    print("🎬 MOVIE MANIA CHATBOT 🎬")
    print("Ask questions about movies, actors, and more!")
    print("Type 'exit' to quit")
    print("=" * 50)
    
    while True:
        # Get user input
        user_query = input("\n💬 Enter your question: ")
        
        # Check if user wants to exit
        if user_query.lower() == 'exit':
            print("\nThank you for using Movie Mania Chatbot! Goodbye! 👋")
            break
        
        # Add user query to conversation history
        user_message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_query)],
        )
        conversation_history.append(user_message)
        
        # Process the query
        final_answer = await process_user_query(user_query, conversation_history)
        
        # Display the final answer
        print("\n" + "-" * 50)
        print("FINAL ANSWER:")
        print(final_answer)
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())