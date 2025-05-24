import asyncio
import asyncpg
import re
from config import DB_CONFIG

async def connect_to_db():
    """Create a database connection pool"""
    try:
        conn_pool = await asyncpg.create_pool(**DB_CONFIG)
        print("📊 Database connection established")
        return conn_pool
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        return None

def clean_sql_query(sql_text):
    """Clean up SQL query by removing markdown code blocks if present"""
    if "```" in sql_text:
        # Use regex to extract the SQL between ```sql and ```
        match = re.search(r"```(?:sql)?\n(.*?)\n```", sql_text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return sql_text.strip()

async def execute_query(pool, sql_object):
    """Execute SQL queries and return the results"""
    result_data = []
    queries_list = sql_object.get('sql_queries', [])

    for i, query in enumerate(queries_list):
        query = clean_sql_query(query)
        try:
            print(f"🔍 Executing SQL query #{i+1}: {query[:100]}...")
            async with pool.acquire() as connection:
                # Determine if the query is a SELECT query or something else
                is_select = query.strip().lower().startswith('select')
                
                if is_select:
                    # For SELECT queries, fetch all rows
                    rows = await connection.fetch(query)
                    
                    # Convert rows to list of dictionaries
                    result = [dict(row) for row in rows]
                    result_data.append({i: result[:100]})
                    print(f"✅ Query #{i+1} returned {len(result)} rows")
                else:
                    result_data.append({i: ["Nothing to show"]})
                    print(f"⚠️ Query #{i+1} is not a SELECT query")
                    
        except Exception as e:
            print(f"❌ Error executing query #{i+1}: {str(e)}")
            result_data.append({i: ["Nothing to show"]})
    
    return result_data