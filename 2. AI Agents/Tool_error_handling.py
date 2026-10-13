from langchain_core.tools import tool

@tool
def robust_database_query(query_str: str) -> str:
    """Executes a SQL query string against the production database."""
    try:
        # Intentionally forcing a failure for invalid syntax demo
        if "SELECT" not in query_str.upper():
            raise ValueError("Syntax Error: Query must start with SELECT clause.")
        
        # Simulated successful return
        return "Query Executed Successfully: 2 rows returned."
        
    except Exception as e:
        # Crucial pattern: Return the string error message directly to the model
        # so the model can inspect it and retry with fixed arguments.
        return f"Tool Execution Error: {str(e)}. Please correct your tool arguments and retry."