# import pandas as pd
# from sqlalchemy import create_engine
# from pathlib import Path



# #  Load job offer data from PostgreSQL database.
# class JobDataLoader:
    
#     def __init__(self, database_uri: str = 'postgresql://postgres:0000@127.0.0.1:5432'):
#         self.database_uri = database_uri
#         self.engine = create_engine(database_uri)
    
#     def load_data(self, query, limit) -> pd.DataFrame:

#         if query is None:
#             query = f"""SELECT mission_clean, profil_clean, title_clean FROM test_schema.portaljob_test ORDER BY title_clean DESC LIMIT {limit}"""  

#         try:
#             df = pd.read_sql(query, self.engine)
#             print(f"!!We've got {len(df)} lines retrieved from the database !!")
#             return df
#         except Exception as e:
#             raise Exception(f">>> Error loading data from database: {e}")
        

# # Convenience function to load job data.
# def load_job_data(limit: int = 1000, database_uri: str = 'postgresql://postgres:0000@127.0.0.1:5432') -> pd.DataFrame:
#     loader = JobDataLoader(database_uri)
#     return loader.load_data(limit=limit)

"""
Data Loading Module
===================
Handles database connections and data retrieval.
"""

"""
Data Loading Module
===================
Handles database connections and data retrieval.
"""

import pandas as pd
from sqlalchemy import create_engine
from typing import Optional


class JobDataLoader:
    """Load job offer data from PostgreSQL database."""
    
    def __init__(self, database_uri: str = 'postgresql://postgres:0000@127.0.0.1:5432'):
        """
        Initialize database connection.
        
        Args:
            database_uri: PostgreSQL connection string
        """
        self.database_uri = database_uri
        self.engine = create_engine(database_uri)
    
    def load_data(self, query: Optional[str] = None, limit: int = 1000) -> pd.DataFrame:
        """
        Load job data from database.
        
        Args:
            query: Custom SQL query (optional)
            limit: Maximum number of rows to retrieve
            
        Returns:
            DataFrame with job offer data
        """
        if query is None:
            query = f"""
            SELECT mission_clean, profil_clean, title_clean FROM test_schema.portaljob_test ORDER BY title_clean DESC LIMIT 1000
            """
        
        try:
            df = pd.read_sql(query, self.engine)
            print(f"✅ Successfully loaded {len(df)} job offers from database")
            return df
        except Exception as e:
            raise Exception(f"❌ Error loading data from database: {e}")
    
    def get_sample_data(self, n: int = 100) -> pd.DataFrame:
        """Get a sample of job data for quick testing."""
        return self.load_data(limit=n)


def load_job_data(limit: int = 1000, database_uri: str = 'postgresql://postgres:0000@127.0.0.1:5432') -> pd.DataFrame:
    """
    Convenience function to load job data.
    
    Args:
        limit: Maximum number of rows
        database_uri: Database connection string
        
    Returns:
        DataFrame with job offers
    """
    loader = JobDataLoader(database_uri)
    return loader.load_data(limit=limit)


