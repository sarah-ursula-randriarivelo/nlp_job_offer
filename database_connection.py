import pandas as pd
from sqlalchemy import create_engine

# path to the localhost server
DATABASE_URI = 'postgresql://postgres:0000@127.0.0.1:5432'

# SQL query to retrieve the data
QUERY = "SELECT mission_clean, profil_clean, title_clean FROM portaljob ORDER BY title_clean DESC LIMIT 1000"

engine = create_engine(DATABASE_URI)

try:
    df = pd.read_sql_query(QUERY, engine)

    print(f"!!We've got {len(df)} lines retrieved from the database !!") 

except Exception as e:
    raise Exception(f" >>> Error occurs during the SQL query {e}")

if __name__ == "__main__":

    
    print(df['profil_clean'].to_string(index= False))