import pandas as pd
from sqlalchemy import create_engine

DATABASE_URI = 'postgresql://postgres:0000@127.0.0.1:5432'
# QUERY = "SELECT mission, profil_clean, title_clean FROM test_schema.portaljob_test ORDER BY title_clean DESC LIMIT 1"
QUERY = "SELECT mission_clean, profil_clean, title_clean FROM portaljob ORDER BY title_clean DESC LIMIT 1000"

engine = create_engine(DATABASE_URI)

try:
    df = pd.read_sql_query(QUERY, engine)
    print(f"✅ {len(df)} lignes récupérées de la base de données.")
    # print(f"{df.to_string(index= False)}")

except Exception as e:
    raise Exception(f"❌ Erreur lors de la requête SQL : {e}")

if __name__ == "__main__":
    # print(df['mission_clean'].to_string(index= False))
    print(df['profil_clean'].to_string(index= False))
    # print(df['title_clean'].to_string(index= False))
    # print(df.to_string(index= False))