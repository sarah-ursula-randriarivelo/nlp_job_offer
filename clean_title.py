import pandas as pd
import re
from sqlalchemy import create_engine



DATABASE_URI = 'postgresql://root:0000@192.168.1.120:5432/test'
QUERY = "SELECT mission_clean, profil_clean, title_clean FROM test_schema.portaljob_test ORDER BY title_clean DESC LIMIT 1"

engine = create_engine(DATABASE_URI)
try:
    df = pd.read_sql_query(QUERY, engine)
    print(f"✅ {len(df)} lignes récupérées de la base de données.")
except Exception as e:
    raise Exception(f"❌ Erreur lors de la requête SQL : {e}")

# Nettoyage de base
df.dropna(subset=['title_clean'], inplace=True)
df = df[df['title_clean'].astype(str).str.len() > 2]


def clean_title(title):
    if not isinstance(title, str):
        return ""
    title = title.lower()
    title = re.sub(r'[^a-z\s]', ' ', title)  # Supprime ponctuation, chiffres, etc.
    title = re.sub(r'\s+', ' ', title).strip()
    return title
df = df['title_clean']

if __name__ == "__main__":
    print(df.to_string(index= False))
