import re
from clean_text import df
from stopword import stop_words
# Nettoyage de base
df.dropna(subset=['title_clean'], inplace=True)
df = df[df['title_clean'].astype(str).str.len() > 2]


def clean_title(title):
    if not isinstance(title, str):
        return ""
    title = title.lower()
    title = re.sub(r"(telma|galaxy|diego|tamatave|axian|antananarivo|mahajanga|toamasina|andraharo|zone|mdg immeuble tanashore fututra andranomena|batiment ariane|batiment|ariane|tana|antsirabe|fianarantsoa|kube|majunga|tolagnaro|er etage|mdg|mdg campus|campus|mdg immeuble shore futura andranomena)", '', title)
    title = re.sub(r"[’'‘\"]", " ", title)
    title = re.sub(r'[^a-z\s]', ' ', title)  # Supprime ponctuation, chiffres, etc.
    title = re.sub(r'\s+', ' ', title).strip()
    return title
df['clean_title'] = df['title_clean'].apply(clean_title)
df = df[df['clean_title'] != ""]

if __name__ == "__main__":
     print(df['clean_title'].to_string(index= False))
        