import re
from Portaljob_code_divided.jupyter_notebook.database_connection import df as df_database_connection
from Portaljob_code_divided.stopword import stop_words


def clean_text(text: str) -> str:
    """
    Clean text to prepare the matching with keyword_mapping.
        - Convert into lowercase
        - delete all punctuations (not the accents and spaces)
        - Normalise spaces
        - Preserving commonly composed terms like "data scientist", "stagiaire informatique") 
   
    """
    if not isinstance(text, str):
        return ""

    # 1. Convert text to lowercase
    text= text.lower()
    
    # normalise spaces
    text = re.sub(r'\\n+', '\n', text)

    # 2. # Replace underscores, slashes with spaces (to separate words)
    text = re.sub(r"[-_/]", " ", text)

    # 3. # Delete all elements that is not a letter (with accents), a space, or a number
    text = re.sub(r"[^a-zàâäéèêëîïôöùûüç0-9\s]", " ", text)

    # 4. Normalise multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # - replacing unecessary terms like 'telma, galaxy', etc with an empty character
    text = re.sub(r"(telma|galaxy|diego|tamatave|axian|antananarivo|mahajanga|toamasina|andraharo|zone|futura|shore|andranomena|immeuble|mdg|batiment ariane|batiment|ariane|tana|antsirabe|fianarantsoa|kube|majunga|tolagnaro|er etage|mdg|mdg campus|campus)", '', text)

    # Removing french articles inside titles
    text = text.split()
    filtered_word = []
    for word in text:
        if word not in stop_words:
            word = filtered_word.append(word)
    text = ' '.join(filtered_word)

    return text

# apply the cleaning function for each column
for columns in df_database_connection.select_dtypes('object').columns:

    df_database_connection[columns] = df_database_connection[columns].apply(clean_text)

df_clean_text = df_database_connection

if __name__ == "__main__":

    print(df_clean_text['title_clean'].to_string(index= False))
   