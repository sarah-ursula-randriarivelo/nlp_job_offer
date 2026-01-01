# import pandas as pd
# from sqlalchemy import create_engine
import re
from database_connection import df


# def clean_text_old(text: str) -> str:
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r"[^a-zàâäéèêëîïôöùûüç\s]", " ", text)  # garder lettres + accents
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

def clean_text(text: str) -> str:
    """
    Nettoie un texte pour préparer le matching avec keyword_mapping.
    - Convertit en minuscules
    - Supprime la ponctuation (sauf les accents et espaces)
    - Normalise les espaces
    - Préserve les termes composés courants (ex: "data scientist", "ui ux", "supply chain")
    """
    if not isinstance(text, str):
        return ""

    # # 1. Conversion en minuscules
    # text= text.lower()

    text = re.sub(r'\\n+', '\n', text)

    # 2. Remplacer les tirets, underscores, slash par des espaces (pour séparer les mots)
    # mais conserver les termes composés courants en les normalisant d'abord
    text = re.sub(r"[-_/]", " ", text)

    # 3. Supprimer tout ce qui n’est pas une lettre (avec accents), un espace, ou un chiffre (optionnel)
    # → On garde les chiffres si vous avez des termes comme "bac+2", "caces 3", etc.
    text = re.sub(r"[^a-zàâäéèêëîïôöùûüç0-9\s]", " ", text)

    # 4. Normaliser les espaces multiples
    text = re.sub(r"\s+", " ", text).strip()

    # 5. [OPTION STRATÉGIQUE] Préserver les bigrammes/trigrammes clés en remplaçant l'espace par un tiret bas
    # Ex: "data scientist" → "data_scientist" pour matcher exactement dans keyword_mapping
    # → À faire APRÈS le nettoyage, et seulement si votre matching gère les underscores
    # → Sinon, laissez en espace — mais adaptez votre matching en conséquence.

    return text


# def addition(a,b) :
#     sum = a + b

#     print(sum)

if __name__ == "__main__":
    clean = clean_text(df.to_string(index=False))
    # print(clean)
    # add1= addition(1,2)
    # print(add1)
    # # addition(add1,2)
