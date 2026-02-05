"""
Text Cleaning Module
====================
Comprehensive text cleaning and preprocessing for job offers.
"""

import re
import pandas as pd
from typing import List, Optional
import spacy

try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    print("⚠️  Modèle spaCy 'fr_core_news_sm' non trouvé.")
    print("👉 Installez-le avec : python -m spacy download fr_core_news_sm")
    nlp = None

try:
    from nltk.corpus import stopwords
    FRENCH_STOPWORDS = set(stopwords.words('french'))
except ImportError:
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    FRENCH_STOPWORDS = set(stopwords.words('french'))

# Custom stopwords for Madagascar job market
CUSTOM_STOPWORDS = {
    "serait", "un", "une", "et", "de", "des", "en", "dans", "la", "le", "les", "du", "au", "aux",
    "ou", "car", "donc", "or", "ni", "par", "pour", "sur", "avec", "sans", "trop", "plus", "tres",
    "très", "cela", "ca", "ça", "ce", "mes", "tes", "ses", "son", "leur", "leurs", "e", "se", 
    "trice", "trices", "a"
}

LOCATION_STOPWORDS = {
    "telma", "galaxy", "diego", "tamatave", "axian", "antananarivo", "mahajanga", "toamasina",
    "andraharo", "zone", "futura", "shore", "andranomena", "immeuble", "mdg", "batiment ariane",
    "batiment", "ariane", "tana", "antsirabe", "fianarantsoa", "kube", "majunga", "tolagnaro",
    "er etage", "mdg campus", "campus", "canadien", "ivato", "ambovobe", "sainte marie", "fort dauphin",
    "sambava"
}

ALL_STOPWORDS = FRENCH_STOPWORDS.union(CUSTOM_STOPWORDS)


def get_stopwords(include_locations: bool = True) -> List[str]:
    """
    Get comprehensive stopword list.
    
    Args:
        include_locations: Whether to include location-specific stopwords
        
    Returns:
        List of stopwords
    """
    stopwords = ALL_STOPWORDS.copy()
    if include_locations:
        stopwords = stopwords.union(LOCATION_STOPWORDS)
    return list(stopwords)


def clean_text(text: str, 
               remove_stopwords: bool = True, 
               remove_locations: bool = True,
               preserve_numbers: bool = True) -> str:
    """
    Clean and normalize text for NLP processing.
    
    Args:
        text: Input text to clean
        remove_stopwords: Remove French stopwords
        remove_locations: Remove Madagascar location names
        preserve_numbers: Keep numbers in text
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Normalize line breaks
    text = re.sub(r'\\n+', '\n', text)
    
    # 3. Replace separators with spaces
    text = re.sub(r"[&()-_/]", " ", text)
    
    # 4. Remove non-letter characters (keep accents and spaces)
    if preserve_numbers:
        text = re.sub(r"[^a-zàâäéèêëîïôöùûüç0-9\s]", " ", text)
    else:
        text = re.sub(r"[^a-zàâäéèêëîïôöùûüç\s]", " ", text)
    
    # 5. Remove location names
    if remove_locations:
        location_pattern = "|".join(LOCATION_STOPWORDS)
        text = re.sub(f"({location_pattern})", '', text)
    
    # 6. Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # 7. Remove stopwords
    if remove_stopwords:
        words = text.split()
        filtered_words = [w for w in words if w not in ALL_STOPWORDS]
        text = ' '.join(filtered_words)
    
    return text


def clean_dataframe(df: pd.DataFrame, 
                    columns: Optional[List[str]] = None,
                    suffix: str = "_clean") -> pd.DataFrame:
    """
    Apply cleaning to multiple DataFrame columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to clean (default: all object columns)
        suffix: Suffix to add to cleaned column names
        
    Returns:
        DataFrame with cleaned columns added
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes('object').columns.tolist()
    
    for col in columns:
        if col in df_clean.columns:
            clean_col = f"{col}{suffix}" if not col.endswith(suffix) else col
            # if col == 'title_clean':
            #     df_clean[clean_col] = df_clean[col].apply(clean_text)
            #     df_clean[clean_col] = re.sub(r"\d", " ", df_clean[clean_col])

            df_clean[clean_col] = df_clean[col].apply(clean_text)
            print(f"✅ Cleaned column: {col} → {clean_col}")
    
    return df_clean


def extract_keywords_spacy(text: str, top_n: int = 10) -> List[str]:
    """
    Extract important keywords using spaCy NER and POS tagging.
    
    Args:
        text: Input text
        top_n: Number of top keywords to extract
        
    Returns:
        List of keywords
    """
    if nlp is None:
        return []
    
    doc = nlp(text)
    
    # Extract nouns, proper nouns, and named entities
    keywords = []
    
    # Named entities
    keywords.extend([ent.text.lower() for ent in doc.ents])
    
    # Nouns and proper nouns
    keywords.extend([token.lemma_.lower() for token in doc 
                     if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop])
    
    # Count frequency and return top N
    from collections import Counter
    counter = Counter(keywords)
    return [word for word, _ in counter.most_common(top_n)]


class TextCleaner:
    """Object-oriented interface for text cleaning."""
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 remove_locations: bool = True,
                 preserve_numbers: bool = True):
        """
        Initialize text cleaner with configuration.
        
        Args:
            remove_stopwords: Remove French stopwords
            remove_locations: Remove Madagascar location names
            preserve_numbers: Keep numbers in text
        """
        self.remove_stopwords = remove_stopwords
        self.remove_locations = remove_locations
        self.preserve_numbers = preserve_numbers
    
    def clean(self, text: str) -> str:
        """Clean a single text."""
        return clean_text(text, 
                         self.remove_stopwords, 
                         self.remove_locations, 
                         self.preserve_numbers)
    
    def clean_dataframe(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Clean multiple columns in a DataFrame."""
        return clean_dataframe(df, columns)
