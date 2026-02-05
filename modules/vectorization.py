"""
Text Vectorization Module
==========================
Converts text into numerical vectors using TF-IDF and embeddings.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from typing import Optional, Tuple, List
import pickle
from pathlib import Path


class TextVectorizer:
    """Vectorize text using TF-IDF with optional dimensionality reduction."""
    
    def __init__(self,
                 max_features: int = 100,
                 min_df: int = 2,
                 max_df: float = 0.8,
                 ngram_range: Tuple[int, int] = (1, 2),
                 stop_words: Optional[List[str]] = None,
                 use_svd: bool = False,
                 n_components: int = 50):
        """
        Initialize TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            ngram_range: Range of n-grams (unigrams and bigrams by default)
            stop_words: List of stopwords
            use_svd: Apply SVD for dimensionality reduction
            n_components: Number of SVD components
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words,
            ngram_range=ngram_range,
            lowercase=True
        )
        self.use_svd = use_svd
        self.svd = TruncatedSVD(n_components=n_components, random_state=42) if use_svd else None
        self.feature_names = None
        self.X_tfidf = None
        self.X_reduced = None
    
    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        """
        Fit vectorizer and transform texts.
        
        Args:
            texts: Series of text documents
            
        Returns:
            TF-IDF matrix (optionally reduced with SVD)
        """
        print(f"🔄 Vectorizing {len(texts)} documents...")
        
        # TF-IDF transformation
        self.X_tfidf = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"✅ TF-IDF matrix shape: {self.X_tfidf.shape}")
        print(f"   Features: {len(self.feature_names)}")
        
        # Optional SVD reduction
        if self.use_svd and self.svd:
            self.X_reduced = self.svd.fit_transform(self.X_tfidf)
            explained_var = self.svd.explained_variance_ratio_.sum()
            print(f"✅ SVD reduced to {self.X_reduced.shape[1]} dimensions")
            print(f"   Explained variance: {explained_var:.2%}")
            return self.X_reduced
        
        return self.X_tfidf
    
    def transform(self, texts: pd.Series) -> np.ndarray:
        """Transform new texts using fitted vectorizer."""
        X = self.vectorizer.transform(texts)
        if self.use_svd and self.svd:
            return self.svd.transform(X)
        return X
    
    def get_feature_names(self) -> np.ndarray:
        """Get feature names from vectorizer."""
        return self.feature_names
    
    def get_top_features_per_document(self, n: int = 10) -> pd.DataFrame:
        """
        Get top TF-IDF features for each document.
        
        Args:
            n: Number of top features per document
            
        Returns:
            DataFrame with document index and top features
        """
        if self.X_tfidf is None:
            raise ValueError("Vectorizer not fitted yet. Call fit_transform first.")
        
        results = []
        for doc_idx in range(self.X_tfidf.shape[0]):
            doc_vector = self.X_tfidf[doc_idx].toarray().flatten()
            top_indices = doc_vector.argsort()[-n:][::-1]
            top_features = [(self.feature_names[i], doc_vector[i]) for i in top_indices if doc_vector[i] > 0]
            results.append({
                'doc_id': doc_idx,
                'top_features': top_features
            })
        
        return pd.DataFrame(results)
    
    def save(self, filepath: str):
        """Save vectorizer to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'svd': self.svd,
                'feature_names': self.feature_names
            }, f)
        print(f"💾 Vectorizer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load vectorizer from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.svd = data.get('svd')
            self.feature_names = data['feature_names']
        print(f"📂 Vectorizer loaded from {filepath}")


def create_tfidf_matrix(texts: pd.Series,
                        max_features: int = 100,
                        min_df: int = 2,
                        max_df: float = 0.8,
                        stop_words: Optional[List[str]] = None) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Convenience function to create TF-IDF matrix.
    
    Args:
        texts: Series of text documents
        max_features: Maximum number of features
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        stop_words: List of stopwords
        
    Returns:
        Tuple of (TF-IDF matrix, fitted vectorizer)
    """
    vectorizer = TextVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def get_vocabulary_stats(vectorizer: TextVectorizer) -> pd.DataFrame:
    """
    Get statistics about the vocabulary.
    
    Args:
        vectorizer: Fitted TextVectorizer
        
    Returns:
        DataFrame with vocabulary statistics
    """
    if vectorizer.X_tfidf is None:
        raise ValueError("Vectorizer not fitted yet.")
    
    # Calculate IDF scores
    idf_scores = vectorizer.vectorizer.idf_
    feature_names = vectorizer.get_feature_names()
    
    # Document frequency
    doc_freq = np.array((vectorizer.X_tfidf > 0).sum(axis=0)).flatten()
    
    stats = pd.DataFrame({
        'feature': feature_names,
        'idf': idf_scores,
        'doc_frequency': doc_freq
    })
    
    return stats.sort_values('idf', ascending=False)
