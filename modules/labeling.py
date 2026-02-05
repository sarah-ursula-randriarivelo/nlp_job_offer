"""
Cluster Labeling Module
========================
Label clusters using keyword mapping and top TF-IDF terms.
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Tuple


# Keyword mapping for Madagascar job market
KEYWORD_MAPPING = {
    'Finance / Comptabilite': [
        "comptable", "audit", "finance", "controle de gestion", "tresorerie",
        "fiscalite", "reporting", "budget", "analyse financiere", "comptabilite",
        "rapprochement bancaire", "saisie comptable", "superviseur comptable",
        "credit", "micro finance"
    ],
    'Business Intelligence / Data': [
        "bi", "data", "analyste", "power bi", "sql", "python", "machine learning", 
        "reporting", "data analyst", "data scientist", "etl", "data engineering", 
        "tableau", "visualisation", "big data", "ia"
    ],
    'Informatique / Developpement': [
        "developpeur", "devops", "logiciel", "fullstack", "backend", "frontend", 
        "java", "c#", "php", "angular", "web", "mobile", "technologie",
        "python", "javascript", "html", "css", "cloud", "database", "api", 
        "framework", "reseau", "cybersecurite", "developpement", "graphisme", 
        "wordpress", "informatique", "stagiaire informatique", "it", "stagiaire it"
    ],
    'Marketing / Digital': [
        "seo", "marketing", "digital", "social media", "growth", "brand", 
        "communication", "publicite", "content", "inbound", "outbound", 
        "email marketing", "campagne", "community manager", "branding", 
        "info-graphie", "stagiaire marketing"
    ],
    'Supply Chain / Logistique': [
        "logistique", "supply chain", "transport", "entrepot", "import", "export",
        "gestion stock", "warehouse", "inventory", "approvisionnement", "fret", 
        "livraison", "procurement", "supply", "magasin", "magasinier", "fabrication"
    ],
    'Industriel / Technique': [
        "production", "qualite", "maintenance", "technicien", "usine", "industriel", 
        "genie", "mecanique", "electronique", "automatisme", "robotique", 
        "energetique", "instrumentation", "controle qualite", "frigoriste", 
        "chaudronnier", "genie electrique", "electrique", "electricien", "housekeeping"
    ],
    'RH / Management': [
        "ressources humaines", "rh", "recrutement", "formation", "paie", "manager", 
        "superviseur recrutement", "superviseur rh", "gestion personnel", 
        "gestion carriere", "leadership", "team management", "relations sociales", 
        "responsable", "talent acquisition", "talent"
    ],
    'Commercial / Vente': [
        "commercial", "vente", "business", "account manager", "prospection",
        "relation client", "fidelisation", "negociation", "partenaire", "contrat", 
        "business development", "negociateur", "client", "magasin"
    ],
    'Support / Service Client': [
        "support", "service client", "helpdesk", "hotline", "assistance", 
        "support technique", "sav", "relation client", "support it", 
        "support fonctionnel", "support client", "back office", 
        "support recrutement", "process btob", "service", "support comptable", 
        "support clientele", "support operation", "team lead", "team", 
        "teleconseiller", "tele", "teleconseillers", "tourisme", "hotellerie"
    ],
    'BTP / Construction': [
        'btp', 'batiment', 'construction', 'ouvrier', 'conducteur', 'travaux',
        'chantier', 'macon', 'charpentier', 'electricien', 'plombier', 
        'ingenierie', 'architecte', 'genie civil'
    ],
    'Agent / Operateur': [
        'agent', 'magasinier', 'manutention', 'guichet', 'operateur',
        'preparateur de commandes', 'logisticien', 'agent de quai', 
        'operateur machine', 'teleoperateurs', 'teleoperateur'
    ],
    'Mecanique / Vehicule': [
        'mecanicien', 'mecanique', 'automobile', 'vehicule', 'camion', 
        'entretien technique', 'diagnostique', 'reparation', 'garage'
    ],
    'Cuisine / Restauration': [
        'cuisinier', 'chef de partie', 'cuisine', 'restauration', 'boulanger',
        'patissier', 'barman', 'serveur', 'commis', 'preparation culinaire', 
        "receptionniste"
    ],
    'Artisanat / Fabrication': [
        'artisan', 'menuisier', 'serrurier', 'couturier', 'tapissier',
        'ebeniste', 'bijoutier', 'horloger', 'fabrication'
    ],
    'Sante / Paramedical': [
        'infirmier', 'aide soignant', 'sage femme', 'kinesthesitherapeute',
        'laboratoire', 'radiologie', 'sante', 'paramedical'
    ],
    'Services a la Personne': [
        'aide domicile', 'auxiliaire vie', 'aide ménagère', 'garde enfant',
        'services a la personne', 'accompagnement'
    ],
    'Education / Formation': [
        'professeur', 'enseignant', 'formateur', 'education', 'pedagogie',
        'instructeur', 'tuteur'
    ],
    'Securite / Gardiennage': [
        'agent securite', 'surveillance', 'gardiennage', 'controle acces',
        'brigade', 'vigile'
    ],
    'Agriculture / Environnement': [
        'agriculteur', 'agronome', 'environnement', 'elevage', 'agriculture',
        'foresterie', 'paysan', "agronomie", "ruraux"
    ]
}


class ClusterLabeler:
    """Label clusters using keyword mapping and frequency analysis."""
    
    def __init__(self, keyword_mapping: Optional[Dict[str, List[str]]] = None):
        """
        Initialize cluster labeler.
        
        Args:
            keyword_mapping: Dictionary mapping category names to keywords
        """
        self.keyword_mapping = keyword_mapping or KEYWORD_MAPPING
    
    def label_cluster(self, 
                     df: pd.DataFrame,
                     cluster_id: int,
                     text_column: str = 'title_cleaned',
                     top_n_terms: int = 10) -> str:
        """
        Label a single cluster based on frequent terms and keyword matching.
        
        Args:
            df: DataFrame with clustered data
            cluster_id: Cluster ID to label
            text_column: Column containing cleaned text
            top_n_terms: Number of top frequent terms to consider
            
        Returns:
            Cluster label
        """
        # Get all texts from this cluster
        cluster_texts = df[df['cluster'] == cluster_id][text_column]
        
        if len(cluster_texts) == 0:
            return "Autre"
        
        # Extract words from all titles
        words = []
        for text in cluster_texts:
            if isinstance(text, str):
                text_parts = text.split()
                # Add unigrams
                words.extend(text_parts)
                # Add bigrams
                words.extend([f"{text_parts[i]} {text_parts[i+1]}" 
                             for i in range(len(text_parts)-1)])
        
        # Get most frequent terms
        frequency = Counter(words)
        top_terms = [w for w, c in frequency.most_common(top_n_terms)]
        combined_text = " ".join(top_terms).lower()
        
        # Score each category based on keyword matches
        scores = {}
        for label, keywords in self.keyword_mapping.items():
            score = sum(1 for k in keywords if k in combined_text)
            if score > 0:
                scores[label] = score
        
        # Return best matching label
        if scores:
            best_label = max(scores, key=scores.get)
            return best_label.title()
        
        # Fallback to most frequent term
        return top_terms[0].title() if top_terms else f"Groupe {cluster_id}"
    
    def label_all_clusters(self, 
                          df: pd.DataFrame,
                          text_column: str = 'title_cleaned',
                          label_column: str = 'famille_poste') -> pd.DataFrame:
        """
        Label all clusters in the DataFrame.
        
        Args:
            df: DataFrame with cluster assignments
            text_column: Column containing cleaned text
            label_column: Column name for cluster labels
            
        Returns:
            DataFrame with cluster labels added
        """
        if 'cluster' not in df.columns:
            raise ValueError("DataFrame must have 'cluster' column")
        
        n_clusters = df['cluster'].nunique()
        print(f"🏷️  Labeling {n_clusters} clusters...")
        
        # Create label mapping
        cluster_labels = {}
        for cluster_id in range(n_clusters):
            label = self.label_cluster(df, cluster_id, text_column)
            cluster_labels[cluster_id] = label
            
            cluster_size = (df['cluster'] == cluster_id).sum()
            print(f"   Cluster {cluster_id} ({cluster_size} items): {label}")
        
        # Add labels to DataFrame
        df[label_column] = df['cluster'].map(cluster_labels)
        
        print(f"✅ All clusters labeled!")
        return df
    
    def get_cluster_keywords(self, 
                            df: pd.DataFrame,
                            cluster_id: int,
                            text_column: str = 'title_cleaned',
                            top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Get top keywords for a specific cluster.
        
        Args:
            df: DataFrame with clustered data
            cluster_id: Cluster ID
            text_column: Column containing cleaned text
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, frequency) tuples
        """
        cluster_texts = df[df['cluster'] == cluster_id][text_column]
        
        words = []
        for text in cluster_texts:
            if isinstance(text, str):
                words.extend(text.split())
        
        frequency = Counter(words)
        return frequency.most_common(top_n)
    
    def get_cluster_summary(self, 
                           df: pd.DataFrame,
                           text_column: str = 'title_cleaned') -> pd.DataFrame:
        """
        Get summary statistics for all clusters.
        
        Args:
            df: DataFrame with clustered and labeled data
            text_column: Column containing cleaned text
            
        Returns:
            DataFrame with cluster summaries
        """
        if 'cluster' not in df.columns:
            raise ValueError("DataFrame must have 'cluster' column")
        
        summaries = []
        
        for cluster_id in df['cluster'].unique():
            cluster_df = df[df['cluster'] == cluster_id]
            keywords = self.get_cluster_keywords(df, cluster_id, text_column, top_n=10)
            
            summary = {
                'cluster_id': cluster_id,
                'size': len(cluster_df),
                'percentage': len(cluster_df) / len(df) * 100,
                'top_keywords': ', '.join([k for k, _ in keywords[:5]]),
                'label': cluster_df['famille_poste'].iloc[0] if 'famille_poste' in cluster_df else 'N/A'
            }
            summaries.append(summary)
        
        return pd.DataFrame(summaries).sort_values('size', ascending=False)


def label_clusters(df: pd.DataFrame,
                  text_column: str = 'title_cleaned',
                  label_column: str = 'famille_poste',
                  keyword_mapping: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    """
    Convenience function to label all clusters.
    
    Args:
        df: DataFrame with cluster assignments
        text_column: Column containing cleaned text
        label_column: Column name for cluster labels
        keyword_mapping: Custom keyword mapping (optional)
        
    Returns:
        DataFrame with cluster labels
    """
    labeler = ClusterLabeler(keyword_mapping)
    return labeler.label_all_clusters(df, text_column, label_column)
