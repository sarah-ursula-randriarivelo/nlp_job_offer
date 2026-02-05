"""
Clustering Module
=================
K-Means clustering with automatic cluster number optimization.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ClusterOptimizer:
    """Find optimal number of clusters using multiple metrics."""
    
    def __init__(self, min_clusters: int = 2, max_clusters: int = 40):
        """
        Initialize cluster optimizer.
        
        Args:
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.results = None
    
    def find_optimal_clusters(self, X: np.ndarray, max_k: Optional[int] = None) -> int:
        """
        Find optimal number of clusters using silhouette score.
        
        Args:
            X: Feature matrix
            max_k: Maximum K to test (overrides max_clusters if provided)
            
        Returns:
            Optimal number of clusters
        """
        n_samples = X.shape[0]
        
        # Maximum clusters should not exceed half the number of samples
        max_possible_k = min(max_k or self.max_clusters, n_samples // 2)
        
        if max_possible_k < 2:
            print("⚠️  Not enough samples for clustering. Using 2 clusters.")
            return 2
        
        k_range = range(self.min_clusters, max_possible_k + 1)
        scores = []
        
        print(f"🔍 Testing cluster range: {self.min_clusters} to {max_possible_k}")
        
        for k in k_range:
            try:
                # Convert sparse matrix to dense for silhouette_score
                X_dense = X.toarray() if hasattr(X, 'toarray') else X
                
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X_dense, labels)
                scores.append(score)
                
                if k % 5 == 0:  # Print progress every 5 iterations
                    print(f"   K={k}: silhouette={score:.3f}")
                    
            except Exception as e:
                print(f"⚠️  Error at K={k}: {e}")
                scores.append(-1)
        
        # Filter out error scores
        valid_scores = [(k, s) for k, s in zip(k_range, scores) if s > -1]
        
        if not valid_scores:
            print("❌ No valid clustering found. Using 2 clusters.")
            return 2
        
        # Find best K
        best_k, best_score = max(valid_scores, key=lambda x: x[1])
        
        print(f"✅ Optimal clusters: K={best_k} (silhouette={best_score:.3f})")
        
        return best_k
    
    def evaluate_clustering(self, 
                           X: np.ndarray, 
                           max_k: Optional[int] = None,
                           step: int = 1) -> pd.DataFrame:
        """
        Comprehensive clustering evaluation across multiple K values.
        
        Args:
            X: Feature matrix
            max_k: Maximum K to test
            step: Step size for K range
            
        Returns:
            DataFrame with evaluation metrics for each K
        """
        n_samples = X.shape[0]
        max_possible_k = min(max_k or self.max_clusters, n_samples // 2)
        
        k_range = range(self.min_clusters, max_possible_k + 1, step)
        
        results = []
        
        print(f"📊 Comprehensive evaluation from K={self.min_clusters} to K={max_possible_k}")
        
        for k in k_range:
            try:
                # Convert sparse matrix to dense array for metrics that don't support sparse
                X_dense = X.toarray() if hasattr(X, 'toarray') else X
                
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                
                # Multiple metrics (use dense array for compatibility)
                silhouette = silhouette_score(X_dense, labels)
                calinski = calinski_harabasz_score(X_dense, labels)
                davies = davies_bouldin_score(X_dense, labels)
                inertia = kmeans.inertia_
                
                results.append({
                    'n_clusters': k,
                    'silhouette_score': silhouette,
                    'calinski_harabasz': calinski,
                    'davies_bouldin': davies,
                    'inertia': inertia
                })
                
                print(f"   K={k}: silhouette={silhouette:.3f}, CH={calinski:.1f}, DB={davies:.3f}")
                
            except Exception as e:
                print(f"⚠️  Error at K={k}: {e}")
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def get_best_k(self, metric: str = 'silhouette_score') -> int:
        """
        Get best K according to specified metric.
        
        Args:
            metric: Metric to use ('silhouette_score', 'calinski_harabasz', 'davies_bouldin')
            
        Returns:
            Best number of clusters
        """
        if self.results is None:
            raise ValueError("Run evaluate_clustering first!")
        
        if metric == 'davies_bouldin':
            # Lower is better for Davies-Bouldin
            best_row = self.results.loc[self.results[metric].idxmin()]
        else:
            # Higher is better for silhouette and Calinski-Harabasz
            best_row = self.results.loc[self.results[metric].idxmax()]
        
        return int(best_row['n_clusters'])


class JobClusterer:
    """K-Means clustering for job offers."""
    
    def __init__(self, n_clusters: Optional[int] = None, random_state: int = 42):
        """
        Initialize clusterer.
        
        Args:
            n_clusters: Number of clusters (if None, will be optimized)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.labels = None
        self.optimizer = ClusterOptimizer()
    
    def fit_predict(self, X: np.ndarray, optimize: bool = True) -> np.ndarray:
        """
        Fit clustering model and predict labels.
        
        Args:
            X: Feature matrix
            optimize: Whether to optimize cluster number
            
        Returns:
            Cluster labels
        """
        if X.shape[0] < 2:
            raise ValueError("Need at least 2 samples for clustering")
        
        # Optimize number of clusters if not specified
        if self.n_clusters is None or optimize:
            print("🔍 Optimizing number of clusters...")
            self.n_clusters = self.optimizer.find_optimal_clusters(X)
        
        # Fit K-Means
        print(f"🎯 Fitting K-Means with {self.n_clusters} clusters...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state,
            n_init=10
        )
        self.labels = self.kmeans.fit_predict(X)
        
        print(f"✅ Clustering complete!")
        self._print_cluster_distribution()
        
        return self.labels
    
    def _print_cluster_distribution(self):
        """Print cluster size distribution."""
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"\n📊 Cluster distribution:")
        for cluster_id, count in zip(unique, counts):
            print(f"   Cluster {cluster_id}: {count} items ({count/len(self.labels)*100:.1f}%)")
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centroids."""
        if self.kmeans is None:
            raise ValueError("Model not fitted yet!")
        return self.kmeans.cluster_centers_
    
    def evaluate(self, X: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering quality.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.labels is None:
            raise ValueError("Model not fitted yet!")
        
        # Convert sparse matrix to dense for metrics compatibility
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        
        metrics = {
            'silhouette_score': silhouette_score(X_dense, self.labels),
            'calinski_harabasz': calinski_harabasz_score(X_dense, self.labels),
            'davies_bouldin': davies_bouldin_score(X_dense, self.labels),
            'inertia': self.kmeans.inertia_
        }
        
        print(f"\n📈 Clustering Metrics:")
        print(f"   Silhouette Score: {metrics['silhouette_score']:.3f} (closer to 1 is better)")
        print(f"   Calinski-Harabasz: {metrics['calinski_harabasz']:.1f} (higher is better)")
        print(f"   Davies-Bouldin: {metrics['davies_bouldin']:.3f} (lower is better)")
        print(f"   Inertia: {metrics['inertia']:.1f}")
        
        return metrics


def cluster_jobs(X: np.ndarray, 
                n_clusters: Optional[int] = None,
                optimize: bool = True) -> Tuple[np.ndarray, JobClusterer]:
    """
    Convenience function for job clustering.
    
    Args:
        X: Feature matrix
        n_clusters: Number of clusters (None for auto-optimization)
        optimize: Whether to optimize cluster number
        
    Returns:
        Tuple of (cluster labels, fitted clusterer)
    """
    clusterer = JobClusterer(n_clusters=n_clusters)
    labels = clusterer.fit_predict(X, optimize=optimize)
    return labels, clusterer
