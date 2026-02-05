"""
Visualization Module
====================
Interactive visualizations for job clustering analysis using Plotly.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional, Tuple
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def plot_cluster_distribution(df: pd.DataFrame, 
                              cluster_column: str = 'cluster',
                              label_column: Optional[str] = 'famille_poste',
                              title: str = 'Job Offer Cluster Distribution') -> go.Figure:
    """
    Create interactive bar chart of cluster distribution.
    
    Args:
        df: DataFrame with cluster assignments
        cluster_column: Column containing cluster IDs
        label_column: Column containing cluster labels
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Count by cluster
    if label_column and label_column in df.columns:
        cluster_counts = df[label_column].value_counts().reset_index()
        cluster_counts.columns = ['label', 'count']
        x_col, color_col = 'label', 'label'
    else:
        cluster_counts = df[cluster_column].value_counts().reset_index()
        cluster_counts.columns = ['cluster', 'count']
        x_col, color_col = 'cluster', 'cluster'
    
    fig = px.bar(cluster_counts,
                 x=x_col,
                 y='count',
                 color=color_col,
                 title=title,
                 labels={'count': 'Number of Job Offers', x_col: 'Cluster'},
                 text='count')
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(showlegend=False, height=500)
    
    return fig


def plot_cluster_scatter_2d(df: pd.DataFrame,
                           X: np.ndarray,
                           cluster_column: str = 'cluster',
                           label_column: Optional[str] = 'famille_poste',
                           method: str = 'pca',
                           title: Optional[str] = None,
                           hover_data: Optional[list] = None) -> go.Figure:
    """
    Create 2D scatter plot of clusters using dimensionality reduction.
    
    Args:
        df: DataFrame with cluster assignments
        X: Feature matrix
        cluster_column: Column containing cluster IDs
        label_column: Column containing cluster labels
        method: Dimensionality reduction method ('pca' or 'tsne')
        title: Plot title
        hover_data: Additional columns to show on hover
        
    Returns:
        Plotly figure
    """
    print(f"🎨 Creating 2D visualization using {method.upper()}...")
    
    # Dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
        explained_var = reducer.explained_variance_ratio_
        method_label = f'PCA (var: {explained_var[0]:.1%} + {explained_var[1]:.1%})'
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df)-1))
        coords = reducer.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
        method_label = 't-SNE'
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'.")
    
    # Create DataFrame for plotting
    plot_df = df.copy()
    plot_df['x'] = coords[:, 0]
    plot_df['y'] = coords[:, 1]
    
    # Determine color column
    color_col = label_column if label_column and label_column in df.columns else cluster_column
    
    # Create scatter plot
    if title is None:
        title = f'Job Clusters Visualization ({method_label})'
    
    fig = px.scatter(plot_df,
                    x='x',
                    y='y',
                    color=color_col,
                    hover_data=hover_data or ['title_cleaned'] if 'title_cleaned' in plot_df else None,
                    title=title,
                    labels={'x': f'{method.upper()} Component 1', 
                           'y': f'{method.upper()} Component 2'})
    
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='white')))
    fig.update_layout(height=600, hovermode='closest')
    
    print("✅ Visualization created!")
    return fig


def plot_cluster_keywords(df: pd.DataFrame,
                         cluster_id: int,
                         text_column: str = 'title_cleaned',
                         top_n: int = 15,
                         title: Optional[str] = None) -> go.Figure:
    """
    Plot top keywords for a specific cluster.
    
    Args:
        df: DataFrame with cluster assignments
        cluster_id: Cluster ID to visualize
        text_column: Column containing cleaned text
        top_n: Number of top keywords to show
        title: Plot title
        
    Returns:
        Plotly figure
    """
    from collections import Counter
    
    # Get cluster texts
    cluster_texts = df[df['cluster'] == cluster_id][text_column]
    
    # Count words
    words = []
    for text in cluster_texts:
        if isinstance(text, str):
            words.extend(text.split())
    
    word_counts = Counter(words).most_common(top_n)
    
    # Create DataFrame
    keywords_df = pd.DataFrame(word_counts, columns=['keyword', 'frequency'])
    
    # Plot
    if title is None:
        label = df[df['cluster'] == cluster_id]['famille_poste'].iloc[0] if 'famille_poste' in df.columns else f'Cluster {cluster_id}'
        title = f'Top Keywords: {label}'
    
    fig = px.bar(keywords_df,
                x='frequency',
                y='keyword',
                orientation='h',
                title=title,
                labels={'frequency': 'Frequency', 'keyword': 'Keyword'},
                text='frequency')
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    
    return fig


def plot_cluster_comparison(df: pd.DataFrame,
                           text_column: str = 'title_cleaned',
                           top_n_clusters: int = 6,
                           keywords_per_cluster: int = 10) -> go.Figure:
    """
    Create multi-cluster keyword comparison.
    
    Args:
        df: DataFrame with cluster assignments
        text_column: Column containing cleaned text
        top_n_clusters: Number of largest clusters to compare
        keywords_per_cluster: Keywords to show per cluster
        
    Returns:
        Plotly figure with subplots
    """
    from collections import Counter
    
    # Get top N largest clusters
    top_clusters = df['cluster'].value_counts().head(top_n_clusters).index.tolist()
    
    # Create subplots
    rows = (len(top_clusters) + 1) // 2
    cols = 2
    
    fig = make_subplots(rows=rows, cols=cols,
                       subplot_titles=[f"Cluster {cid}" for cid in top_clusters])
    
    for idx, cluster_id in enumerate(top_clusters):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        # Get keywords
        cluster_texts = df[df['cluster'] == cluster_id][text_column]
        words = []
        for text in cluster_texts:
            if isinstance(text, str):
                words.extend(text.split())
        
        word_counts = Counter(words).most_common(keywords_per_cluster)
        keywords, frequencies = zip(*word_counts) if word_counts else ([], [])
        
        # Add bar trace
        fig.add_trace(
            go.Bar(x=list(frequencies), y=list(keywords), orientation='h',
                  showlegend=False),
            row=row, col=col
        )
    
    fig.update_layout(height=300*rows, title_text="Cluster Keyword Comparison")
    fig.update_xaxes(title_text="Frequency")
    
    return fig


def create_wordcloud(df: pd.DataFrame,
                    cluster_id: Optional[int] = None,
                    text_column: str = 'title_cleaned',
                    width: int = 800,
                    height: int = 400,
                    background_color: str = 'white') -> plt.Figure:
    """
    Create word cloud for cluster or entire dataset.
    
    Args:
        df: DataFrame with text data
        cluster_id: Specific cluster ID (None for all data)
        text_column: Column containing cleaned text
        width: Word cloud width
        height: Word cloud height
        background_color: Background color
        
    Returns:
        Matplotlib figure
    """
    # Filter by cluster if specified
    if cluster_id is not None:
        data = df[df['cluster'] == cluster_id]
        title = f'Word Cloud - Cluster {cluster_id}'
        if 'famille_poste' in df.columns:
            label = data['famille_poste'].iloc[0]
            title = f'Word Cloud - {label}'
    else:
        data = df
        title = 'Word Cloud - All Job Offers'
    
    # Combine all text
    text = ' '.join(data[text_column].dropna().astype(str))
    
    # Generate word cloud
    wordcloud = WordCloud(width=width, 
                         height=height,
                         background_color=background_color,
                         colormap='viridis').generate(text)
    
    # Plot
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_metrics_evolution(metrics_df: pd.DataFrame,
                          title: str = 'Clustering Metrics vs Number of Clusters') -> go.Figure:
    """
    Plot clustering evaluation metrics across different K values.
    
    Args:
        metrics_df: DataFrame from ClusterOptimizer.evaluate_clustering()
        title: Plot title
        
    Returns:
        Plotly figure with multiple y-axes
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Silhouette score (primary y-axis)
    fig.add_trace(
        go.Scatter(x=metrics_df['n_clusters'], 
                  y=metrics_df['silhouette_score'],
                  name='Silhouette Score',
                  mode='lines+markers',
                  line=dict(color='blue', width=2)),
        secondary_y=False
    )
    
    # Davies-Bouldin (primary y-axis)
    fig.add_trace(
        go.Scatter(x=metrics_df['n_clusters'],
                  y=metrics_df['davies_bouldin'],
                  name='Davies-Bouldin',
                  mode='lines+markers',
                  line=dict(color='red', width=2)),
        secondary_y=False
    )
    
    # Calinski-Harabasz (secondary y-axis)
    fig.add_trace(
        go.Scatter(x=metrics_df['n_clusters'],
                  y=metrics_df['calinski_harabasz'],
                  name='Calinski-Harabasz',
                  mode='lines+markers',
                  line=dict(color='green', width=2)),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Number of Clusters (K)")
    fig.update_yaxes(title_text="Silhouette / Davies-Bouldin", secondary_y=False)
    fig.update_yaxes(title_text="Calinski-Harabasz", secondary_y=True)
    fig.update_layout(title=title, height=500, hovermode='x unified')
    
    return fig


def create_cluster_dashboard(df: pd.DataFrame,
                            X: np.ndarray,
                            text_column: str = 'title_cleaned') -> None:
    """
    Create comprehensive dashboard with multiple visualizations.
    
    Args:
        df: DataFrame with cluster assignments
        X: Feature matrix
        text_column: Column containing cleaned text
    """
    print("📊 Creating comprehensive dashboard...")
    
    # 1. Distribution
    fig1 = plot_cluster_distribution(df)
    fig1.show()
    
    # 2. 2D Scatter (PCA)
    fig2 = plot_cluster_scatter_2d(df, X, method='pca', hover_data=['title_cleaned'])
    fig2.show()
    
    # 3. Cluster comparison
    fig3 = plot_cluster_comparison(df, text_column=text_column)
    fig3.show()
    
    print("✅ Dashboard created! Check the plots above.")
