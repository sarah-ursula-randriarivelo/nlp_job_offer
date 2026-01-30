from Portaljob_code_divided.find_optimal_clusters import n_clusters
from Portaljob_code_divided.find_optimal_clusters import df as df_optimal
from collections import Counter
from keyword_mapping import keyword_mapping
from clean_text import df_clean_text

def label_cluster(df, cluster_id):
    # asign a cluster for each document
    titles = df_clean_text[df['cluster'] == cluster_id]['title_clean']
    
    print(titles.str.value)
    if len(titles) == 0:
        return "Autre"

    words = []
    for word in titles:
        titles_parts = word.split()

        # Utiliser plus de termes fréquents, par exemple les 10 premiers unigrammes et bigrammes
        words.extend(titles_parts) # Unigrammes
        words.extend([f"{titles_parts[i]} {titles_parts[i+1]}" for i in range(len(titles_parts)-1)]) # Bigrammes

    frequency = Counter(words)                              

    # Utiliser plus de termes pour l'analyse
    # top_terms = [w for w, c in freq.most_common(5)] # Ancien
    top_terms = [w for w, c in frequency.most_common(10)] # Nouveau: 10 termes
    text = " ".join(top_terms).lower()

    # --- Logique de classement améliorée ---
    # Compter les correspondances pour chaque catégorie
    scores = {}
    for label, keywords in keyword_mapping.items():
        score = sum(1 for k in keywords if k in text) # Compte les correspondances
        if score > 0:
            scores[label] = score

    # Si des correspondances sont trouvées, choisir celle avec le score le plus élevé
    if scores:
        # sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        # best_label = sorted_scores[0][0] # Label avec le meilleur score
        # return best_label.title()
        # Ou simplement: retourner le label avec le score max
        best_label = max(scores, key=scores.get)
        return best_label.title() # Utiliser .title() si vous voulez des majuscules

    # Sinon, utiliser la logique par défaut (premier terme fréquent)
    return top_terms[0].title() if top_terms else f"Groupe {cluster_id}"


# --- Suite du code inchangée ---
cluster_labels = {cid: label_cluster(df, cid) for cid in range(n_clusters)}
df['famille_poste'] = df['cluster'].map(cluster_labels)

print(df[['clean_title', 'famille_poste']].to_string(index=False))

print(df_clean_title)

print(df_optimal)

titles = df_clean_title[df_optimal['cluster'] == 3]['clean_title']
    
print(titles.head(3))