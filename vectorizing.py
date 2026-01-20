from clean_title import df
from Portaljob_code_divided.jupyter_notebook.stopword import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer



if len(df) < 1:
    raise ValueError("📊 Pas assez de données pour effectuer un clustering.")

# converts texts into matrix
vectorizer = TfidfVectorizer(
    max_features= 50,
    min_df = 2,
    max_df = 0.8, 
    stop_words= stop_words,           
    ngram_range=(1, 2),
    lowercase=True
)

X = vectorizer.fit_transform(df['clean_title'])

X_document_name = vectorizer.get_feature_names_out(X)

row, col = X.nonzero()

document_name = X.data



if __name__ == "__main__":

    for (row, col, data) in zip(row, col, document_name):
        print(f"{row}, {X_document_name[col]}, {data}")


