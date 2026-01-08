import re
from clean_text import df_clean_text
from stopword import stop_words


# Keep lines with title length longer than 2
df = df_clean_text


# this function cleans the 'title_clean' columns only by
    # - replacing unecessary termns like 'telma, galaxy', etc
    # - replacing accents and punctuations with a space
    # - remove articles in french like 'le, la, des, un, une', etc

def clean_title(title):
    if not isinstance(title, str):
        return ""
    title = title.lower()
    title = re.sub(r"(telma|galaxy|diego|tamatave|axian|antananarivo|mahajanga|toamasina|andraharo|zone|mdg immeuble tanashore fututra andranomena|batiment ariane|batiment|ariane|tana|antsirabe|fianarantsoa|kube|majunga|tolagnaro|er etage|mdg|mdg campus|campus|mdg immeuble shore futura andranomena)", '', title)


    # Removing french articles inside titles
    title = title.split()
    filtered_word = []
    for word in title:
        if word not in stop_words:
            word = filtered_word.append(word)
    title = ' '.join(filtered_word)

    return title


df['clean_title'] = df['title_clean'].apply(clean_title)

# Filtering empty title
df_clean_title = df[df['clean_title'] != ""]

if __name__ == "__main__":
     
     print(df_clean_title['clean_title'] .to_string(index= False))

        