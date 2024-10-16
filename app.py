import streamlit as st
import gensim.downloader as api
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Titre de l'application
st.title('Manipulation de mots avec des Embeddings')

# Charger le modèle GloVe pré-entraîné
@st.cache_resource
def load_glove_model():
    return api.load("glove-wiki-gigaword-100")  # Modèle GloVe avec 100 dimensions

# Charger le modèle
with st.spinner('Chargement du modèle GloVe...'):
    model = load_glove_model()


# Nouvelle fonctionnalité : Calcul de la similarité entre deux mots
st.header("Calculer la similarité entre deux mots")
word_a = st.text_input("Entrez le premier mot (ex: king)", "")
word_b = st.text_input("Entrez le deuxième mot (ex: queen)", "")

if st.button("Calculer la similarité"):
    try:
        # Récupérer les vecteurs des deux mots
        vector_a = model[word_a]
        vector_b = model[word_b]

        # Calcul de la similarité cosinus
        similarity = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        st.write(f"La similarité entre '{word_a}' et '{word_b}' est : **{similarity:.4f}**")
    except KeyError as e:
        st.error(f"Le mot '{e.args[0]}' n'est pas dans le modèle.")

# Interface Streamlit pour l'entrée des mots
st.header("Opération : Addition et Soustraction de mots")
word1 = st.text_input("Mot de base (ex : king)", "")
word2 = st.text_input("Mot à soustraire (ex : man)", "")
word3 = st.text_input("Mot à ajouter (ex : woman)", "")

result = None

# Calculer l'opération de mots
if st.button("Calcule"):
    try:
        # Calcul de l'opération vectorielle
        result = model.most_similar(positive=[word1, word3], negative=[word2], topn=1)
        st.write(f"Result: '{word1} - {word2} + {word3}' = **{result[0][0]}**")
    except KeyError as e:
        st.error(f"Le mot '{e.args[0]}' n'est pas dans le vocabulaire.")

# Option pour visualiser les relations entre les mots en 3D
if st.checkbox("Visualize word relations in 3D", key="3d_visualization_checkbox"):
    if result:
        try:
            # Récupérer les mots et leurs vecteurs
            words = [word1, word2, word3, result[0][0]]
            word_vectors = np.array([model[word] for word in words])

            # Réduire les dimensions à 3D avec PCA
            pca = PCA(n_components=3)
            reduced_vectors = pca.fit_transform(word_vectors)

            # Créer un graphique en 3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2])

            # Annoter chaque point avec le mot correspondant
            for i, word in enumerate(words):
                ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2], word, fontsize=12)

            st.pyplot(fig)
        except KeyError as e:
            st.error(f"Le mot '{e.args[0]}' n'est pas dans le vocabulaire.")
    else:
        st.warning("Veuillez d'abord calculer un résultat avant de visualiser.")
