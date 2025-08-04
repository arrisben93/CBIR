import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from descripteurs import glcm_RGB, haralick_feat_RGB, bitdesc_feat_RGB, concatenation_RGB
from Distances import Recherche_images_similaires


# Configuration

USERS = {"ARISS": "2000", "admin": "admin"}
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Login

def login():
    st.title(" Accès à l'Explorateur d'Images")
    user = st.text_input("Identifiant")
    password = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter"):
        if USERS.get(user) == password:
            st.session_state.logged_in = True
            st.success("Connexion réussie")
        else:
            st.error("Identifiants invalides")

if not st.session_state.logged_in:
    login()
    st.stop()


st.title(""""
## Système Intelligent de Recherche d'Images
###  Basé sur l'Analyse du Contenu Visuel
""")




# Chargement
signatures_options = {
    "GLCM_RGB": np.load("signatures/signatures_GLCM_RGB.npy", allow_pickle=True),
    "Haralick_RGB": np.load("signatures/signatures_Haralick_RGB.npy", allow_pickle=True),
    "BiT_RGB": np.load("signatures/signatures_BiT_RGB.npy", allow_pickle=True),
    "concaténation des trois": np.load("signatures/signatures_Concat_RGB.npy", allow_pickle=True),
}

data = np.load("signatures/signatures_Concat_RGB.npy", allow_pickle=True)
print(len(data))

descripteur_nom = st.selectbox("Choisir le descripteur caractéristiques :", list(signatures_options.keys()))
distance_nom = st.selectbox("Selectionner la mesure de distance", ["eucledienne", "manhattan", "chebyshav", "canberra"])
k = st.slider("Nombre d'images identique à afficher", 1, 20, 5)



uploaded_file = st.file_uploader("Téléversé une image à rechercher :", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image de requête", use_container_width=True)

    with open("requete_temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if descripteur_nom == "GLCM_RGB":
        vecteur = glcm_RGB("requete_temp.jpg")
    elif descripteur_nom == "Haralick_RGB":
        vecteur = haralick_feat_RGB("requete_temp.jpg")
    elif descripteur_nom == "BiT_RGB":
        vecteur = bitdesc_feat_RGB("requete_temp.jpg")
    else:
        vecteur = concatenation_RGB("requete_temp.jpg")

    resultats = Recherche_images_similaires(
        bdd_signatures=signatures_options[descripteur_nom],
        carac_requete=vecteur,
        Distances=distance_nom,
        K=k
    )







    st.subheader(" Résultats similaires")
    cols = st.columns(k)
    for i, (score, label, chemin_relatif) in enumerate(resultats):
        chemin_image = os.path.join("dataset", chemin_relatif)
        if os.path.exists(chemin_image):
            img = Image.open(chemin_image)
            cols[i].image(img, caption=f"{label} ({score:.2f})", use_container_width=True)
