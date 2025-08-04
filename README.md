
# Système de Recherche d'Images Basé sur le Contenu

 projet propose une application Web de **recherche d'images basée sur le contenu (CBIR)**

##  Fonctionnalités principales

-  Authentification des utilisateurs
-  Téléversement d’une image requête
-  Extraction de caractéristiques : GLCM, Haralick, BiT, concaténation
-  Choix de la mesure de distance : Euclidienne, Manhattan, Chebychev, Canberra
-  Affichage des images les plus similaires
- Interface Web réalisée avec **Streamlit**



##  Descripteurs de caractéristiques

- **GLCM_RGB** : Matrice de cooccurrence de niveaux de gris par canal R, G, B.
- **Haralick_RGB** : Statistiques de texture Haralick sur chaque canal RGB.
- **BiT_RGB** : Descripteur bio-inspiré pour textures.
- **Concat_RGB** : Fusion des trois descripteurs.

> Les descripteurs sont extraits via `extracion.py` et stockés dans `/signatures/*.npy`.



###  Installer les dépendances :

streamlit
numpy
opencv-python
scikit-image
mahotas
Pillow

---

## Génération des signatures

j'ai mis  les images du dossier `dataset/`, puis exécutez :(python extracion.py)

Pour créer les fichiers :

* `signatures_GLCM_RGB.npy`
* `signatures_Haralick_RGB.npy`
* `signatures_BiT_RGB.npy`
* `signatures_Concat_RGB.npy`

---

## Lancement de l'application Web

streamlit run app.py

Identifiants par défaut :
* Utilisateur : `admin` – Mot de passe : `admin`
* Utilisateur : `ARISS` – Mot de passe : `2000`

