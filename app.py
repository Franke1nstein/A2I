import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import numpy as np
import time
from model_utils import model_prediction

selected_model_path = None

#Sidebar
st.sidebar.title("Page")
app_mode = st.sidebar.selectbox("Sélectionner une page",["Accueil","À propos","Reconnaissance des maladies"])

#Main Page
if(app_mode=="Accueil"):
    st.header("SYSTÈME DE RECONNAISSANCE DES MALADIES DES PLANTES")
    image_path = "home_page.jpeg"
    st.image(image_path,use_container_width=True)
    st.markdown("""
   

    ### À propos 
    Apprenez-en plus sur le projet, notre équipe et nos objectifs sur la page **À propos**.
    """)

#About Project
elif(app_mode=="À propos"):
    st.header("À propos")
    st.markdown("""
                 Bienvenue dans le Système de Reconnaissance des Maladies des Plantes ! 🌿🔍

    Notre mission est d'aider à identifier les maladies des plantes par les symptomes au niveau des feuilles. TFournisser une image d'une plante et notre système l'analysera pour détecter tout signe de maladie. Ensemble, protégeons nos cultures et assurons une récolte plus saine !

    ### Comment ça marche ?
    1. **Télécharger l'image :** Allez à la page **Reconnaissance des maladies** et téléchargez une image d'une plante présentant des symptômes de maladies.
    2. **Chosir le modèle :** Parmi les Choix disponibles ou laisser par défauts.
    3. **Analyse :** Notre système traitera l'image à l'aide d'algorithmes avancés pour identifier les maladies potentielles.
    4. **Résultats :** Consultez les résultats et les recommandations pour les actions à entreprendre.

    ### Avantages

    - **Facile à utiliser :** Interface simple et intuitive pour une expérience utilisateur fluide.
    - **Rapide :** Recevez les résultats en quelques secondes, ce qui permet une prise de décision rapide.
    - **Adaptif :** Differents Modèle d'Intelligence Artificielle disponibles pour la prédiction des  maladies.

    ### Commencez
    Cliquez sur la page **Reconnaissance des maladies** dans la barre latérale pour télécharger une image et découvrir la puissance de notre Système de Reconnaissance des Maladies des Plantes !
              
    #### À propos du jeu de données(DATASET)
    Ce jeu de données a été recréé à l'aide d'une augmentation hors ligne à partir du jeu de données original.
    Ce jeu de données contient environ 87 000 images de feuilles de cultures saines et malades, classées en 38 classes différentes. L'ensemble du jeu de données est divisé en un rapport 80/20 pour les ensembles d'apprentissage et de validation, en préservant la structure du répertoire.
    Un nouveau répertoire contenant 33 images de test a été créé ultérieurement à des fins de prédiction.
    
    """)


#Prediction Page

elif(app_mode=="Reconnaissance des maladies"):
    st.header("Reconnaissance des maladies")
    st.spinner(text="En Cours..")
    
    # --- AJOUT DU SELECTBOX POUR LE CHOIX DU MODÈLE ---
    st.sidebar.header("Modèle de Prédiction")
    model_options = [] # Liste pour stocker les noms des modèles
    model_dir = "./Model/" # Répertoire où sont stockés vos modèles
    
    # Récupérer la liste des fichiers .h5 dans le répertoire des modèles
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith(".h5")]
        if model_files:
            model_options = model_files
        else:
            st.sidebar.warning("Aucun fichier de modèle (.h5) trouvé dans le répertoire 'Model/'.")
    else:
        st.sidebar.error(f"Répertoire de modèles '{model_dir}' non trouvé.")
    
    # Selectbox pour choisir le modèle
    selected_model_file = st.sidebar.selectbox("Sélectionner le modèle", model_options)
    # --- ASSIGN VALUE TO THE TOP-LEVEL selected_model_path VARIABLE ---
    selected_model_path = os.path.join(model_dir, selected_model_file)

    test_image = st.file_uploader("Choisissez une image :", type=["png", "jpg", "jpeg"])
    if(st.button("Afficher l'image")):
        if test_image is not None:
            st.image(test_image,width=4,use_container_width=True)
        else:
            st.warning("Veuillez télécharger une image avant de cliquer sur 'Afficher l'image'.")


    #Predict button
    if st.button("Predire"):
        if test_image is not None:

            with st.spinner(text="Analyse en Cours..."):
                # --- PRE-TRAITEMENT DE L'IMAGE (REDIMENSIONNEMENT à 256x256) ---
                image = Image.open(test_image)
                resized_image = image.resize((256, 256)) # Redimensionner à 256x256
                img_array = np.array(resized_image)
                img_array = np.expand_dims(img_array, axis=0)
                # --- FIN DU PRE-TRAITEMENT ---

                predicted_class_index, predictions_array = model_prediction(test_image, selected_model_path)

            # Lecture des étiquettes (doit être en dehors du spinner pour afficher le résultat après la prédiction)
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                            'Tomato___healthy']
            try:    
              predicted_class_name = class_name[predicted_class_index]
            
              # --- Obtenir le POURCENTAGE de confiance ---
              max_probability = np.max(predictions_array) # Trouver la probabilité maximale
              percentage_confidence = max_probability * 100 # Convertir en pourcentage
              percentage_confidence_formatted = "{:.2f}".format(percentage_confidence)
            
              st.success(f"Le modèle prédit qu'il s'agit de : **{predicted_class_name}** (Confiance : **{percentage_confidence_formatted}%**)")
            except IndexError: # Capture de l'erreur IndexError
                st.error("Erreur lors de la prédiction : Index de classe invalide détecté. Veuillez réessayer ou vérifier l'image.")
                st.warning("Il pourrait y avoir un problème avec le modèle ou les étiquettes de classes. Vérifiez que le modèle est correctement chargé et que la liste des noms de classes correspond aux prédictions du modèle.")
        else:
            st.warning("Veuillez télécharger une image avant de lancer la prédiction.") # Message d'avertissement si aucune image n'est téléchargée