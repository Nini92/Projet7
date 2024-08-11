<<<<<<< HEAD
import os
import gc
import numpy as np
import pandas as pd
import pickle
import logging
from flask import Flask, request, jsonify
from sklearn.ensemble import GradientBoostingClassifier
import warnings
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
=======
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import logging
>>>>>>> c8aba6cb345e78fa3356b8e7d1d0195a117de908

app = Flask(__name__)

# Configurer le logging
logging.basicConfig(level=logging.INFO)

<<<<<<< HEAD
def load_model(path):
    app.logger.info(f"Checking if model file exists at: {path}")
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        app.logger.info(f"Model file found at: {path}")
        model = joblib.load(path)
        
        if hasattr(model, 'predict_proba'):
            app.logger.info("Model loaded successfully and has 'predict_proba' method.")
            return model
        else:
            raise ValueError("Le modèle chargé n'a pas de méthode 'predict_proba'.")
    except Exception as e:
        app.logger.error(f"Error loading model: {e}")
        return None

# Chemin vers le modèle sauvegardé
model_path = './best_gb.pkl'
app.logger.info(f"Loading model from {model_path}")
best_gb = load_model(model_path)
if best_gb is None:
    app.logger.error("Failed to load the model.")

# Définir le seuil optimisé de la régression logistique
THRESHOLD = 0.53

# Charger les données des clients
try:
    app.logger.info("Loading client data from data.csv")
=======
# Charger le modèle de régression logistique depuis le fichier best_lr.pkl
try:
    app.logger.info("Loading model from best_lr.pkl")
    with open('best_lr.pkl', 'rb') as model_file:
        best_lr = pickle.load(model_file)
    app.logger.info("Model loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading model: {e}")
    raise e

# Définir le seuil optimisé de la régression logistique
THRESHOLD = 0.85

# Charger les données des clients
try:
    app.logger.info("Loading client data from cleaned_application_train.csv")
>>>>>>> c8aba6cb345e78fa3356b8e7d1d0195a117de908
    clients_data = pd.read_csv('data.csv')
    app.logger.info("Client data loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading client data: {e}")
    raise e

# Tenter de convertir chaque colonne en float, et ignorer celles qui échouent
numeric_columns = []
for col in clients_data.columns:
    if col != 'SK_ID_CURR':  # Ne pas essayer de convertir l'ID du client
        try:
            clients_data[col] = clients_data[col].astype(float)
            numeric_columns.append(col)
        except ValueError:
            app.logger.warning(f"Column {col} cannot be converted to float and will be ignored.")

@app.route('/')
def home():
    return "API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        app.logger.info("Received a prediction request")
        data = request.get_json(force=True)
        app.logger.info(f"Request data: {data}")
        client_id = data['client_id']
        app.logger.info(f"Predicting for client ID: {client_id}")
        
        # Récupérer les caractéristiques numériques du client à partir des données chargées
        client_data = clients_data[clients_data['SK_ID_CURR'] == client_id][numeric_columns]
        
        if client_data.empty:
            app.logger.error(f"Client ID {client_id} not found")
            return jsonify({"error": f"Client ID {client_id} not found"}), 404
        
        features = client_data.values.reshape(1, -1)
        app.logger.info(f"Features for client ID {client_id}: {features}")
        
        # Vérifier que le nombre de caractéristiques est correct
<<<<<<< HEAD
=======
        # Ici nous vérifions uniquement si nous avons au moins une caractéristique pour la prédiction
>>>>>>> c8aba6cb345e78fa3356b8e7d1d0195a117de908
        if features.shape[1] == 0:
            app.logger.error("No valid features available for prediction.")
            return jsonify({"error": "No valid features available for prediction."}), 400
        
<<<<<<< HEAD
        if best_gb is None:
            app.logger.error("Model is not loaded properly.")
            return jsonify({"error": "Model is not loaded properly."}), 500
        
        # Calculer la probabilité de défaut
        proba = best_gb.predict_proba(features)[0][1]
=======
        # Calculer la probabilité de défaut
        proba = best_lr.predict_proba(features)[0][1]
>>>>>>> c8aba6cb345e78fa3356b8e7d1d0195a117de908
        app.logger.info(f"Probability for client ID {client_id}: {proba}")
        
        # Déterminer la classe en fonction du seuil
        if proba >= THRESHOLD:
            prediction = 'refused'
        else:
            prediction = 'accepted'
        
        # Retourner la probabilité et la classe
        app.logger.info(f"Prediction for client ID {client_id}: {prediction}")
        return jsonify({
            'probability': proba,
            'class': prediction
        })
    except Exception as e:
        app.logger.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
