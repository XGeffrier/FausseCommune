"""
Ce fichier a pour vocation d'être, en attendant mieux, l'entry point de CF pour faire tourner le "back" (la partie logique)
"""

import os
import time

from flask import Flask, request

from back.data_interfaces.storage import StorageClient
from back.markov.multi_prediction import generate_name

app = Flask(__name__)


@app.route("/")
def hello_world():
    """Example Hello World route."""
    name = os.environ.get("NAME", "World")
    return f"Hello {name}!"


@app.route("/api/get_available_models")
def get_available_models():
    return StorageClient.download_json_file_as_dict(StorageClient.models_json_path) or {}


@app.route("/api/generate_name", methods=['POST'])
def generate_name():
    data = request.get_json()
    model_key = data.get('model_key', None)
    return generate_name(lat, long, model_key)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
