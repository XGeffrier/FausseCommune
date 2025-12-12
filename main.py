"""
Ce fichier a pour vocation d'être, en attendant mieux, l'entry point de CF pour faire tourner le "back" (la partie logique)
"""

import os

from flask import Flask, request

from back.data_interfaces.storage import StorageClient
from back.markov.markov_model import MarkovModel

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
    # read payload
    data = request.get_json()
    model_key = data.get('model_key', None)
    number_names = data.get('number_names', 1)

    model = MarkovModel.from_model_key(model_key)
    return model.generate_names(
        number_names
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
