"""
Ce fichier a pour vocation d'être, en attendant mieux, l'entry point de CF pour faire tourner le "back" (la partie logique)
"""
import json
import logging
import os

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    """Route pour servir la page principale du jeu"""
    return render_template('index.html')

@app.route("/api/get_available_models", methods=['GET'])
def get_available_models():
    from back.data_interfaces.storage import StorageClient
    return StorageClient.get_available_models()


@app.route("/api/generate_name", methods=['POST'])
def generate_name():
    from back.markov.markov_model import MarkovModel
    # read payload
    data = request.get_json()
    model_key = data.get('model_key', None)
    number_names = data.get('number_names', 1)

    model = MarkovModel.from_model_key(model_key)
    return model.generate_names(
        number_names
    )


@app.route("/api/get_round", methods=['POST'])
def get_round():
    from back.game import play_round
    data = request.get_json()
    game_seed = data.get('game_seed', None)
    round_ix = data.get('round_ix', None)
    good_coords, fake_names, bad_coords = play_round(round_ix, game_seed)
    output_payload = {"good_coords": good_coords, "fake_names": fake_names, "bad_coords": bad_coords}
    logging.info(f"Seed {game_seed}, round {round_ix}: "
                 f"{json.dumps(output_payload, ensure_ascii=False, sort_keys=True)}")
    return output_payload


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
