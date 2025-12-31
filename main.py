"""
Ce fichier a pour vocation d'être, en attendant mieux, l'entry point de CF pour faire tourner le "back" (la partie logique)
"""
import json
import logging
import os

from flask import Flask, render_template, request

if os.getenv("GCP_PROJECT"):
    import google.cloud.logging

    client = google.cloud.logging.Client()
    client.get_default_handler()
    client.setup_logging()
else:
    logging.getLogger().setLevel(logging.DEBUG)
app = Flask(__name__)


@app.route('/')
def index():
    """Route pour servir la page principale du jeu"""
    return render_template('index.html')


@app.route('/robots.txt')
def robots():
    """Route pour servir le fichier robots.txt"""
    return "User-agent: *\nAllow: /", 200, {'Content-Type': 'text/plain; charset=utf-8'}


@app.route("/api/get_round", methods=['POST'])
def get_round():
    from back.game import play_round
    data = request.get_json()
    game_seed = data.get('game_seed', None)
    round_ix = data.get('round_ix', None)
    logging.info(f"Get round request: seed={game_seed}, round={round_ix}, ip={request.remote_addr}")
    good_coords, fake_names, bad_coords = play_round(round_ix, game_seed)
    output_payload = {"good_coords": good_coords, "fake_names": fake_names, "bad_coords": bad_coords}
    logging.info(f"Seed {game_seed}, round {round_ix}: "
                 f"{json.dumps(output_payload, ensure_ascii=False, sort_keys=True)}")
    return output_payload


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
