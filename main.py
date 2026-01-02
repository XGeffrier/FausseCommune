"""
Ce fichier a pour vocation d'être, en attendant mieux, l'entry point de FC pour faire tourner le "back" (la partie logique)
"""
import base64
import json
import logging
import os
import random
import uuid
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from flask import Flask, abort, render_template, request

if os.getenv("GCP_PROJECT"):
    import google.cloud.logging

    client = google.cloud.logging.Client()
    client.get_default_handler()
    client.setup_logging()
else:
    logging.getLogger().setLevel(logging.DEBUG)
app = Flask(__name__)
app.config.setdefault("SECRET_KEY", os.environ.get("SECRET_KEY", "dev-secret"))

_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    """
    Lazily instantiate the Fernet helper from the SECRET_KEY.
    """
    global _fernet
    if _fernet is None:
        import hashlib

        secret = app.config.get("SECRET_KEY") or "dev-secret"
        if secret == "dev-secret":
            logging.warning("SECRET_KEY not set; using insecure development key.")
        digest = hashlib.sha256(secret.encode("utf-8")).digest()
        key = base64.urlsafe_b64encode(digest)
        _fernet = Fernet(key)
    return _fernet


def _encode_round_token(payload: dict[str, Any]) -> str:
    """
    Encrypt payload to avoid leaking the correct answer to clients.
    """
    raw = json.dumps(payload).encode("utf-8")
    return _get_fernet().encrypt(raw).decode("utf-8")


def _decode_round_token(token: str) -> dict[str, Any]:
    raw = _get_fernet().decrypt(token.encode("utf-8"))
    return json.loads(raw.decode("utf-8"))


def _build_round_payload(game_seed: str, round_ix: int) -> tuple[dict[str, Any], str]:
    """
    Generate round data along with the encrypted token used to validate the answer.
    """
    from back.game import play_round

    good_coords, fake_names, bad_coords = play_round(round_ix, game_seed)
    options = [(True, good_coords)] + [(False, coords) for coords in bad_coords]
    random.shuffle(options)

    round_options = []
    correct_id = None
    for is_correct, coords in options:
        option_id = uuid.uuid4().hex
        round_options.append({"id": option_id, "coords": coords})
        if is_correct:
            correct_id = option_id
    if correct_id is None:
        raise RuntimeError("No correct option generated for round.")

    token = _encode_round_token({
        "round_ix": round_ix,
        "game_seed": game_seed,
        "correct_id": correct_id
    })
    round_payload = {
        "round_ix": round_ix,
        "fake_names": fake_names,
        "options": round_options
    }
    return round_payload, token


@app.route('/')
def index():
    """Route pour servir la page principale du jeu"""
    return render_template('index.html')


@app.route('/robots.txt')
def robots():
    """Route pour servir le fichier robots.txt"""
    return "User-agent: *\nAllow: /", 200, {'Content-Type': 'text/plain; charset=utf-8'}


@app.route("/api/round", methods=['POST'])
def round_endpoint():
    data = request.get_json() or {}
    token = data.get('token')
    guess_id = data.get('guess_id')
    previous_result = None

    if token:
        if not guess_id:
            abort(400, description="Missing guess identifier.")
        try:
            token_payload = _decode_round_token(token)
        except InvalidToken:
            abort(400, description="Invalid token.")
        game_seed = token_payload["game_seed"]
        round_ix = token_payload["round_ix"]
        correct_id = token_payload["correct_id"]
        won = guess_id == correct_id
        previous_result = {
            "round_ix": round_ix,
            "won": won,
            "correct_id": correct_id,
            "guess_id": guess_id
        }
        next_round_ix = round_ix + 1
    else:
        game_seed = data.get('game_seed')
        if not game_seed:
            abort(400, description="Missing game_seed.")
        next_round_ix = 0

    logging.info("Round endpoint: seed=%s, next_round=%s, ip=%s", game_seed, next_round_ix, request.remote_addr)
    round_payload, next_token = _build_round_payload(game_seed, next_round_ix)
    return {
        "round": round_payload,
        "token": next_token,
        "previous_result": previous_result
    }


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
