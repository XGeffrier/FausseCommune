import random
from typing import TYPE_CHECKING

from shapely.geometry.point import Point

from back.data_interfaces.public import PublicData
from back.data_interfaces.storage import StorageClient
from back.markov.markov_model import MarkovModel
from back.markov.math_utils import coords_dist

if TYPE_CHECKING:
    import shapely.geometry


def get_alternative_coords(coords: tuple[float, float],
                           nb_alternatives: int,
                           km_min: int,
                           mask_shape: "shapely.geometry.shape",
                           round_seed: str) -> list[tuple[float, float]]:
    random.seed(round_seed)
    min_long, min_lat, max_long, max_lat = mask_shape.bounds
    for i in range(100):
        alternative_coords = [coords]
        while len(alternative_coords) < nb_alternatives + 1:
            lat = random.uniform(min_lat, max_lat)
            long = random.uniform(min_long, max_long)
            if Point((long, lat)).within(mask_shape):
                alternative_coords.append((lat, long))
        ok_alternative = True
        for a, coord_a in enumerate(alternative_coords):
            for b, coord_b in enumerate(alternative_coords):
                if coords_dist(coord_a, coord_b) < km_min:
                    ok_alternative = True
                    break
            if not ok_alternative:
                break
        if ok_alternative:
            return alternative_coords
    raise ValueError("No valid alternative coords found")


def play_round(round_ix: int,
               game_seed: str,
               nb_names: int = 7) -> tuple[
    tuple[float, float], list[str], list[tuple[float, float]]]:
    random.seed(game_seed)
    available_models = StorageClient.get_available_models()
    model_keys = list(available_models.keys())
    random.shuffle(model_keys)
    model_key = model_keys[round_ix]
    model = MarkovModel.from_model_key(model_key)
    nb_alternatives = round_ix // 3 + 1
    km_min = max(75, 500 - round_ix * 25)
    alternative_coords = get_alternative_coords(model.center_coords, nb_alternatives, km_min,
                                                PublicData.get_france_shape(), f"{game_seed}_{round_ix}")
    return model.center_coords, model.generate_names(nb_names), alternative_coords


if __name__ == '__main__':
    game_seed = "42"
    for round_ix in range(10):
        print(play_round(round_ix, game_seed))
