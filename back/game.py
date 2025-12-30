import random
from typing import TYPE_CHECKING

from shapely.geometry.point import Point

from back.data_interfaces.public import PublicData
from back.data_interfaces.storage import StorageClient
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
        alternative_coords = []
        while len(alternative_coords) < nb_alternatives:
            lat = random.uniform(min_lat, max_lat)
            long = random.uniform(min_long, max_long)
            if Point((long, lat)).within(mask_shape):
                alternative_coords.append((lat, long))
        ok_alternative_set = True
        for a, coord_a in enumerate(alternative_coords + [coords]):
            for b, coord_b in enumerate(alternative_coords + [coords]):
                if a < b and coords_dist(coord_a, coord_b) < km_min:
                    ok_alternative_set = False
                    break
            if not ok_alternative_set:
                break
        if ok_alternative_set:
            return alternative_coords
    raise ValueError(
        f"No valid alternative coords found for coords {coords}, nb_alternatives {nb_alternatives}, and km_min {km_min}.")


def play_round(round_ix: int,
               game_seed: str,
               nb_names: int = 7) -> tuple[tuple[float, float], list[str], list[tuple[float, float]]]:
    for i in range(100):
        # pick a random model
        names_by_model_key = StorageClient.get_pre_generated_names_by_model()
        model_keys = list(names_by_model_key.keys())
        random.seed(f"{game_seed}_{round_ix}_{i}")
        model_key = random.choice(model_keys)

        # get names
        names = sorted(names_by_model_key[model_key]["names"])
        random.shuffle(names)
        names = names[:nb_names]

        # generate alternative coords
        center_coords = names_by_model_key[model_key]["coords"]
        nb_alternatives = min(6, round_ix // 3 + 1)
        km_min = max(75, 500 - round_ix * 25)
        try:
            alternative_coords = get_alternative_coords(center_coords, nb_alternatives, km_min,
                                                        PublicData.get_france_shape(), f"{game_seed}_{round_ix}")
        except ValueError as e:
            print(e)
            continue
        else:
            return center_coords, names, alternative_coords
    raise ValueError(f"No valid round found for round_ix {round_ix} and game_seed {game_seed}.")
