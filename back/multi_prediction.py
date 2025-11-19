import logging
import os
from typing import Optional

from back.external import load_france_shape
from back.markov_model import MarkovModel, LENGTH_MIN, LENGTH_MAX, DISTANCE_POWER, MixedModels
from back.math_utils import generate_grid_coords


def load_models_from_filesystem(sub_dir_path: str) -> Optional[list["MarkovModel"]]:
    """
    Return None if not found.
    """
    if os.path.isdir(sub_dir_path):
        models = []
        for elt in os.listdir(sub_dir_path):
            if elt.startswith("model"):
                elt_path = os.path.join(sub_dir_path, elt)
                if os.path.isdir(elt_path):
                    models.append(MarkovModel.load_from_local_filesystem(elt_path))
        return models
    return None


def find_or_create_all_models(size_grid: int,
                              order: int = 3,
                              length_min: int = LENGTH_MIN,
                              length_max: int = LENGTH_MAX,
                              distance_power: float = DISTANCE_POWER,
                              models_dir_path: str = "models") -> list["MarkovModel"]:
    """
    Find coords meshing France, then compute model for each pair of coords.
    """
    # try to load models
    sub_dir_name = f"models_{size_grid}_{order}_{length_min}_{length_max}_{distance_power}"
    sub_dir_path = os.path.join(models_dir_path, sub_dir_name)
    models = load_models_from_filesystem(sub_dir_path)

    if models is None:
        # find centers
        logging.debug(f"Checking if {size_grid}^2 = {size_grid ** 2} points are in France...")
        france_shape = load_france_shape()
        all_coords = generate_grid_coords(france_shape, size_grid)
        logging.debug("Coords of the models to train: ")
        logging.debug(str(all_coords))

        # create models
        logging.debug(f"{len(all_coords)} points found. Creating models...")
        models = [MarkovModel(coords, order, length_min, length_max, distance_power)
                  for coords in all_coords]

        # train models
        logging.debug(f"Models created. Training...")
        for i, model in enumerate(models):
            logging.debug(f"{i + 1}/{len(models)}")
            model.train()

        # save models
        for i, model in enumerate(models):
            model.save(os.path.join(sub_dir_path, f"model_{i}"))
    return models


COORDS_BUISSON = 47.238766631730776, 3.110139518759569
COORDS_PARIS = 48.85508109030361, 2.34699411756656
COORDS_STRASBOURG = 48.58108424658188, 7.745253905112268
COORDS_SUD_OUEST = 43.687335933217035, 0.2663092740410821
COORDS_BRETAGNE = 48.62180709508752, -3.725627970256062

COORDS_RITA = 45.546343905837986, -1.064130968218902
COORDS_CALIXTE = 45.0630641, 1.0970657
COORDS_DUROC = 48.84693116822202, 2.316860216366209
COORDS_PANTIN = 48.894351122965475, 2.410677105637959
COORDS_ALIX = 47.6254407, -2.7748677
COORDS_SAUMUR = 47.26023, -0.07543

COORDS_SAM = 47.9084244968888, 4.11970654513218
COORDS_BARCELONE = 41.48932484100134, 2.202460701405498
COORDS_SPDO = 48.306808437385754, 0.4232448707091479
COORDS_PAYS_BASQUE = 43.195420950963474, -1.1718909484033513
COORDS_ARGONNE = 49.13803311676605, 4.951462302602254

if __name__ == '__main__':
    import time

    logging.getLogger().setLevel(logging.INFO)
    start = time.time()

    models = find_or_create_all_models(30)
    models_trained = time.time()

    end_model = MixedModels(models, COORDS_PARIS)
    models_mixed = time.time()

    names = end_model.generate_names(50)
    names_generated = time.time()

    print(*names, sep='\n')

    print(models_trained - start, "to train all models,\n",
          models_mixed - models_trained, "to mix all models,\n",
          names_generated - models_mixed, "to generate names,\n")
