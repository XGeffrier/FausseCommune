import json
import logging
import os.path
import time
from pathlib import Path

from geopy.distance import geodesic
import pandas as pd
import numpy as np
from unidecode import unidecode
import urllib.request

LENGTH_MIN = 4
LENGTH_MAX = 40
DISTANCE_POWER = 1.8


class MarkovModel:
    END_TOKEN = '\n'

    def __init__(self,
                 coords: tuple[float, float],
                 order: int,
                 length_min: int = LENGTH_MIN,
                 length_max: int = LENGTH_MAX,
                 distance_power: float = DISTANCE_POWER):
        """
        Coords are used to define the center of the model from where it should generate names.
        Coords: (latitude, longitude)

        Order is number of chars taken into account to generate new char.

        Length min/max are constraints on number of chars of generated names.

        Distance power (p) is used to put a weight 1/(d^p).
        At 0, distance not taken into account, then the more it grows, the more distance is important.
        """

        # model params
        self.coords = coords
        self.order = order
        self.length_min = length_min
        self.length_max = length_max
        self.distance_power = distance_power

        # data
        self._data = None

        # trained model
        self.trained = False
        self._all_init_nuples: list[str] = []
        self._all_init_coeffs: list[float] = []
        self._all_tokens: list[str] = []
        self._model_matrix: dict[str, list[float]] = {}

    def save(self, dir_path):
        matrix_values = np.array(list(self._model_matrix.values()))
        init_coeffs = np.array(self._all_init_coeffs)
        other_data = {
            "coords": self.coords,
            "order": self.order,
            "length_min": self.length_min,
            "length_max": self.length_max,
            "distance_power": self.distance_power,
            "trained": self.trained,
            "init_nuples": self._all_init_nuples,
            "all_tokens": self._all_tokens,
            "matrix_nuples": list(self._model_matrix.keys())
        }
        Path(dir_path).mkdir(parents=True, exist_ok=True)

        numpy_file_path = os.path.join(dir_path, "coeffs.npy")
        data_file_path = os.path.join(dir_path, "data.json")
        with open(numpy_file_path, 'wb') as f:
            np.save(f, matrix_values)
            np.save(f, init_coeffs)
        with open(data_file_path, "w") as f:
            json.dump(other_data, f)

    @classmethod
    def load(self, dir_path) -> "MarkovModel":
        numpy_file_path = os.path.join(dir_path, "coeffs.npy")
        data_file_path = os.path.join(dir_path, "data.json")
        if not (os.path.isfile(numpy_file_path) and os.path.isfile(data_file_path)):
            raise ValueError("Unable to find saved data.")

        with open(numpy_file_path, "rb") as f:
            matrix_values = np.load(f)
            init_coeffs = np.load(f)
        with open(data_file_path) as f:
            data = json.load(f)
        coords = data["coords"]
        order = data["order"]
        length_min = data["length_min"]
        length_max = data["length_max"]
        distance_power = data["distance_power"]
        trained = data["trained"]
        init_nuples = data["init_nuples"]
        all_tokens = data["all_tokens"]
        matrix_nuples = data["matrix_nuples"]

        model = MarkovModel(coords, order, length_min, length_max, distance_power)
        model.trained = trained
        model._all_init_nuples = init_nuples
        model._all_init_coeffs = init_coeffs
        model._model_matrix = {nuple: matrix_values[i] for i, nuple in enumerate(matrix_nuples)}
        model._all_tokens = all_tokens

        return model

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = MarkovModel.fetch_data()
        return self._data

    @classmethod
    def fetch_data(cls) -> pd.DataFrame:
        path = "https://www.data.gouv.fr/fr/datasets/r/dbe8a621-a9c4-4bc3-9cae-be1699c5ff25"
        df = pd.read_csv(path)
        df = df[["nom_commune_complet", "latitude", "longitude"]]
        df["nom_commune_clean"] = df["nom_commune_complet"].apply(lambda s: MarkovModel._clean_input_name(s))
        return df.dropna()

    @classmethod
    def mix_models(cls, models: list["MarkovModel"],
                   coords: tuple[float, float],
                   nb_considered: int = 4) -> "MarkovModel":
        """
        Mix models, pre-trained from distinct coords, to create an already trained model.
        Ponderate each model depending on its distance to desired coords
        """
        dists = [coords_dist(model.coords, coords) for model in models]
        close_models = sorted(models, key=lambda m: dists[models.index(m)])[:nb_considered]
        print([m.coords for m in close_models])
        dists = sorted(dists)[:nb_considered]
        m = close_models[0]

        mix = MarkovModel(coords, m.order, m.length_min, m.length_max, m.distance_power)
        mix._data = m._data
        mix._all_init_nuples = m._all_init_nuples
        mix._all_tokens = m._all_tokens
        mix._all_init_coeffs = np.array([sum(model._all_init_coeffs[i] / (dists[m] ** mix.distance_power)
                                             for m, model in enumerate(close_models))
                                         for i in range(len(m._all_init_coeffs))])
        mix._all_init_coeffs /= mix._all_init_coeffs.sum()
        mix._model_matrix = {key: np.array([sum(model._model_matrix[key][i] / (dists[m] ** mix.distance_power)
                                                for m, model in enumerate(close_models))
                                            for i in range(len(m._model_matrix[key]))])
                             for key in m._model_matrix.keys()}
        mix._model_matrix = {key: value / value.sum()
                             for key, value in mix._model_matrix.items()}
        mix.trained = True
        return mix

    def train(self) -> None:
        """
        Train the model.
        Usually take a few seconds. Once the model is trained, it can generate lot of names very fast.
        """
        # build raw matrices (raw = they contain all transitions with their distance)
        model_raw_matrix = {}
        model_raw_init = {}
        for i, row in self.data.iterrows():
            # get commune infos
            name, lat, long = row['nom_commune_clean'], row['latitude'], row['longitude']
            distance = coords_dist(self.coords, (lat, long))

            # add each nuple -> char transition in commune name with distance
            first_nuple = name[0:self.order]
            model_raw_init.setdefault(first_nuple, []).append(distance)
            for j in range(len(name) - self.order):
                nuple = name[j:j + self.order]
                next_token = name[j + self.order]
                model_raw_matrix.setdefault(nuple, {}).setdefault(next_token, []).append(distance)
            last_nuple = name[-self.order:]
            model_raw_matrix.setdefault(last_nuple, {}).setdefault(self.END_TOKEN, []).append(distance)

        # build final matrices (ponderated averages, depending on distances)
        model_matrix = {}
        all_tokens = sorted({token for nexts in model_raw_matrix.values() for token in nexts.keys()})
        for nuple, nexts in model_raw_matrix.items():
            avg = {token: sum([1 / (d ** self.distance_power) for d in distances])
                   for token, distances in nexts.items()}
            coeffs = np.array([avg.get(token, 0) for token in all_tokens])
            model_matrix[nuple] = coeffs / coeffs.sum()
        model_init = {nuple: sum([1 / (d ** self.distance_power) for d in distances])
                      for nuple, distances in model_raw_init.items()}
        all_init_nuples = list(model_init.keys())
        all_init_coeffs = np.array(list(model_init.values()))
        all_init_coeffs /= all_init_coeffs.sum()

        # store model
        self._all_init_nuples = all_init_nuples
        self._all_init_coeffs = all_init_coeffs
        self._all_tokens = all_tokens
        self._model_matrix = model_matrix
        self.trained = True

    def generate_names(self, number_names: int = 5) -> list[str]:
        """
        Main function to generate names. Train the model if it is not already.
        """
        if not self.trained:
            self.train()

        names = []
        while len(names) < number_names:
            # generate name
            logging.info("Generating name")
            name = np.random.choice(self._all_init_nuples, p=self._all_init_coeffs)
            while name[-1] != self.END_TOKEN:
                token = np.random.choice(self._all_tokens, p=self._model_matrix[name[-self.order:]])
                name = name + token
            name = name[:-1]

            # check
            if self._is_generated_name_valid(name):
                nice = self._nicer_output_name(name)
                if nice not in names:
                    logging.info("Name is valid and new")
                    names.append(nice)

        return names

    def _is_generated_name_valid(self, name: str) -> bool:
        """
        Perform several checks on a generated name.
        Since generation is really fast (once the training is done), we want to generate a lot of names and filter the
        bad ones, wich is much easier and faster than generate only good ones in the first place.
        """

        # check length
        if not (LENGTH_MIN <= len(name) <= LENGTH_MAX):
            return False

        # check not existing
        if name in self.data["nom_commune_clean"].values:
            return False

        words = name.split()

        # check articles in a row
        name.replace("de le", "du")
        articles = {"le", "la", "les", "du", "de", "des", "l", "d"}
        for i, word in enumerate(words):
            if word in articles and (i == len(words) - 1 or (words[i + 1] in articles
                                                             and not (word == "de" and words[i + 1] == "la"))):
                return False

        # check prepositions in a row
        prepositions = {"en", "dans", "sur", "sous", "aux", "a"}
        for i, word in enumerate(words):
            if word in prepositions and (i == len(words) - 1 or words[i + 1] in prepositions):
                return False

        return True

    @staticmethod
    def _clean_input_name(name: str) -> str:
        """
        Remove diacritics and punctuation and apply lower case.
        """
        name = unidecode(name).lower()
        name = ''.join(c if c.isalpha() else ' ' for c in name)
        return name

    @staticmethod
    def _nicer_output_name(name: str) -> str:
        """
        Improve the output name to make it more realistic for France.
        """
        LOWERCASE_WORDS = ('a', 'au', 'aux', 'd', 'de', 'del', 'dels', 'derriere', 'des', 'deux', 'devant', 'di', 'dit',
                           'du', 'en', 'entre', 'environs', 'es', 'es', 'et', 'ez', 'h', 'huis', 'l', 'la', 'las', 'le',
                           'les', 'les', 'lez', 'nouvelle', 'o', 'plages', 'pres', 'sans', 'ses', 'sous', 'sur')
        words = name.split()

        # la ferté sur loing -> La Ferté sur Loing
        words = [word.title()
                 if (len(word) > 2 and word not in LOWERCASE_WORDS) or i == 0
                 else word
                 for i, word in enumerate(words)]

        # les -> lès
        if "les" in words:
            words = words[0:1] + list(map(lambda x: x.replace('les', 'lès'), words[1:]))

        # Saint Martin l Abbaye -> Saint Martin l'Abbaye
        to_merge = [ix for ix, word in enumerate(words) if len(word) == 1 and ix < len(words) - 1]
        for ix in to_merge[::-1]:
            words[ix] = "'".join((words[ix], words[ix + 1]))
            del words[ix + 1]

        # Saint Michel sur Garonne -> Saint-Michel-sur-Garonne
        name = '-'.join(words)

        return name


def coords_dist(coords_a: tuple[float, float], coords_b: tuple[float, float]) -> float:
    """
    Compute geodesic distance between (latitude, longitude) coordinates.
    """
    return geodesic(coords_a, coords_b).km


def generate_grid_coords(size_grid: int = 50) -> list[tuple[float, float]]:
    """
    Return (latitude, longitude) coords of points evenly distributed in France.
    """
    from shapely.geometry import shape, Point
    from shapely.ops import unary_union
    regions_url = "https://france-geojson.gregoiredavid.fr/repo/regions.geojson"
    regions_fp = "regions.json"
    urllib.request.urlretrieve(regions_url, regions_fp)
    with open(regions_fp) as f:
        regions_data = json.load(f)
    france_shape = unary_union([shape(feature['geometry'])
                                for feature in regions_data['features']
                                if feature["properties"]["code"][0] != "0"])
    min_long, min_lat, max_long, max_lat = france_shape.bounds
    lats = np.linspace(min_lat, max_lat, size_grid)
    longs = np.linspace(min_long, max_long, size_grid)
    lats, longs = np.meshgrid(lats, longs)
    grid_coords = np.array([lats.flatten(), longs.flatten()]).T
    # we reverse coords to test them in France: (lat, long) is the norm, but it is equivalent to (y, x)
    french_coords = [tuple(c) for c in grid_coords if Point((c[1], c[0])).within(france_shape)]
    return french_coords


def create_and_train_all_models(size_grid: int,
                                order: int = 3,
                                length_min: int = LENGTH_MIN,
                                length_max: int = LENGTH_MAX,
                                distance_power: float = DISTANCE_POWER,
                                models_dir_path: str = "models") -> list[MarkovModel]:
    """
    Find coords meshing France, then compute model for each pair of coords.
    """
    # try to load models
    sub_dir_name = f"models_{size_grid}_{order}_{length_min}_{length_max}_{distance_power}"
    sub_dir_path = os.path.join(models_dir_path, sub_dir_name)
    if os.path.isdir(sub_dir_path):
        models = []
        for elt in os.listdir(sub_dir_path):
            if elt.startswith("model"):
                elt_path = os.path.join(sub_dir_path, elt)
                if os.path.isdir(elt_path):
                    models.append(MarkovModel.load(elt_path))
        return models

    # find centers
    logging.debug(f"Checking if {size_grid}^2 = {size_grid ** 2} points are in France...")
    all_coords = generate_grid_coords(size_grid)
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


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
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
    start = time.time()

    models = create_and_train_all_models(30)
    models_trained = time.time()

    end_model = MarkovModel.mix_models(models, COORDS_PARIS, 4)
    models_mixed = time.time()

    names = end_model.generate_names(50)
    names_generated = time.time()

    print(*names, sep='\n')

    print(models_trained - start, "to train all models,\n",
          models_mixed - models_trained, "to mix all models,\n",
          names_generated - models_mixed, "to generate names,\n")
