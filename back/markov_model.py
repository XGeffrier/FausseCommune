import json
import logging
import os.path
from pathlib import Path

import numpy as np
import pandas as pd
from unidecode import unidecode

from back.external import fetch_communes_data
from back.math_utils import coords_dist

LENGTH_MIN = 4
LENGTH_MAX = 40
DISTANCE_POWER = 1.8


class MarkovModel:
    END_TOKEN = '\n'
    _communes_data = None

    def __init__(self,
                 center_coords: tuple[float, float],
                 markov_order: int,
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
        self.center_coords = center_coords
        self.markov_order = markov_order
        self.length_min = length_min
        self.length_max = length_max
        self.distance_power = distance_power

        # trained model
        self.trained = False
        self._all_init_nuples: list[str] = []
        self._all_init_coeffs: list[float] = []
        self._all_tokens: list[str] = []
        self._model_matrix: dict[str, list[float]] = {}

    @classmethod
    def communes_data(cls) -> pd.DataFrame:
        if cls._communes_data is None:
            cls._communes_data = fetch_communes_data()
            cls._communes_data["nom_commune_clean"] = cls._communes_data["nom_commune_complet"].apply(
                lambda s: MarkovModel._clean_input_name(s))
        return cls._communes_data

    def train(self) -> None:
        """
        Train the model.
        Usually take a few seconds. Once the model is trained, it can generate lot of names very fast.
        """
        # build raw matrices (raw = they contain all transitions with their distance)
        model_raw_matrix = {}
        model_raw_init = {}
        for i, row in self.communes_data().iterrows():
            # get commune infos
            name, lat, long = row['nom_commune_clean'], row['latitude'], row['longitude']
            distance = coords_dist(self.center_coords, (lat, long))

            # add each nuple -> char transition in commune name with distance
            first_nuple = name[0:self.markov_order]
            model_raw_init.setdefault(first_nuple, []).append(distance)
            for j in range(len(name) - self.markov_order):
                nuple = name[j:j + self.markov_order]
                next_token = name[j + self.markov_order]
                model_raw_matrix.setdefault(nuple, {}).setdefault(next_token, []).append(distance)
            last_nuple = name[-self.markov_order:]
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
                token = np.random.choice(self._all_tokens, p=self._model_matrix[name[-self.markov_order:]])
                name = name + token
            name = name[:-1]

            # check
            if self._is_generated_name_valid(name):
                nice = self._nicer_output_name(name)
                if nice not in names:
                    logging.info("Name is valid and new")
                    names.append(nice)

        return names

    def save(self, dir_path):
        matrix_values = np.array(list(self._model_matrix.values()))
        init_coeffs = np.array(self._all_init_coeffs)
        other_data = {
            "coords": self.center_coords,
            "order": self.markov_order,
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
    def load_from_local_filesystem(self, dir_path) -> "MarkovModel":
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

    @classmethod
    def _is_generated_name_valid(cls, name: str) -> bool:
        """
        Perform several checks on a generated name.
        Since generation is really fast (once the training is done), we want to generate a lot of names and filter the
        bad ones, wich is much easier and faster than generate only good ones in the first place.
        """

        # check length
        if not (LENGTH_MIN <= len(name) <= LENGTH_MAX):
            return False

        # check not existing
        if name in cls.communes_data()["nom_commune_clean"].values:
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


class MixedModels(MarkovModel):

    def __init__(self,
                 models: list["MarkovModel"],
                 coords: tuple[float, float],
                 nb_considered: int = 4):
        """
        Mix models, pre-trained from distinct coords, to create an already trained model.
        Ponderate each model depending on its distance to desired coords
        """
        super().__init__(coords, models[0].markov_order, models[0].length_min, models[0].length_max, models[0].distance_power)

        # note: this could be optimized
        dists = [coords_dist(model.center_coords, coords) for model in models]
        close_models = sorted(models, key=lambda m: dists[models.index(m)])[:nb_considered]
        dists = sorted(dists)[:nb_considered]
        m = close_models[0]

        self._all_init_nuples = m._all_init_nuples
        self._all_tokens = m._all_tokens
        self._all_init_coeffs = np.array([sum(model._all_init_coeffs[i] / (dists[m] ** self.distance_power)
                                              for m, model in enumerate(close_models))
                                          for i in range(len(m._all_init_coeffs))])
        self._all_init_coeffs /= self._all_init_coeffs.sum()
        self._model_matrix = {key: np.array([sum(model._model_matrix[key][i] / (dists[m] ** self.distance_power)
                                                 for m, model in enumerate(close_models))
                                             for i in range(len(m._model_matrix[key]))])
                              for key in m._model_matrix.keys()}
        self._model_matrix = {key: value / value.sum()
                              for key, value in self._model_matrix.items()}
        self.trained = True
