import json
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import shapely.geometry
    import pandas as pd


class PublicData:
    COMMUNES_URL = "https://www.data.gouv.fr/fr/datasets/r/dbe8a621-a9c4-4bc3-9cae-be1699c5ff25"
    REGIONS_URL = "https://france-geojson.gregoiredavid.fr/repo/regions.geojson"

    _france_shape = None
    _communes_data = None

    regions_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regions.json")

    @classmethod
    def get_france_shape(cls) -> "shapely.geometry.shape":
        if cls._france_shape is None:
            from shapely.ops import unary_union
            from shapely.geometry import shape
            if os.path.exists(cls.regions_json_path):
                logging.info("Loading France shape from local file")
                with open(cls.regions_json_path) as f:
                    regions_data = json.load(f)
            else:
                import requests
                logging.info("Downloading France shape from remote source")
                regions_data = requests.get(cls.REGIONS_URL).json()
                with open(cls.regions_json_path, "w") as f:
                    json.dump(regions_data, f)
            france_shape = unary_union([shape(feature['geometry'])
                                        for feature in regions_data['features']
                                        if feature["properties"]["code"][0] != "0"])
            cls._france_shape = france_shape
        return cls._france_shape

    @classmethod
    def fetch_communes_data(cls) -> "pd.DataFrame":
        if cls._communes_data is None:
            import pandas as pd
            df = pd.read_csv(cls.COMMUNES_URL)
            df = df[["nom_commune_complet", "latitude", "longitude"]]
            cls._communes_data = df.dropna()
        return cls._communes_data
