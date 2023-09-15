import json

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import shapely.lib
    import pandas as pd

communes_url = "https://www.data.gouv.fr/fr/datasets/r/dbe8a621-a9c4-4bc3-9cae-be1699c5ff25"
regions_url = "https://france-geojson.gregoiredavid.fr/repo/regions.geojson"


def load_france_shape() -> "shapely.lib.Geometry":
    import urllib.request
    from shapely.ops import unary_union
    from shapely.geometry import shape
    regions_fp = "regions.json"
    urllib.request.urlretrieve(regions_url, regions_fp)
    with open(regions_fp) as f:
        regions_data = json.load(f)
    france_shape = unary_union([shape(feature['geometry'])
                                for feature in regions_data['features']
                                if feature["properties"]["code"][0] != "0"])
    return france_shape


def fetch_communes_data() -> "pd.DataFrame":
    import pandas as pd
    df = pd.read_csv(communes_url)
    df = df[["nom_commune_complet", "latitude", "longitude"]]
    return df.dropna()
