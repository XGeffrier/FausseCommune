import json

from typing import TYPE_CHECKING

from google.cloud import storage

from back import auth_data

if TYPE_CHECKING:
    import shapely.geometry
    import pandas as pd

COMMUNES_URL = "https://www.data.gouv.fr/fr/datasets/r/dbe8a621-a9c4-4bc3-9cae-be1699c5ff25"
REGIONS_URL = "https://france-geojson.gregoiredavid.fr/repo/regions.geojson"


class StorageClient:
    _storage_client = None

    @classmethod
    def storage_client(cls) -> storage.Client:
        if cls._storage_client is None:
            cls._storage_client = storage.Client(credentials=auth_data.gcp_key)
        return cls._storage_client

    @classmethod
    def upload_file_in_blob(cls, bucket_name: str, local_file_path: str, destination_blob_name):
        """Uploads a file to the bucket."""
        storage_client = cls.storage_client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)

    @classmethod
    def download_blob_str(cls, bucket_name: str, blob_name: str) -> str:
        """Downloads a blob into memory."""
        storage_client = cls.storage_client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_string()


def load_france_shape() -> "shapely.geometry.shape":
    import urllib.request
    from shapely.ops import unary_union
    from shapely.geometry import shape
    regions_fp = "regions.json"
    urllib.request.urlretrieve(REGIONS_URL, regions_fp)
    with open(regions_fp) as f:
        regions_data = json.load(f)
    france_shape = unary_union([shape(feature['geometry'])
                                for feature in regions_data['features']
                                if feature["properties"]["code"][0] != "0"])
    return france_shape


def fetch_communes_data() -> "pd.DataFrame":
    import pandas as pd
    df = pd.read_csv(COMMUNES_URL)
    df = df[["nom_commune_complet", "latitude", "longitude"]]
    return df.dropna()

if __name__ == '__main__':
    StorageClient.upload_file_in_blob("myBucket",
                                      r"C:\Users\Xavier Geffrier\PycharmProjects\flaskProject\back\external.py",
                                      "blob_name")