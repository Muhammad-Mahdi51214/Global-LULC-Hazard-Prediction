import os
import requests
from dotenv import load_dotenv

load_dotenv()

CDSE_CLIENT_ID = os.getenv("CDSE_CLIENT_ID")
CDSE_CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET")
CDSE_AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STAC_URL = "https://stac.dataspace.copernicus.eu/v1"

def get_cdse_token():
    data = {
        "client_id": CDSE_CLIENT_ID,
        "client_secret": CDSE_CLIENT_SECRET,
        "grant_type": "client_credentials",
    }
    resp = requests.post(CDSE_AUTH_URL, data=data)
    resp.raise_for_status()
    return resp.json()["access_token"]

def stac_search_sentinel2(bbox, start_date, end_date, limit=1):
    """
    bbox: [minLon, minLat, maxLon, maxLat]
    Returns STAC items (simplified).
    """
    token = get_cdse_token()
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "collections": ["SENTINEL-2"],
        "bbox": bbox,
        "datetime": f"{start_date}/{end_date}",
        "limit": limit
    }
    resp = requests.post(f"{STAC_URL}/search", json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data.get("features", [])

def download_asset(asset_href, out_path):
    token = get_cdse_token()
    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(asset_href, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return out_path