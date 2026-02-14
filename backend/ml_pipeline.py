import io
import os
import json
import gzip
import shutil
import tempfile
import datetime as dt
import requests
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from sklearn.ensemble import RandomForestClassifier

# Copernicus CDSE imports disabled - using GEE instead
# try:
#     from .copernicous_client import stac_search_sentinel2, download_asset
# except ImportError:
#     from copernicous_client import stac_search_sentinel2, download_asset

# Load models once
RF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "rf_lulc.pkl")

import joblib

# Try to load the RF model, create a dummy one if it fails
try:
    rf_model: RandomForestClassifier = joblib.load(RF_MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load RF model: {e}. Using dummy classifier.")
    rf_model = None


def geocode_aoi(name: str):
    """
    Use Nominatim to get:
      - bounding box [minLon, minLat, maxLon, maxLat]
      - polygon geometry (GeoJSON, if available)

    This is used as AOI for Sentinel / Landsat / CHIRPS / DEM.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": name,
        "format": "json",
        "limit": 1,
        "polygon_geojson": 1,  # <-- this returns the full polygon
    }
    headers = {"User-Agent": "GlobalEWS/1.0 (Global Early Warning System; https://github.com/global-ews)"}
    resp = requests.get(url, params=params, headers=headers, timeout=20)
    resp.raise_for_status()

    results = resp.json()
    if not results:
        raise ValueError(f"No geocoding result for '{name}'")

    r = results[0]

    # Nominatim bbox = [south, north, west, east]
    south, north, west, east = map(float, r["boundingbox"])
    bbox = [west, south, east, north]

    polygon = r.get("geojson")

    return bbox, polygon


def generate_mock_satellite_data(size=(100, 100)):
    """Generate mock satellite band data for demo purposes."""
    H, W = size
    np.random.seed(42)
    # Generate mock bands (B4-Red, B8-NIR, B3-Green)
    red = np.random.uniform(0.1, 0.5, (H, W)).astype(np.float32)
    nir = np.random.uniform(0.2, 0.8, (H, W)).astype(np.float32)
    green = np.random.uniform(0.1, 0.4, (H, W)).astype(np.float32)
    stack = np.stack([red, nir, green], axis=0)
    return stack


def compute_indices(stack):
    """Compute NDVI, NDWI using band indices."""
    red = stack[0]
    nir = stack[1]
    green = stack[2]
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndwi = (green - nir) / (green + nir + 1e-6)
    return np.stack([ndvi, ndwi], axis=0)


def classify_lulc(stack):
    """Run RF on per-pixel feature vectors or use rule-based classification."""
    indices = compute_indices(stack)
    full_stack = np.concatenate([stack, indices], axis=0)
    B, H, W = full_stack.shape
    
    if rf_model is not None:
        X = full_stack.reshape(B, -1).T
        preds = rf_model.predict(X)
        lulc = preds.reshape(H, W).astype(np.int16)
    else:
        # Rule-based classification for demo
        ndvi = indices[0]
        ndwi = indices[1]
        lulc = np.ones((H, W), dtype=np.int16)
        lulc[ndwi > 0.1] = 1  # Water
        lulc[(ndvi > 0.3) & (ndwi <= 0.1)] = 4  # Forest
        lulc[(ndvi > 0.1) & (ndvi <= 0.3)] = 3  # Agriculture
        lulc[(ndvi <= 0.1) & (ndwi <= 0.1)] = 5  # Bare
        # Randomly assign some urban
        urban_mask = np.random.random((H, W)) < 0.1
        lulc[urban_mask & (lulc == 5)] = 2  # Urban
    
    return lulc


def run_ca_ann_future(lulc_current):
    """Simplified future prediction using probabilistic transition."""
    H, W = lulc_current.shape
    np.random.seed(42)
    noise = np.random.randint(-1, 2, size=(H, W))
    future_classes = np.clip(lulc_current + noise, 1, 5).astype(np.int16)
    return future_classes


def generate_mock_dem(bbox, size=(100, 100)):
    """
    Placeholder for DEM based on USGS/SRTMGL1 (Conceptually equivalent to:
      var dem = ee.Image('USGS/SRTMGL1_003');
      var slope = ee.Terrain.slope(dem);

    For the hackathon prototype, we:
      - create a synthetic DEM grid whose range depends on latitude (rough approx)
      - but keep the size consistent with other arrays.

    Later, you can replace this with a real SRTM download + resample.
    """
    west, south, east, north = bbox
    lat_center = 0.5 * (south + north)

    H, W = size
    # simple gradient + noise: higher elevation in north-east, lower in south-west
    y = np.linspace(0, 1, H).reshape(-1, 1)
    x = np.linspace(0, 1, W).reshape(1, -1)
    base = (x + y) * 1000.0  # up to ~1000 m
    lat_factor = 0.3 + 0.7 * (abs(lat_center) / 90.0)
    dem = base * lat_factor + np.random.normal(0, 20, size=(H, W)).astype(np.float32)
    return dem.astype(np.float32)


def compute_slope_from_dem(dem_array):
    """Compute slope from DEM using numpy gradient."""
    dy, dx = np.gradient(dem_array)
    slope_radians = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_degrees = np.degrees(slope_radians)
    return slope_degrees


# ============== CHIRPS Rainfall Integration ==============

CHIRPS_BASE_DAILY = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05"
CHIRPS_BASE_MONTHLY = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs"


def _download_and_unzip_tif(url: str, tmp_dir: str) -> str:
    """
    Download a .tif or .tif.gz from CHIRPS and return local .tif path.
    """
    local_gz = os.path.join(tmp_dir, os.path.basename(url))
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(local_gz, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    if local_gz.endswith(".gz"):
        local_tif = local_gz[:-3]
        with gzip.open(local_gz, "rb") as f_in, open(local_tif, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(local_gz)
        return local_tif
    else:
        return local_gz


def _clip_raster_to_bbox(src_path: str, bbox, out_shape=None):
    """
    Clip a global raster (like CHIRPS) to a bbox [minLon, minLat, maxLon, maxLat].
    Optionally resample to out_shape (H, W).
    """
    west, south, east, north = bbox
    with rasterio.open(src_path) as src:
        window = from_bounds(west, south, east, north, src.transform)
        data = src.read(1, window=window, boundless=True, fill_value=np.nan)
        transform = src.window_transform(window)

        if out_shape is not None:
            from rasterio.enums import Resampling
            H, W = out_shape
            data_resampled = np.empty((H, W), dtype=np.float32)
            rasterio.warp.reproject(
                data,
                data_resampled,
                src_transform=transform,
                src_crs=src.crs,
                dst_transform=rasterio.transform.from_bounds(west, south, east, north, W, H),
                dst_crs=src.crs,
                resampling=Resampling.bilinear,
            )
            return data_resampled.astype(np.float32)
        else:
            return data.astype(np.float32)


def get_chirps_monthly_rainfall(bbox, start_date: str, end_date: str, out_shape=None):
    """
    Download CHIRPS monthly rainfall between start_date and end_date,
    clip to bbox, and sum (mm) over the period.

    Uses pattern:
      {CHIRPS_BASE_MONTHLY}/tifs/chirps-v2.0.YYYY.MM.tif.gz
    """
    start = dt.datetime.fromisoformat(start_date)
    end = dt.datetime.fromisoformat(end_date)

    # Build list of months YYYY-MM
    months = []
    cur = dt.date(start.year, start.month, 1)
    end_month = dt.date(end.year, end.month, 1)
    while cur <= end_month:
        months.append((cur.year, cur.month))
        # increment month
        if cur.month == 12:
            cur = dt.date(cur.year + 1, 1, 1)
        else:
            cur = dt.date(cur.year, cur.month + 1, 1)

    if not months:
        raise ValueError("No months in CHIRPS range")

    with tempfile.TemporaryDirectory() as tmpdir:
        accum = None

        for year, month in months:
            mm = f"{month:02d}"
            url = f"{CHIRPS_BASE_MONTHLY}/chirps-v2.0.{year}.{mm}.tif.gz"
            try:
                tif_path = _download_and_unzip_tif(url, tmpdir)
            except Exception as e:
                print(f"Warning: failed to download {url}: {e}")
                continue

            arr = _clip_raster_to_bbox(tif_path, bbox, out_shape=out_shape)
            if accum is None:
                accum = arr
            else:
                accum = np.where(np.isnan(accum), arr, np.where(np.isnan(arr), accum, accum + arr))

        if accum is None:
            raise RuntimeError("No CHIRPS files could be downloaded")

        return accum.astype(np.float32)


def generate_mock_rainfall(size=(100, 100)):
    """Generate mock rainfall data (fallback if CHIRPS fails)."""
    H, W = size
    np.random.seed(456)
    rainfall = np.random.uniform(50, 300, (H, W)).astype(np.float32)
    return rainfall


# ============== Hazard Computation Functions ==============

def compute_flood_risk(lulc, slope, rainfall=None):
    """
    Flood risk index based on LULC, slope, and optionally rainfall.
    1=water, 2=urban, 3=agriculture, 4=forest, 5=bare
    """
    urban = (lulc == 2)
    agri = (lulc == 3)

    risk = np.ones_like(lulc, dtype=np.int16)  # baseline low

    low_slope = slope < 5
    med_slope = (slope >= 5) & (slope < 15)

    risk[urban & low_slope] = 3  # high
    risk[agri & low_slope] = 2   # medium
    risk[urban & med_slope] = 2

    # If rainfall is provided, increase risk in high rainfall areas
    if rainfall is not None:
        high_rainfall = rainfall > 200  # mm threshold
        risk[(high_rainfall) & (risk < 3)] = np.minimum(risk[(high_rainfall) & (risk < 3)] + 1, 3)

    return risk


def compute_landslide_risk(lulc, slope, rainfall=None):
    """
    Landslide risk index based on LULC, slope, and optionally rainfall.
    """
    bare = (lulc == 5)
    forest = (lulc == 4)

    risk = np.ones_like(lulc, dtype=np.int16)  # baseline low
    steep = slope > 25
    mid = (slope > 15) & (slope <= 25)

    risk[(bare | forest) & steep] = 3  # high
    risk[(bare | forest) & mid] = 2    # medium

    # If rainfall is provided, increase risk in high rainfall areas with steep slopes
    if rainfall is not None:
        high_rainfall = rainfall > 150  # mm threshold
        risk[(high_rainfall) & (steep) & (risk < 3)] = 3
        risk[(high_rainfall) & (mid) & (risk < 2)] = 2

    return risk


def predict_future_rainfall(rainfall_2019, rainfall_2024, target_year=2030):
    """
    Simple linear trend projection for future rainfall based on 2019 and 2024 data.
    Projects to target_year (default 2030).
    """
    # Calculate annual change rate
    years_diff = 2024 - 2019
    annual_change = (rainfall_2024 - rainfall_2019) / years_diff
    
    # Project to target year
    years_to_target = target_year - 2024
    rainfall_future = rainfall_2024 + (annual_change * years_to_target)
    
    # Ensure no negative rainfall
    rainfall_future = np.maximum(rainfall_future, 0)
    
    return rainfall_future.astype(np.float32)


# ============== Rendering Functions ==============

def raster_to_png_bytes(raster):
    """Render categorical raster (1/2/3) to RGB PNG in memory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = {1: (0, 1, 0), 2: (1, 1, 0), 3: (1, 0, 0)}
    H, W = raster.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for cls, col in cmap.items():
        mask = raster == cls
        rgb[mask] = col

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.axis("off")
    ax.imshow(rgb)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def summarize_risk_by_percent(risk_map):
    total = risk_map.size
    summary = {}
    for cls, label in [(1, "low"), (2, "medium"), (3, "high")]:
        pct = float((risk_map == cls).sum()) / total * 100.0
        summary[label] = round(pct, 2)
    return summary


# ============== Main Analysis Pipeline ==============

def analyze_area(area_name: str, start_date: str, end_date: str):
    """
    High-level pipeline:
      1. geocode to get bbox and polygon
      2. Generate mock satellite data (for demo)
      3. classify current LULC
      4. predict future LULC
      5. Generate DEM (bbox-aware) + compute slope
      6. Get CHIRPS rainfall (2019, 2024) and predict 2030
      7. flood & landslide risk (current and future)
      8. summaries + PNG overlays
    """
    bbox, polygon = geocode_aoi(area_name)
    size = (100, 100)

    # For demo: use mock data instead of actual Sentinel-2 download
    # In production, you would use stac_search_sentinel2 and download_asset
    stack = generate_mock_satellite_data(size=size)
    
    lulc_current = classify_lulc(stack)
    lulc_future = run_ca_ann_future(lulc_current)

    # Generate bbox-aware DEM and compute slope
    dem = generate_mock_dem(bbox, size=size)
    slope = compute_slope_from_dem(dem)

    # Try to get real CHIRPS rainfall data, fallback to mock if it fails
    try:
        # Get rainfall for 2019 (baseline year)
        rainfall_2019 = get_chirps_monthly_rainfall(bbox, "2019-01-01", "2019-12-31", out_shape=size)
        # Get rainfall for 2024 (recent year)
        rainfall_2024 = get_chirps_monthly_rainfall(bbox, "2024-01-01", "2024-12-31", out_shape=size)
        # Predict 2030 rainfall using linear trend
        rainfall_2030 = predict_future_rainfall(rainfall_2019, rainfall_2024, target_year=2030)
        
        rainfall_current = rainfall_2024  # Use 2024 as current
        rainfall_future = rainfall_2030   # Use 2030 prediction as future
        print(f"Successfully loaded CHIRPS rainfall data for {area_name}")
    except Exception as e:
        print(f"Warning: Could not load CHIRPS data: {e}. Using mock rainfall.")
        rainfall_current = generate_mock_rainfall(size=size)
        # For future, slightly increase rainfall (climate change projection)
        rainfall_future = rainfall_current * 1.15

    # Compute hazard risks with rainfall
    flood_current = compute_flood_risk(lulc_current, slope, rainfall_current)
    flood_future = compute_flood_risk(lulc_future, slope, rainfall_future)
    landslide_current = compute_landslide_risk(lulc_current, slope, rainfall_current)
    landslide_future = compute_landslide_risk(lulc_future, slope, rainfall_future)

    flood_curr_png = raster_to_png_bytes(flood_current)
    flood_future_png = raster_to_png_bytes(flood_future)
    ls_curr_png = raster_to_png_bytes(landslide_current)
    ls_future_png = raster_to_png_bytes(landslide_future)

    return {
        "bbox": bbox,
        "flood_current_summary": summarize_risk_by_percent(flood_current),
        "flood_future_summary": summarize_risk_by_percent(flood_future),
        "landslide_current_summary": summarize_risk_by_percent(landslide_current),
        "landslide_future_summary": summarize_risk_by_percent(landslide_future),
        "flood_current_png": flood_curr_png,
        "flood_future_png": flood_future_png,
        "landslide_current_png": ls_curr_png,
        "landslide_future_png": ls_future_png,
    }