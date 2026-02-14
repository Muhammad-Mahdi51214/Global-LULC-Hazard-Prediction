"""
GEE LULC Classification + CA-Markov 2030 Prediction + CHIRPS Rainfall Analysis
- Uses ESA WorldCover 2021 to auto-generate training samples (no external training asset needed)
- RF classification for 2019 and 2024
- CA-Markov for 2030 prediction
- CHIRPS rainfall analysis and prediction
- Returns Kappa and OA accuracy metrics
"""

import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import ee

# Initialize EE
def initialize_ee():
    sa_email = os.getenv("EE_SERVICE_ACCOUNT")
    creds_file = os.getenv("EE_CREDENTIALS_FILE")
    
    if creds_file and not os.path.isabs(creds_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        creds_file = os.path.join(script_dir, creds_file)
    
    if sa_email and creds_file and os.path.exists(creds_file):
        credentials = ee.ServiceAccountCredentials(sa_email, key_file=creds_file)
        ee.Initialize(credentials)
        print(f"Initialized EE with service account: {sa_email}")
    else:
        try:
            ee.Initialize()
            print("Initialized EE with default credentials.")
        except Exception as e:
            print(f"Warning: EE initialization failed: {e}")

initialize_ee()

app = FastAPI(title="GEE LULC + CA-Markov + Rainfall")

# ------------ Constants ------------
# 6 classes: 0=water, 1=bare, 2=built, 3=vegetation, 4=cropland, 5=snow
PALETTE = ['419BDF', 'C2B280', 'C4281B', '3D9A50', 'F9D057', 'FFFFFF']
CLASS_LABELS = {0: 'water', 1: 'bare', 2: 'built', 3: 'vegetation', 4: 'cropland', 5: 'snow'}
CLASS_VALUES = [0, 1, 2, 3, 4, 5]

# Landsat 8 bands for classification
PRED_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'NDVI', 'NDBI', 'MNDWI', 'slope']

# ------------ Request model ------------
class ClassifyRequest(BaseModel):
    aoi: Dict[str, Any]  # GeoJSON Feature or FeatureCollection
    years: Optional[List[int]] = [2019, 2024]
    export: Optional[bool] = False
    scale: Optional[int] = 30

# ------------ Utility Functions ------------
def geojson_to_ee(geojson):
    """Convert GeoJSON to EE FeatureCollection"""
    if not geojson:
        raise ValueError("Empty geojson")
    
    t = geojson.get("type")
    print(f"  [geojson_to_ee] Input type: {t}")
    
    if t == "FeatureCollection":
        features = geojson.get("features", [])
        if features:
            first_geom = features[0].get("geometry", {})
            geom_type = first_geom.get("type", "Unknown")
            coords = first_geom.get("coordinates", [])
            print(f"  [geojson_to_ee] First feature geometry type: {geom_type}")
            if geom_type == "Polygon" and coords:
                print(f"  [geojson_to_ee] Polygon has {len(coords[0]) if coords else 0} vertices")
            elif geom_type == "MultiPolygon" and coords:
                print(f"  [geojson_to_ee] MultiPolygon has {len(coords)} parts")
            elif geom_type == "Point":
                print(f"  [geojson_to_ee] WARNING: Geometry is a Point, not a Polygon!")
        return ee.FeatureCollection(geojson)
    elif t == "Feature":
        geom = geojson.get("geometry", {})
        print(f"  [geojson_to_ee] Feature geometry type: {geom.get('type', 'Unknown')}")
        return ee.FeatureCollection(ee.Feature(geojson))
    else:
        print(f"  [geojson_to_ee] Raw geometry type: {t}")
        return ee.FeatureCollection(ee.Feature(ee.Geometry(geojson)))

# ------------ Landsat Processing ------------
def get_landsat_composite(aoi, year):
    """Get annual Landsat 8 composite with spectral indices"""
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)
    
    # Get the geometry - handle both FeatureCollection and Geometry
    aoi_geom = aoi.geometry()
    
    # Debug: print geometry bounds
    try:
        bounds = aoi_geom.bounds().getInfo()
        print(f"    AOI bounds: {bounds['coordinates']}")
    except Exception as e:
        print(f"    Warning: Could not get AOI bounds: {e}")
    
    # Landsat 8 Collection 2 Level 2
    collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(aoi_geom) \
        .filterDate(start, end) \
        .filter(ee.Filter.lt('CLOUD_COVER', 30))
    
    # Check collection size
    col_size = collection.size().getInfo()
    print(f"    Landsat collection for {year}: {col_size} images")
    
    if col_size == 0:
        raise ValueError(f"No Landsat images found for {year} in the AOI. Try a different area or year.")
    
    # Cloud masking function
    def mask_clouds(img):
        qa = img.select('QA_PIXEL')
        mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        return img.updateMask(mask)
    
    # Apply cloud mask and scale
    collection = collection.map(mask_clouds)
    
    # Create median composite and unmask to fill in gaps
    composite = collection.median().multiply(0.0000275).add(-0.2).unmask(0)
    
    # Select and rename bands
    composite = composite.select(
        ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
        ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    )
    
    # Add spectral indices
    ndvi = composite.normalizedDifference(['B5', 'B4']).rename('NDVI')
    ndbi = composite.normalizedDifference(['B6', 'B5']).rename('NDBI')
    mndwi = composite.normalizedDifference(['B3', 'B6']).rename('MNDWI')
    
    # Add slope from SRTM
    slope = ee.Terrain.slope(ee.Image('USGS/SRTMGL1_003')).rename('slope')
    
    # Combine all bands - NO CLIP (clipping causes empty tiles)
    result = composite.addBands([ndvi, ndbi, mndwi, slope])
    
    return result

# ------------ Load User's Training Assets ------------
def load_user_training_assets():
    """
    Load and merge multiple class-wise FeatureCollection assets from user's GEE account.
    Assets are at: users/muhammadmahdi/training_points/
    """
    base_path = "users/muhammadmahdi/training_points"
    
    # Asset names and their class values
    class_assets = [
        ('waterFC', 0),
        ('barelandFC', 1),
        ('builtupFC', 2),
        ('treeFC', 3),
        ('croplandFC', 4),
        ('snowFC', 5)
    ]
    
    merged_samples = None
    
    for asset_name, class_value in class_assets:
        asset_path = f"{base_path}/{asset_name}"
        try:
            fc = ee.FeatureCollection(asset_path)
            # Add lulc property - use direct set with server-side constant
            fc = fc.map(lambda f, cv=class_value: f.set('lulc', cv))
            
            if merged_samples is None:
                merged_samples = fc
            else:
                merged_samples = merged_samples.merge(fc)
            
            print(f"Loaded {asset_name} (class {class_value})")
        except Exception as e:
            print(f"Warning: Could not load {asset_path}: {e}")
    
    if merged_samples is None:
        raise ValueError("No training assets could be loaded")
    
    return merged_samples

# ------------ Training Sample Generation (Fallback) ------------
def generate_training_samples(aoi, num_samples=500):
    """
    Generate training samples from ESA WorldCover 2021.
    ESA WorldCover classes -> Our classes mapping:
    10 (Trees) -> 3 (vegetation)
    20 (Shrubland) -> 3 (vegetation)
    30 (Grassland) -> 3 (vegetation)
    40 (Cropland) -> 4 (cropland)
    50 (Built-up) -> 2 (built)
    60 (Bare/sparse veg) -> 1 (bare)
    70 (Snow/Ice) -> 5 (snow)
    80 (Water) -> 0 (water)
    90 (Herbaceous wetland) -> 3 (vegetation)
    95 (Mangroves) -> 3 (vegetation)
    100 (Moss/lichen) -> 3 (vegetation)
    """
    # ESA WorldCover 2021
    worldcover = ee.Image('ESA/WorldCover/v200/2021').clip(aoi.geometry())
    
    # Remap to our 6 classes
    from_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    to_values = [3, 3, 3, 4, 2, 1, 5, 0, 3, 3, 3]
    
    lulc_reference = worldcover.remap(from_values, to_values).rename('lulc')
    
    # Generate stratified random samples
    samples = lulc_reference.stratifiedSample(
        numPoints=num_samples,
        classBand='lulc',
        region=aoi.geometry(),
        scale=30,
        seed=42,
        geometries=True
    )
    
    return samples, lulc_reference

# ------------ Classification ------------
def get_training_composite(year):
    """
    Get Landsat composite for the TRAINING REGION (Haripur, Pakistan).
    This is where your training points are located.
    """
    # Haripur region coordinates (where training points are located)
    haripur_bbox = ee.Geometry.Rectangle([72.7, 33.7, 73.3, 34.1])
    
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)
    
    def mask_clouds(img):
        qa = img.select('QA_PIXEL')
        mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        return img.updateMask(mask)
    
    collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(haripur_bbox) \
        .filterDate(start, end) \
        .filter(ee.Filter.lt('CLOUD_COVER', 20)) \
        .map(mask_clouds)
    
    composite = collection.median().multiply(0.0000275).add(-0.2)
    
    composite = composite.select(
        ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
        ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    )
    
    # Add spectral indices
    ndvi = composite.normalizedDifference(['B5', 'B4']).rename('NDVI')
    ndbi = composite.normalizedDifference(['B6', 'B5']).rename('NDBI')
    mndwi = composite.normalizedDifference(['B3', 'B6']).rename('MNDWI')
    slope = ee.Terrain.slope(ee.Image('USGS/SRTMGL1_003')).rename('slope')
    
    return composite.addBands([ndvi, ndbi, mndwi, slope])

def classify_lulc(aoi, year, training_samples, scale=30):
    """
    Classify LULC for a given year using RF classifier.
    - Training: Uses HARIPUR Landsat + your training points
    - Classification: Applies trained model to ANY AOI globally
    """
    print(f"  Getting Landsat composite for HARIPUR TRAINING AREA ({year})...")

    # 1) TRAINING COMPOSITE (Haripur only)
    training_composite = get_training_composite(year)

    # 2) SAMPLE TRAINING DATA AT YOUR TRAINING POINTS
    print("  Sampling spectral values at training point locations...")
    training_data = training_composite.select(PRED_BANDS).sampleRegions(
        collection=training_samples,
        properties=["lulc"],
        scale=scale,
        geometries=False,
    )

    training_count = training_data.size().getInfo()
    print(f"  Found {training_count} training samples")
    if training_count == 0:
        raise ValueError(
            "No training samples could be extracted. "
            "Check if training points lie inside the Haripur training area."
        )

    # 3) TRAIN / TEST SPLIT
    training_data = training_data.randomColumn("random", seed=42)
    train_set = training_data.filter(ee.Filter.lte("random", 0.7))
    test_set = training_data.filter(ee.Filter.gt("random", 0.7))

    # 4) TRAIN RF
    print("  Training Random Forest classifier (250 trees)...")
    classifier = (
        ee.Classifier.smileRandomForest(numberOfTrees=250, seed=42)
        .train(
            features=train_set,
            classProperty="lulc",
            inputProperties=PRED_BANDS,
        )
    )

    # 5) GET COMPOSITE FOR TARGET AOI
    print(f"  Getting Landsat composite for TARGET AOI ({year})...")
    aoi_geom = aoi.geometry()
    aoi_composite = get_landsat_composite(aoi, year)

    # 6) CLASSIFY + CLIP STRICTLY TO AOI
    print("  Applying classifier to AOI...")
    raw_pred = (
        aoi_composite.select(PRED_BANDS)
        .unmask(0)
        .classify(classifier)
        .toInt()
        .rename("lulc")
    )

    # Strong clip to AOI so:
    #  - tiles only draw inside AOI
    #  - area stats only computed there
    prediction = raw_pred.clip(aoi_geom)

    print(f"  Classification complete for {year}")

    # 7) ACCURACY (based on Haripur training region)
    test_classified = test_set.classify(classifier)
    confusion_matrix = test_classified.errorMatrix("lulc", "classification")

    oa = float(confusion_matrix.accuracy().getInfo() or 0.0)
    kappa = float(confusion_matrix.kappa().getInfo() or 0.0)
    print(f"  Accuracy: OA={oa:.4f}, Kappa={kappa:.4f}")

    # 8) AREA STATS INSIDE AOI ONLY
    area_stats = compute_area_stats(prediction, aoi, scale)

    # 9) TILE URL FOR LEAFLET
    tile_url = None
    try:
        vis_params = {"min": 0, "max": 5, "palette": PALETTE}
        map_id_dict = prediction.getMapId(vis_params)
        tile_url = map_id_dict["tile_fetcher"].url_format
        print(f"  Generated RF LULC map tile URL for {year}")
    except Exception as e:
        print(f"  Warning: Could not generate map tile URL: {e}")

    # 10) THUMBNAIL FOR SIDEBAR
    thumb_url = None
    try:
        region_coords = aoi_geom.bounds().getInfo()["coordinates"]
        thumb_url = prediction.getThumbURL(
            {
                "min": 0,
                "max": 5,
                "palette": ",".join(PALETTE),
                "region": region_coords,
                "dimensions": 512,
            }
        )
    except Exception as e:
        print(f"  Warning: Could not generate thumbnail: {e}")

    return {
        "year": year,
        "prediction": prediction,
        "oa": round(oa, 4),
        "kappa": round(kappa, 4),
        "area_stats": area_stats,
        "thumb_url": thumb_url,
        "tile_url": tile_url,
    }

def compute_area_stats(prediction, aoi, scale):
    """Compute area statistics for each class INSIDE AOI."""
    area_image = ee.Image.pixelArea().divide(1e6)  # km²

    stats = []
    aoi_geom = aoi.geometry()

    for class_val in CLASS_VALUES:
        class_mask = prediction.eq(class_val)
        class_area = area_image.updateMask(class_mask).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi_geom,
            scale=scale,
            maxPixels=1e13,
        ).get("area")

        area_km2 = float(ee.Number(class_area).getInfo() or 0.0)
        stats.append(
            {
                "class": class_val,
                "label": CLASS_LABELS[class_val],
                "area_km2": round(area_km2, 2),
            }
        )

    total_area = sum(s["area_km2"] for s in stats)
    for s in stats:
        s["pct"] = round(
            (s["area_km2"] / total_area * 100) if total_area > 0 else 0.0, 2
        )

    return stats

# ------------ CA-Markov Prediction ------------
def predict_2030(pred_2019, pred_2024, aoi, scale):
    """
    CA-Markov prediction for 2030 based on transition matrix from 2019→2024.
    Output is clipped to AOI and returned with tile_url + area_stats.
    """
    aoi_geom = aoi.geometry()

    # 1) TRANSITION MATRIX (client side)
    T = compute_transition_matrix(pred_2019, pred_2024, aoi, scale)

    # years: 2019→2024 (5y), then 2024→2030 (6y). Approx 1 step.
    steps = 1
    Tn = np.linalg.matrix_power(T, steps)

    # 2) APPLY MARKOV TO 2024 MAP
    prob_bands = []
    for j_idx, _ in enumerate(CLASS_VALUES):
        prob = ee.Image.constant(0).toFloat()
        for i_idx, ci in enumerate(CLASS_VALUES):
            mask_i = pred_2024.eq(ci).toFloat()
            prob = prob.add(mask_i.multiply(float(Tn[i_idx, j_idx])))
        prob_bands.append(prob.rename(f"prob_{j_idx}"))

    prob_stack = ee.Image.cat(prob_bands)

    # Argmax → predicted class
    prediction_2030 = (
        prob_stack.toArray()
        .arrayArgmax()
        .arrayFlatten([["lulc"]])
        .toInt()
        .clip(aoi_geom)
    )

    # 3) SIMPLE CA SMOOTHING (mode filter) - reduced to 1 iteration to maintain resolution
    kernel = ee.Kernel.square(1)
    for _ in range(1):
        prediction_2030 = (
            prediction_2030.reduceNeighborhood(
                reducer=ee.Reducer.mode(), kernel=kernel
            )
            .rename("lulc")
            .toInt()
            .clip(aoi_geom)
        )

    # 4) AREA STATS + TILE URL + THUMB
    area_stats = compute_area_stats(prediction_2030, aoi, scale)

    tile_url = None
    try:
        vis_params = {"min": 0, "max": 5, "palette": PALETTE}
        map_id_dict = prediction_2030.getMapId(vis_params)
        tile_url = map_id_dict["tile_fetcher"].url_format
        print("  Generated 2030 map tile URL")
    except Exception as e:
        print(f"  Warning: Could not generate 2030 map tile URL: {e}")

    thumb_url = None
    try:
        region_coords = aoi_geom.bounds().getInfo()["coordinates"]
        thumb_url = prediction_2030.getThumbURL(
            {
                "min": 0,
                "max": 5,
                "palette": ",".join(PALETTE),
                "region": region_coords,
                "dimensions": 512,
            }
        )
    except Exception as e:
        print(f"  Warning: Could not generate 2030 thumbnail: {e}")

    return {
        "year": 2030,
        "area_stats": area_stats,
        "thumb_url": thumb_url,
        "tile_url": tile_url,
        "transition_matrix": T.tolist(),
        "image": prediction_2030  # Include image for hazard computation
    }

def compute_transition_matrix(pred1, pred2, aoi, scale):
    """Compute transition probability matrix between two classification maps"""
    n = len(CLASS_VALUES)
    counts = np.zeros((n, n), dtype=float)
    
    for i, ci in enumerate(CLASS_VALUES):
        for j, cj in enumerate(CLASS_VALUES):
            mask = pred1.eq(ci).And(pred2.eq(cj))
            area = ee.Image.pixelArea().updateMask(mask).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=aoi.geometry(),
                scale=scale,
                maxPixels=1e13
            ).get('area')
            counts[i, j] = ee.Number(area).getInfo() or 0
    
    # Row-normalize
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums

# ============== Hazard Computation Functions (GEE) ==============

def flood_risk_image(lulc_img, slope_img, rainfall_img=None):
    """
    GEE version.
    lulc codes (0..5):
      0 = water, 1 = bare, 2 = built, 3 = vegetation, 4 = cropland, 5 = snow
    Risk: 1=low, 2=medium, 3=high
    """
    # Booleans for LULC classes
    built = lulc_img.eq(2)
    crop = lulc_img.eq(4)

    # Baseline = low risk everywhere
    risk = ee.Image.constant(1).toInt()

    low_slope = slope_img.lt(5)
    med_slope = slope_img.gte(5).And(slope_img.lt(15))

    # Rules similar to numpy code
    risk = risk.where(built.And(low_slope), 3)     # high
    risk = risk.where(crop.And(low_slope), 2)      # medium
    risk = risk.where(built.And(med_slope), 2)     # medium

    if rainfall_img is not None:
        high_rain = rainfall_img.gt(200)  # mm threshold
        # increase risk by 1, cap at 3, where high_rain and risk<3
        incr = risk.add(1).min(3)
        risk = risk.where(high_rain.And(risk.lt(3)), incr)

    return risk.rename('flood_risk').toInt()


def landslide_risk_image(lulc_img, slope_img, rainfall_img=None):
    """
    GEE version.
    Risk: 1=low, 2=medium, 3=high
    """
    bare = lulc_img.eq(1)    # bare soil
    forest = lulc_img.eq(3)  # vegetation / forest-ish

    risk = ee.Image.constant(1).toInt()
    steep = slope_img.gt(25)
    mid = slope_img.gt(15).And(slope_img.lte(25))

    risk = risk.where(bare.Or(forest).And(steep), 3)  # high
    risk = risk.where(bare.Or(forest).And(mid),   2)  # medium

    if rainfall_img is not None:
        high_rain = rainfall_img.gt(150)
        risk = risk.where(high_rain.And(steep).And(risk.lt(3)), 3)
        risk = risk.where(high_rain.And(mid).And(risk.lt(2)),   2)

    return risk.rename('landslide_risk').toInt()


def make_hazard_result(hazard_img, aoi, scale, name):
    """
    Compute area stats + tile URL for a 3-level hazard image.
    Values: 1=low, 2=medium, 3=high
    """
    # Clip hazard image to AOI
    hazard_clipped = hazard_img.clip(aoi.geometry())
    
    area_img = ee.Image.pixelArea().divide(1e6)  # km²

    stats = []
    for level, label in [(1, "Low"), (2, "Medium"), (3, "High")]:
        mask = hazard_clipped.eq(level)
        area = area_img.updateMask(mask).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi.geometry(),
            scale=scale,
            maxPixels=1e13
        ).get('area')
        val = ee.Number(area).getInfo() or 0
        stats.append({
            "level": level,
            "label": label,
            "area_km2": round(val, 2)
        })

    # tile URL - use clipped image
    tile_url = None
    try:
        vis = {
            "min": 1,
            "max": 3,
            "palette": ["#2ecc71", "#f1c40f", "#e74c3c"]  # green, yellow, red
        }
        map_id = hazard_clipped.getMapId(vis)
        tile_url = map_id["tile_fetcher"].url_format
    except Exception as e:
        print(f"Warning: could not get tile for {name}: {e}")

    return {
        "name": name,
        "stats": stats,
        "tile_url": tile_url
    }


# ------------ CHIRPS Rainfall Analysis ------------
def get_rainfall_image_and_stats(aoi, year):
    """Return CHIRPS annual rainfall image + stats + thumbnail."""
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)

    chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
        .filterDate(start, end) \
        .filterBounds(aoi.geometry())

    annual_precip = chirps.sum().clip(aoi.geometry())  # mm/year

    stats_dict = annual_precip.reduceRegion(
        reducer=ee.Reducer.mean()
            .combine(ee.Reducer.min(), sharedInputs=True)
            .combine(ee.Reducer.max(), sharedInputs=True),
        geometry=aoi.geometry(),
        scale=5000,
        maxPixels=1e13
    ).getInfo() or {}

    thumb_url = None
    tile_url = None
    try:
        vis_params = {
            'min': 0,
            'max': 2000,
            'palette': ['white', 'lightblue', 'blue', 'darkblue', 'purple']
        }
        thumb_url = annual_precip.getThumbURL({
            **vis_params,
            'region': aoi.geometry().bounds().getInfo()['coordinates'],
            'dimensions': 512
        })
        map_id = annual_precip.getMapId(vis_params)
        tile_url = map_id['tile_fetcher'].url_format
        print(f"  Generated rainfall tile URL for {year}")
    except Exception as e:
        print(f"  Warning: Could not generate rainfall visuals: {e}")

    return annual_precip, {
        'year': year,
        'mean_mm': round(stats_dict.get('precipitation_mean', 0), 2),
        'min_mm': round(stats_dict.get('precipitation_min', 0), 2),
        'max_mm': round(stats_dict.get('precipitation_max', 0), 2),
        'thumb_url': thumb_url,
        'tile_url': tile_url
    }

def predict_rainfall_2030(aoi, rainfall_2019, rainfall_2024):
    """Simple linear projection of rainfall to 2030"""
    # Linear trend from 2019 to 2024
    rate_per_year = (rainfall_2024['mean_mm'] - rainfall_2019['mean_mm']) / 5
    
    # Project to 2030 (6 years from 2024)
    projected_mean = rainfall_2024['mean_mm'] + (rate_per_year * 6)
    
    return {
        'year': 2030,
        'projected_mean_mm': round(projected_mean, 2),
        'trend_per_year_mm': round(rate_per_year, 2),
        'note': 'Linear projection based on 2019-2024 trend'
    }

# ------------ Main Classification Endpoint ------------
@app.post("/classify")
def classify(req: ClassifyRequest):
    """
    Main classification endpoint.
    - Classifies LULC for 2019 and 2024 using RF
    - Predicts LULC 2030 using CA-Markov
    - Analyzes rainfall for 2019, 2024 and predicts 2030
    - Computes flood and landslide hazard maps
    - Returns accuracy metrics (OA, Kappa)
    """
    try:
        aoi = geojson_to_ee(req.aoi)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid AOI: {e}")
    
    # Validate the geometry is not empty
    try:
        geom = aoi.geometry()
        bounds = geom.bounds().getInfo()
        print(f"AOI bounds: {bounds}")
        if not bounds or not bounds.get('coordinates'):
            raise ValueError("AOI geometry bounds are empty or invalid")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid AOI geometry: {e}")
    
    years = req.years or [2019, 2024]
    scale = req.scale or 30
    
    print(f"Processing classification for {years} at scale {scale}m...")
    
    # Step 1: Load user's training samples from GEE assets
    print("Loading YOUR training samples from GEE assets...")
    print("  Loading training assets from users/muhammadmahdi/training_points/...")
    training_samples = load_user_training_assets()
    print(f"  Successfully loaded training samples")
    
    # Step 2: Classify each year
    results = []
    predictions = {}
    
    for year in years:
        print(f"Classifying LULC for {year}...")
        result = classify_lulc(aoi, year, training_samples, scale)
        predictions[year] = result['prediction']
        results.append({
            'year': result['year'],
            'oa': result['oa'],
            'kappa': result['kappa'],
            'area_stats': result['area_stats'],
            'thumb_url': result['thumb_url'],
            'tile_url': result['tile_url']
        })
    
    # Step 3: CA-Markov prediction for 2030
    print("Predicting LULC 2030 using CA-Markov...")
    year0, yearN = years[0], years[-1]
    projection_2030 = predict_2030(predictions[year0], predictions[yearN], aoi, scale)
    
    # Step 4: Rainfall analysis (using new function that returns images)
    print("Analyzing rainfall data...")
    rainfall_results = []
    
    # Get images + stats for 2019 and 2024
    rain_img_2019, rainfall_2019 = get_rainfall_image_and_stats(aoi, 2019)
    rain_img_2024, rainfall_2024 = get_rainfall_image_and_stats(aoi, 2024)
    
    rainfall_results.append(rainfall_2019)
    rainfall_results.append(rainfall_2024)
    
    # Predict 2030 rainfall (stats only)
    rainfall_2030 = predict_rainfall_2030(aoi, rainfall_2019, rainfall_2024)
    rainfall_results.append(rainfall_2030)
    
    # Step 5: Compute slope image
    print("Computing slope image...")
    slope_img = ee.Terrain.slope(ee.Image('USGS/SRTMGL1_003').clip(aoi.geometry())).rename('slope')
    
    # Step 6: Compute hazard maps (2019, 2024, 2030)
    print("Computing hazard maps (2019, 2024, 2030)...")
    hazards = {}
    
    # ---- 2019 hazards ----
    lulc_2019_img = predictions[2019]
    flood_2019_img = flood_risk_image(lulc_2019_img, slope_img, rain_img_2019)
    landslide_2019_img = landslide_risk_image(lulc_2019_img, slope_img, rain_img_2019)
    
    hazards["2019"] = {
        "flood": make_hazard_result(flood_2019_img, aoi, scale, "Flood 2019"),
        "landslide": make_hazard_result(landslide_2019_img, aoi, scale, "Landslide 2019")
    }
    
    # ---- 2024 hazards ----
    lulc_2024_img = predictions[2024]
    flood_2024_img = flood_risk_image(lulc_2024_img, slope_img, rain_img_2024)
    landslide_2024_img = landslide_risk_image(lulc_2024_img, slope_img, rain_img_2024)
    
    hazards["2024"] = {
        "flood": make_hazard_result(flood_2024_img, aoi, scale, "Flood 2024"),
        "landslide": make_hazard_result(landslide_2024_img, aoi, scale, "Landslide 2024")
    }
    
    # ---- 2030 hazards ----
    lulc_2030_img = projection_2030.get("image")
    if lulc_2030_img:
        # For rainfall 2030, use 2024 rainfall image since we don't have full 2030 image
        flood_2030_img = flood_risk_image(lulc_2030_img, slope_img, rain_img_2024)
        landslide_2030_img = landslide_risk_image(lulc_2030_img, slope_img, rain_img_2024)
        
        hazards["2030"] = {
            "flood": make_hazard_result(flood_2030_img, aoi, scale, "Flood 2030"),
            "landslide": make_hazard_result(landslide_2030_img, aoi, scale, "Landslide 2030")
        }
    else:
        print("Warning: 2030 LULC image not available; skipping 2030 hazards.")
    
    print("Classification complete!")
    
    # Remove image from projection_2030 before returning (not JSON serializable)
    projection_2030_response = {k: v for k, v in projection_2030.items() if k != 'image'}
    
    return {
        'status': 'ok',
        'lulc_results': results,
        'projection_2030': projection_2030_response,
        'rainfall': rainfall_results,
        'hazards': hazards
    }


@app.get("/health")
def health():
    return {"status": "alive"}