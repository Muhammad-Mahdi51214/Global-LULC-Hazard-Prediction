"""
Geo Boundaries Service - Nominatim Only
Fetches area boundaries from OpenStreetMap Nominatim API.
Works for any city, district, province, or country worldwide.
"""

import requests
import time
from fastapi import HTTPException
from typing import Dict, Any


# User-Agent that works with Nominatim
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def _make_nominatim_request(area_name: str, retries: int = 3) -> Dict[str, Any]:
    """
    Make a request to Nominatim with retry logic.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": area_name,
        "format": "json",
        "polygon_geojson": 1,
        "polygon_threshold": 0,  # Get full resolution polygon
        "limit": 1
    }
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Accept-Language": "en"
    }

    last_error = None
    for attempt in range(retries):
        try:
            if attempt > 0:
                time.sleep(1)
            
            r = requests.get(url, params=params, headers=headers, timeout=30)
            
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                time.sleep(2)
                continue
            else:
                last_error = f"Nominatim returned status {r.status_code}"
                
        except requests.RequestException as e:
            last_error = str(e)
            continue
    
    raise ValueError(f"Failed to contact Nominatim after {retries} attempts: {last_error}")


def _create_polygon_from_bbox(bbox_list):
    """
    Create a GeoJSON Polygon from a bounding box.
    bbox_list: [south, north, west, east] (Nominatim format, as strings)
    """
    south, north, west, east = [float(x) for x in bbox_list]
    
    # Create a polygon from the bounding box
    return {
        "type": "Polygon",
        "coordinates": [[
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south]  # Close the ring
        ]]
    }


def fetch_aoi_from_name(area_name: str) -> Dict[str, Any]:
    """
    Fetch AOI polygon from Nominatim (OpenStreetMap).
    Returns a FeatureCollection GeoJSON ready for GEE.
    If Nominatim returns a Point, we create a Polygon from the bounding box.
    """
    data = _make_nominatim_request(area_name)
    
    if not data:
        raise HTTPException(status_code=404, detail=f"No area found for '{area_name}'")

    result = data[0]
    polygon = result.get("geojson")
    
    # Check if the geometry is a Point - if so, create polygon from bbox
    if polygon and polygon.get("type") == "Point":
        print(f"  [geocode] Warning: Nominatim returned Point for '{area_name}', creating polygon from bbox")
        bbox = result.get("boundingbox")
        if bbox:
            polygon = _create_polygon_from_bbox(bbox)
        else:
            raise HTTPException(status_code=404, detail=f"No boundary polygon or bbox found for '{area_name}'")
    
    if not polygon:
        # No geojson at all, try to create from bbox
        bbox = result.get("boundingbox")
        if bbox:
            print(f"  [geocode] No geojson for '{area_name}', creating polygon from bbox")
            polygon = _create_polygon_from_bbox(bbox)
        else:
            raise HTTPException(status_code=404, detail=f"No boundary polygon found for '{area_name}'")

    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": area_name,
                    "display_name": result.get("display_name", area_name),
                },
                "geometry": polygon
            }
        ]
    }

    return feature_collection


def get_area_boundary(area_name: str) -> Dict[str, Any]:
    """
    Get area boundary with bbox for frontend display.
    If Nominatim returns a Point, we create a Polygon from the bounding box.
    """
    data = _make_nominatim_request(area_name)
    
    if not data:
        raise ValueError(f"No area found for '{area_name}'")

    result = data[0]
    polygon = result.get("geojson")
    
    # Nominatim bbox = [south, north, west, east]
    south, north, west, east = map(float, result["boundingbox"])
    bbox = [west, south, east, north]  # [minLon, minLat, maxLon, maxLat]
    
    # Check if the geometry is a Point - if so, create polygon from bbox
    if polygon and polygon.get("type") == "Point":
        print(f"  [geocode] Warning: Nominatim returned Point for '{area_name}', creating polygon from bbox")
        polygon = _create_polygon_from_bbox(result["boundingbox"])
    
    if not polygon:
        # No geojson at all, create from bbox
        print(f"  [geocode] No geojson for '{area_name}', creating polygon from bbox")
        polygon = _create_polygon_from_bbox(result["boundingbox"])

    display_name = result.get("display_name", area_name)

    feature = {
        "type": "Feature",
        "properties": {
            "name": area_name,
            "display_name": display_name,
        },
        "geometry": polygon
    }

    feature_collection = {
        "type": "FeatureCollection",
        "features": [feature]
    }

    return {
        "bbox": bbox,
        "aoi": feature_collection,
        "display_name": display_name,
        "source": "nominatim"
    }

