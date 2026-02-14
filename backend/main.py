from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import local modules
try:
    from . import gee_service as gee_ca
    from . import geo_boundaries
except ImportError:
    import gee_service as gee_ca
    import geo_boundaries

app = FastAPI(title="Global EO Hazard EWS")

origins = ["http://localhost:5173", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== Request Models ==============
class GeocodeRequest(BaseModel):
    area: str

class ClassifyByNameRequest(BaseModel):
    area: str
    years: list = [2019, 2024]

# ============== Geocode Endpoint ==============
@app.post("/api/geocode")
def api_geocode(req: GeocodeRequest):
    """
    Geocode an area name to get bbox and GeoJSON polygon.
    Uses Nominatim (OpenStreetMap) API.
    """
    try:
        result = geo_boundaries.get_area_boundary(req.area)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geocoding failed: {str(e)}")

# ============== Classify by Area Name Endpoint ==============
@app.post("/api/lulc/by-name")
def classify_by_name(req: ClassifyByNameRequest):
    """
    One-step classification: Area name → Nominatim → GEE Classification
    
    Frontend user types area name, backend:
    1. Fetches polygon from Nominatim
    2. Sends polygon to GEE classifier
    3. Runs RF LULC for each year
    4. Predicts LULC 2030 (CA-Markov)
    5. Returns all LULC maps + stats
    """
    # Step 1: Get AOI GeoJSON boundary from Nominatim
    aoi_geojson = geo_boundaries.fetch_aoi_from_name(req.area)

    # Step 2: Build GEE request
    gee_request = gee_ca.ClassifyRequest(
        aoi=aoi_geojson,
        years=req.years,
        export=False,
        scale=30
    )

    # Step 3: Run classification
    try:
        result = gee_ca.classify(gee_request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============== GEE LULC Classification Endpoint ==============
@app.post("/api/lulc/classify")
def api_lulc_classify(req: gee_ca.ClassifyRequest):
    """
    Run GEE RF LULC + CA-Markov for a given AOI (GeoJSON).
    """
    try:
        result = gee_ca.classify(req)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "alive"}