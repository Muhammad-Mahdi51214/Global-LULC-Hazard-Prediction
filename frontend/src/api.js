// frontend/src/api.js

const API_BASE = "http://localhost:8000"; // FastAPI backend

// Geocode an area name to get bbox and GeoJSON
export async function geocodeArea(area) {
    const res = await fetch(`${API_BASE}/api/geocode`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ area }),
    });

    if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || "Geocoding failed");
    }

    return res.json(); // { bbox, aoi, display_name }
}

// GEE-based LULC RF + CA-Markov (2019, 2024, 2030)
export async function runGeeLulcClassification(aoiGeoJson, years = [2019, 2024]) {
    const payload = {
        aoi: aoiGeoJson,      // GeoJSON Feature or FeatureCollection
        years,                // e.g. [2019, 2024]
        export: false,
        scale: 250,           // Use coarser scale for faster processing
        export_folder: "GEE_LULC",
        export_prefix: "LULC_",
    };

    // 10-minute timeout for long GEE processing
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10 * 60 * 1000);

    try {
        const res = await fetch(`${API_BASE}/api/lulc/classify`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!res.ok) {
            const txt = await res.text();
            throw new Error(txt || "GEE classification failed");
        }

        return res.json(); // { status, results: [...], projection_2030: {...} }
    } catch (err) {
        clearTimeout(timeoutId);
        if (err.name === 'AbortError') {
            throw new Error('Classification timed out after 10 minutes. Try a smaller area.');
        }
        throw err;
    }
}
