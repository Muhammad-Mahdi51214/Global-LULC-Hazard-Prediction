import React, { useState } from "react";
import "./App.css";
import MapView from "./MapView";
import { geocodeArea, runGeeLulcClassification } from "./api";

export default function App() {
  const [area, setArea] = useState("Islamabad, Pakistan");
  const [loading, setLoading] = useState(false);
  const [bbox, setBbox] = useState(null);
  const [aoiGeoJson, setAoiGeoJson] = useState(null);
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState("");
  const [lulcLayers, setLulcLayers] = useState(null);


  // Classification results
  const [lulcResults, setLulcResults] = useState(null);
  const [projection2030, setProjection2030] = useState(null);
  const [rainfallData, setRainfallData] = useState(null);
  const [hazardsData, setHazardsData] = useState(null);
  const [isClassifying, setIsClassifying] = useState(false);

  // Step 1: Geocode and show boundary
  async function handleGeocode() {
    setLoading(true);
    setError("");
    setLulcResults(null);
    setProjection2030(null);
    setRainfallData(null);

    try {
      const geoResult = await geocodeArea(area);
      setBbox(geoResult.bbox);
      setAoiGeoJson(geoResult.aoi);
      setDisplayName(geoResult.display_name || area);
    } catch (e) {
      console.error(e);
      setError(e.message || "Geocoding failed");
    } finally {
      setLoading(false);
    }
  }

  // Step 2: Run classification
  async function handleClassify() {
    if (!aoiGeoJson) {
      setError("Please search for an area first");
      return;
    }

    setIsClassifying(true);
    setError("");

    try {
      const data = await runGeeLulcClassification(aoiGeoJson, [2019, 2024]);

      // Extract tile URLs from backend
      const layers = {};

      // LULC layers
      data.lulc_results.forEach(r => {
        if (r.year === 2019 && r.tile_url) layers.lulc2019 = r.tile_url;
        if (r.year === 2024 && r.tile_url) layers.lulc2024 = r.tile_url;
      });

      if (data.projection_2030?.tile_url) {
        layers.lulc2030 = data.projection_2030.tile_url;
      }

      // Rainfall layers
      if (data.rainfall) {
        data.rainfall.forEach(r => {
          if (r.year === 2019 && r.tile_url) layers.rainfall2019 = r.tile_url;
          if (r.year === 2024 && r.tile_url) layers.rainfall2024 = r.tile_url;
        });
      }

      // Hazard layers
      if (data.hazards) {
        if (data.hazards["2019"]?.flood?.tile_url) layers.flood2019 = data.hazards["2019"].flood.tile_url;
        if (data.hazards["2019"]?.landslide?.tile_url) layers.landslide2019 = data.hazards["2019"].landslide.tile_url;
        if (data.hazards["2024"]?.flood?.tile_url) layers.flood2024 = data.hazards["2024"].flood.tile_url;
        if (data.hazards["2024"]?.landslide?.tile_url) layers.landslide2024 = data.hazards["2024"].landslide.tile_url;
        if (data.hazards["2030"]?.flood?.tile_url) layers.flood2030 = data.hazards["2030"].flood.tile_url;
        if (data.hazards["2030"]?.landslide?.tile_url) layers.landslide2030 = data.hazards["2030"].landslide.tile_url;
      }

      // Save into React state
      console.log("MapView lulcLayers:", layers);
      setLulcLayers(layers);

      setLulcResults(data.lulc_results);
      setProjection2030(data.projection_2030);
      setHazardsData(data.hazards);
      setRainfallData(data.rainfall);
    } catch (e) {
      console.error(e);
      setError(e.message || "Classification failed");
    } finally {
      setIsClassifying(false);
    }
  }

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="logo-mark">🌍</div>
        <div className="logo-text">
          <h1>EarthGuard: Global LULC & Hazard Prediction</h1>
          <p>RF Classification + CA-Markov 2030 + Flood & Landslide Risk Analysis</p>
        </div>
      </header>

      <main className="app-main">
        <aside className="sidebar">
          {/* Search Panel */}
          <section className="panel">
            <h2>🌍 Search Area</h2>
            <input
              type="text"
              value={area}
              onChange={(e) => setArea(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleGeocode()}
              placeholder="e.g., Lahore, Pakistan"
              style={{ width: "100%", padding: "0.75rem", fontSize: "1rem", marginBottom: "0.75rem" }}
            />
            <button
              className="btn-primary"
              disabled={loading}
              onClick={handleGeocode}
              style={{ width: "100%", marginBottom: "0.5rem" }}
            >
              {loading ? "Searching..." : "🔍 Search & Show Boundary"}
            </button>

            {bbox && (
              <div style={{ marginTop: "0.75rem", padding: "0.5rem", background: "#1a1a2e", borderRadius: "4px" }}>
                <p style={{ color: "#8be9fd", fontSize: "0.9rem" }}>📍 {displayName}</p>
              </div>
            )}
            {error && <p className="error-msg" style={{ marginTop: "0.5rem" }}>{error}</p>}
          </section>

          {/* Classification Panel */}
          {bbox && (
            <section className="panel">
              <h2>🛰️ Run Classification</h2>
              <p style={{ fontSize: "0.85rem", color: "#aaa", marginBottom: "0.75rem" }}>
                RF LULC for 2019 & 2024 → CA-Markov 2030
              </p>
              <button
                className="btn-primary"
                disabled={isClassifying}
                onClick={handleClassify}
                style={{ width: "100%" }}
              >
                {isClassifying ? "⏳ Processing (may take 1-2 min)..." : "🚀 Run Classification"}
              </button>
            </section>
          )}

          {/* LULC Results */}
          {lulcResults && (
            <section className="panel">
              <h2>📊 LULC Classification Results</h2>
              {lulcResults.map((r) => (
                <div key={r.year} style={{ marginBottom: "1.5rem" }}>
                  <h3 style={{ marginBottom: "0.5rem", color: "#50fa7b" }}>LULC {r.year}</h3>
                  <p style={{ fontSize: "0.85rem", color: "#bd93f9", marginBottom: "0.5rem" }}>
                    OA: {(r.oa * 100).toFixed(1)}% | Kappa: {r.kappa.toFixed(3)}
                  </p>
                  <img
                    src={r.thumb_url}
                    alt={`LULC ${r.year}`}
                    style={{ maxWidth: "100%", border: "1px solid #444", borderRadius: "4px" }}
                  />
                  <ul
                    style={{
                      fontSize: "0.8rem",
                      marginTop: "0.5rem",
                      paddingLeft: "1rem",
                      color: "#ccc",
                    }}
                  >
                    {r.area_stats?.map((s) => (
                      <li key={s.class}>
                        {s.label}: {s.area_km2} km² ({s.pct}%)
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </section>
          )}

          {/* 2030 Projection */}
          {projection2030 && (
            <section className="panel">
              <h2>🔮 LULC 2030 Prediction</h2>
              <p style={{ fontSize: "0.85rem", color: "#aaa", marginBottom: "0.5rem" }}>
                CA-Markov Projection
              </p>
              <img
                src={projection2030.thumb_url}
                alt="LULC 2030"
                style={{ maxWidth: "100%", border: "1px solid #444", borderRadius: "4px" }}
              />
              <ul
                style={{
                  fontSize: "0.8rem",
                  marginTop: "0.5rem",
                  paddingLeft: "1rem",
                  color: "#ccc",
                }}
              >
                {projection2030.area_stats?.map((s) => (
                  <li key={s.class}>
                    {s.label}: {s.area_km2} km² ({s.pct}%)
                  </li>
                ))}
              </ul>

            </section>
          )}

          {/* Rainfall Data */}
          {rainfallData && (
            <section className="panel">
              <h2>🌧️ Rainfall Analysis (CHIRPS)</h2>
              {rainfallData.map((r) => (
                <div key={r.year} style={{ marginBottom: "1rem" }}>
                  <h4 style={{ color: "#8be9fd" }}>
                    {r.year} {r.year === 2030 ? "(Projected)" : ""}
                  </h4>
                  {r.thumb_url && (
                    <img
                      src={r.thumb_url}
                      alt={`Rainfall ${r.year}`}
                      style={{ maxWidth: "100%", border: "1px solid #444", borderRadius: "4px", marginBottom: "0.5rem" }}
                    />
                  )}
                  <p style={{ fontSize: "0.85rem", color: "#ccc" }}>
                    {r.projected_mean_mm !== undefined ? (
                      <>Projected Mean: {r.projected_mean_mm} mm/year<br />Trend: {r.trend_per_year_mm} mm/year</>
                    ) : (
                      <>Mean: {r.mean_mm} mm | Min: {r.min_mm} mm | Max: {r.max_mm} mm</>
                    )}
                  </p>
                </div>
              ))}
            </section>
          )}

          {/* Hazard Risk Data */}
          {hazardsData && (
            <section className="panel">
              <h2>⚠️ Hazard Risk Analysis</h2>
              {["2019", "2024", "2030"].map((year) => (
                hazardsData[year] && (
                  <div key={year} style={{ marginBottom: "1.5rem" }}>
                    <h3 style={{ marginBottom: "0.5rem", color: "#ff79c6" }}>Year {year}</h3>

                    {/* Flood Risk */}
                    {hazardsData[year].flood && (
                      <div style={{ marginBottom: "0.75rem" }}>
                        <h4 style={{ color: "#50fa7b", fontSize: "0.9rem" }}>🌊 Flood Risk</h4>
                        <ul style={{ fontSize: "0.8rem", paddingLeft: "1rem", color: "#ccc" }}>
                          {hazardsData[year].flood.stats?.map((s) => (
                            <li key={s.level} style={{ color: s.level === 1 ? "#2ecc71" : s.level === 2 ? "#f1c40f" : "#e74c3c" }}>
                              {s.label}: {s.area_km2} km²
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Landslide Risk */}
                    {hazardsData[year].landslide && (
                      <div>
                        <h4 style={{ color: "#bd93f9", fontSize: "0.9rem" }}>🏔️ Landslide Risk</h4>
                        <ul style={{ fontSize: "0.8rem", paddingLeft: "1rem", color: "#ccc" }}>
                          {hazardsData[year].landslide.stats?.map((s) => (
                            <li key={s.level} style={{ color: s.level === 1 ? "#2ecc71" : s.level === 2 ? "#f1c40f" : "#e74c3c" }}>
                              {s.label}: {s.area_km2} km²
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )
              ))}
            </section>
          )}


          {/* Legend */}
          <section className="panel">
            <h2>🎨 Legend</h2>
            <h4 style={{ marginBottom: "0.5rem", color: "#8be9fd" }}>LULC Classes</h4>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", fontSize: "0.8rem", marginBottom: "1rem" }}>
              <span style={{ background: "#419BDF", padding: "2px 8px", borderRadius: "3px" }}>Water</span>
              <span style={{ background: "#C2B280", padding: "2px 8px", borderRadius: "3px", color: "#000" }}>Bare</span>
              <span style={{ background: "#C4281B", padding: "2px 8px", borderRadius: "3px" }}>Built</span>
              <span style={{ background: "#3D9A50", padding: "2px 8px", borderRadius: "3px" }}>Vegetation</span>
              <span style={{ background: "#F9D057", padding: "2px 8px", borderRadius: "3px", color: "#000" }}>Cropland</span>
              <span style={{ background: "#FFFFFF", padding: "2px 8px", borderRadius: "3px", color: "#000" }}>Snow</span>
            </div>
            <h4 style={{ marginBottom: "0.5rem", color: "#8be9fd" }}>Hazard Risk</h4>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", fontSize: "0.8rem" }}>
              <span style={{ background: "#2ecc71", padding: "2px 8px", borderRadius: "3px", color: "#000" }}>Low</span>
              <span style={{ background: "#f1c40f", padding: "2px 8px", borderRadius: "3px", color: "#000" }}>Medium</span>
              <span style={{ background: "#e74c3c", padding: "2px 8px", borderRadius: "3px" }}>High</span>
            </div>
          </section>
        </aside>

        <section className="map-container">
          <MapView
            bbox={bbox}
            aoiGeoJson={aoiGeoJson}
            lulcLayers={lulcLayers}
          />

        </section>
      </main>

      <footer className="app-footer">
        <p>Powered by Google Earth Engine, ESA WorldCover, CHIRPS & Random Forest ML</p>
      </footer>
    </div>
  );
}