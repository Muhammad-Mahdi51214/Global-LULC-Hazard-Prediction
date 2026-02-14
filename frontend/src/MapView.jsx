import React from "react";
import { MapContainer, TileLayer, Rectangle, GeoJSON, useMap, LayersControl } from "react-leaflet";
import "leaflet/dist/leaflet.css";

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN;

// Fallback to OpenStreetMap if no Mapbox token
const getTileUrl = () => {
  if (MAPBOX_TOKEN) {
    return `https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12/tiles/{z}/{x}/{y}?access_token=${MAPBOX_TOKEN}`;
  }
  return "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
};

const getTileAttribution = () => {
  if (MAPBOX_TOKEN) {
    return '&copy; <a href="https://www.mapbox.com/about/maps/">Mapbox</a>';
  }
  return '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>';
};

const defaultCenter = [30, 70]; // Default center (Asia)
const defaultZoom = 4;

function MapController({ bbox }) {
  const map = useMap();

  React.useEffect(() => {
    if (bbox && bbox.length === 4) {
      const [minLon, minLat, maxLon, maxLat] = bbox;
      const bounds = [
        [minLat, minLon],
        [maxLat, maxLon],
      ];
      try {
        map.fitBounds(bounds, { padding: [20, 20] });
      } catch (e) {
        console.error("Error fitting bounds:", e);
      }
    }
  }, [bbox, map]);

  return null;
}

export default function MapView({ bbox, aoiGeoJson, lulcLayers }) {
  // Debug: log the lulcLayers prop
  console.log("MapView lulcLayers:", lulcLayers);

  // Convert bbox to Leaflet bounds format
  const rectangleBounds = bbox
    ? [
      [bbox[1], bbox[0]], // [minLat, minLon]
      [bbox[3], bbox[2]], // [maxLat, maxLon]
    ]
    : null;

  // Style for the boundary rectangle
  const boundaryStyle = {
    color: "#ff6b6b",
    weight: 3,
    fillColor: "#ff6b6b",
    fillOpacity: 0.1,
    dashArray: "5, 5",
  };

  return (
    <MapContainer
      style={{ height: "100%", width: "100%" }}
      center={defaultCenter}
      zoom={defaultZoom}
      scrollWheelZoom={true}
    >
      <LayersControl position="topright">
        {/* Base Layers */}
        <LayersControl.BaseLayer checked name="Satellite">
          <TileLayer
            attribution={getTileAttribution()}
            url={getTileUrl()}
            tileSize={MAPBOX_TOKEN ? 512 : 256}
            zoomOffset={MAPBOX_TOKEN ? -1 : 0}
          />
        </LayersControl.BaseLayer>

        <LayersControl.BaseLayer name="OpenStreetMap">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
        </LayersControl.BaseLayer>

        {/* LULC Overlay Layers */}
        {lulcLayers?.lulc2019 && (
          <LayersControl.Overlay checked name="🗺️ LULC 2019">
            <TileLayer
              url={lulcLayers.lulc2019}
              opacity={0.8}
              maxZoom={18}
              tileSize={256}
            />
          </LayersControl.Overlay>
        )}

        {lulcLayers?.lulc2024 && (
          <LayersControl.Overlay checked name="🗺️ LULC 2024">
            <TileLayer
              url={lulcLayers.lulc2024}
              opacity={0.8}
              maxZoom={18}
              tileSize={256}
            />
          </LayersControl.Overlay>
        )}

        {lulcLayers?.lulc2030 && (
          <LayersControl.Overlay name="🔮 LULC 2030 (Predicted)">
            <TileLayer
              url={lulcLayers.lulc2030}
              opacity={0.8}
              maxZoom={18}
              tileSize={256}
            />
          </LayersControl.Overlay>
        )}

        {/* Rainfall Overlay Layers */}
        {lulcLayers?.rainfall2019 && (
          <LayersControl.Overlay name="🌧️ Rainfall 2019">
            <TileLayer
              url={lulcLayers.rainfall2019}
              opacity={0.7}
              maxZoom={18}
              tileSize={256}
            />
          </LayersControl.Overlay>
        )}

        {lulcLayers?.rainfall2024 && (
          <LayersControl.Overlay name="🌧️ Rainfall 2024">
            <TileLayer
              url={lulcLayers.rainfall2024}
              opacity={0.7}
              maxZoom={18}
              tileSize={256}
            />
          </LayersControl.Overlay>
        )}

        {/* Flood Hazard Overlay Layers */}
        {lulcLayers?.flood2019 && (
          <LayersControl.Overlay name="🌊 Flood Risk 2019">
            <TileLayer
              url={lulcLayers.flood2019}
              opacity={0.7}
              maxZoom={18}
              tileSize={256}
            />
          </LayersControl.Overlay>
        )}

        {lulcLayers?.flood2024 && (
          <LayersControl.Overlay name="🌊 Flood Risk 2024">
            <TileLayer
              url={lulcLayers.flood2024}
              opacity={0.7}
              maxZoom={18}
              tileSize={256}
            />
          </LayersControl.Overlay>
        )}

        {lulcLayers?.flood2030 && (
          <LayersControl.Overlay name="🌊 Flood Risk 2030">
            <TileLayer
              url={lulcLayers.flood2030}
              opacity={0.7}
              maxZoom={18}
              tileSize={256}
            />
          </LayersControl.Overlay>
        )}

        {/* Landslide Hazard Overlay Layers */}
        {lulcLayers?.landslide2019 && (
          <LayersControl.Overlay name="🏔️ Landslide Risk 2019">
            <TileLayer
              url={lulcLayers.landslide2019}
              opacity={0.7}
              maxZoom={18}
              tileSize={256}
            />
          </LayersControl.Overlay>
        )}

        {lulcLayers?.landslide2024 && (
          <LayersControl.Overlay name="🏔️ Landslide Risk 2024">
            <TileLayer
              url={lulcLayers.landslide2024}
              opacity={0.7}
              maxZoom={18}
              tileSize={256}
            />
          </LayersControl.Overlay>
        )}

        {lulcLayers?.landslide2030 && (
          <LayersControl.Overlay name="🏔️ Landslide Risk 2030">
            <TileLayer
              url={lulcLayers.landslide2030}
              opacity={0.7}
              maxZoom={18}
              tileSize={256}
            />
          </LayersControl.Overlay>
        )}
      </LayersControl>

      <MapController bbox={bbox} />

      {/* Show boundary rectangle when bbox is available */}
      {rectangleBounds && (
        <Rectangle
          bounds={rectangleBounds}
          pathOptions={boundaryStyle}
        />
      )}

      {/* Alternatively, show GeoJSON polygon if provided */}
      {aoiGeoJson && aoiGeoJson.geometry && (
        <GeoJSON
          key={JSON.stringify(aoiGeoJson)}
          data={aoiGeoJson}
          style={() => boundaryStyle}
        />
      )}
    </MapContainer>
  );
}
