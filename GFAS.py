from __future__ import annotations
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit_folium import st_folium  # keep if you want folium later
import folium  # not used in this cleaned version, but kept if needed
from PIL import Image
import shapely  
# Spatial helpers
from shapely.geometry import Point, Polygon

# Optional deps checks
try:
    import cv2
    OPENCV_OK = True
except Exception:
    OPENCV_OK = False

try:
    from PIL import Image as PIL_Image  # alias to verify PIL availability
    PIL_OK = True
except Exception:
    PIL_OK = False

# ---- Page Config (single call) ----
st.set_page_config(page_title="Geo-Fencing Alert System (DEMO)", layout="wide")

# ---- Terms & Conditions Pop-up (must accept before app loads) ----
if "agreed" not in st.session_state:
    st.session_state.agreed = False

if not st.session_state.agreed:
    st.markdown("<h2 style='text-align:center;'>📜 Terms & Conditions</h2>", unsafe_allow_html=True)
    st.info("Scroll through the terms below and then click Agree to proceed:")

    terms_text = """
1. This is a demo prototype; not real-time authenticated GPS/CCTV.
2. All data is simulated or uploaded by the user and not real-time.
3. Educational & hackathon purposes only.
4. We do not take responsibility for misuse.
5. The demo may include dummy users, random coordinates, and AI-generated photos.
6. No personal data is collected or stored in this demo system.
7. Alerts, zones, and suspect database shown are mock data.
8. Do not rely on this prototype for security, safety, or emergency use.
9. Features may not be stable; errors may occur.
10. Future development could integrate authentication, CCTV feeds, and real-time GPS devices.
11. Accuracy is limited; false positives/negatives may occur.
12. Face detection demo is only for concept showcase.
13. Not optimized for production use.
14. Liability is disclaimed for any damages or consequences.
15. Ethical use is mandatory; misuse is discouraged.
16. Do not deploy this demo in real environments.
17. Compliance with laws is your responsibility.
18. Any modifications should retain these disclaimers.
19. If you disagree with terms, close the demo immediately.
20. You must accept these terms to proceed.
    """

    scrolled = st.text_area("Terms and Conditions (read before proceeding)", terms_text, height=280)

    # Simple heuristic: enable button when the user has the full text in the text area
    if scrolled.strip().endswith("proceed."):
        if st.button("✅ I Agree and Proceed"):
            st.session_state.agreed = True
            st.experimental_rerun()
    else:
        st.button("✅ I Agree and Proceed", disabled=True)

    st.stop()

# ---- Load logo (relative path) ----
# Make sure logo.jpg is in the repo root or adjust path to /images/logo.jpg
LOGO_PATH = "logo.jpg"
if not os.path.exists(LOGO_PATH):
    st.warning(f"Logo not found at {LOGO_PATH}. Place logo.jpg in the repo root.")
    team_logo = None
else:
    team_logo = Image.open(LOGO_PATH)

team_name = "💀 Deadsec — Presenting 💀"

# ---- Top header with logo ----
col1, col2 = st.columns([1, 3])
with col1:
    if team_logo:
        st.image(team_logo, width=180)
with col2:
    st.markdown(
        f"""
        <div style="font-family: 'Courier New', Courier, monospace; font-size:32px; font-weight:700;">
            {team_name}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<h1 style='text-align:center; margin-top:10px;'><u>🚨 Geo-Fencing Alert System (DEMO)</u></h1>", unsafe_allow_html=True)

# ---- App styles ----
st.markdown(
    """
    <style>
    .stApp { max-width: 1350px; margin: auto; }
    .flash { animation: blinker 1s linear infinite; color:#d90429; font-weight:700; }
    @keyframes blinker { 50% { opacity: 0; } }
    .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; }
    .badge-low { background:#e8f5e9; color:#2e7d32; }
    .badge-medium { background:#fff8e1; color:#f57f17; }
    .badge-high { background:#ffebee; color:#c62828; }
    .card { border:1px solid #eee; border-radius:12px; padding:10px; margin-bottom:8px; }
    .small { font-size:12px; color:#666; }
    .alert-enter { background:#ffebee; padding:8px; border-radius:8px; margin:4px 0; color:#c62828; }
    .alert-exit  { background:#fff8e1; padding:8px; border-radius:8px; margin:4px 0; color:#f57f17; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# Session-state defaults
# --------------------------
def _init_people() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": "U1", "name": "Asha", "lat": 19.0760, "lon": 72.8777, "risk_level": "Medium", "photo_url": "https://i.pravatar.cc/80?img=5"},
            {"id": "U2", "name": "Rahul", "lat": 28.6139, "lon": 77.2090, "risk_level": "High", "photo_url": "https://i.pravatar.cc/80?img=11"},
        ]
    )


if "people" not in st.session_state:
    st.session_state.people = _init_people()

if "zones" not in st.session_state:
    st.session_state.zones = [
        {"name": "Mumbai Square", "polygon": [[19.07, 72.86], [19.07, 72.90], [19.05, 72.90], [19.05, 72.86]], "active": True},
        {"name": "Delhi Square", "polygon": [[28.62, 77.20], [28.62, 77.22], [28.60, 77.22], [28.60, 77.20]], "active": True},
    ]

if "alerts" not in st.session_state:
    st.session_state.alerts: List[Dict] = []
if "last_zone_map" not in st.session_state:
    st.session_state.last_zone_map: Dict[str, str] = {}
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = True
if "last_flash" not in st.session_state:
    st.session_state.last_flash: str | None = None

# --------------------------
# Helpers
# --------------------------
def risk_badge(risk: str) -> str:
    risk = (risk or "Medium").strip().lower()
    if risk.startswith("h"):
        return f"<span class='badge badge-high'>Risk: High</span>"
    if risk.startswith("l"):
        return f"<span class='badge badge-low'>Risk: Low</span>"
    return f"<span class='badge badge-medium'>Risk: Medium</span>"


def point_in_zones(lat: float, lon: float, zones: List[Dict]) -> Tuple[str, bool]:
    p = Point(lon, lat)
    for z in zones:
        if not z.get("active", True):
            continue
        poly = Polygon([(lo, la) for la, lo in z["polygon"]])
        if poly.contains(p):
            return z["name"], True
    return "", False


def jitter(lat: float, lon: float, scale: float) -> Tuple[float, float]:
    return float(lat + np.random.uniform(-scale, scale)), float(lon + np.random.uniform(-scale, scale))


def simulate_and_update(people: pd.DataFrame, demo: bool, speed: float) -> pd.DataFrame:
    people = people.copy()
    for i, r in people.iterrows():
        lat, lon = (jitter(float(r["lat"]), float(r["lon"]), scale=speed) if demo else (r["lat"], r["lon"]))
        zone_name, restricted = point_in_zones(lat, lon, st.session_state.zones)
        status = "RESTRICTED" if restricted else "OK"
        last_zone = st.session_state.last_zone_map.get(r["id"], "")

        if restricted and zone_name != last_zone:
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            st.session_state.alerts.insert(0, {"ts": ts, "id": r["id"], "name": r["name"], "zone": zone_name, "event": "ENTERED"})
            st.session_state.last_flash = f"{ts} — {r['name']} ({r['id']}) ENTERED {zone_name}"
        elif (not restricted) and last_zone != "":
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            st.session_state.alerts.insert(0, {"ts": ts, "id": r["id"], "name": r["name"], "zone": last_zone, "event": "EXITED"})

        st.session_state.last_zone_map[r["id"]] = zone_name if restricted else ""
        people.at[i, "lat"] = lat
        people.at[i, "lon"] = lon
        people.at[i, "status"] = status
        people.at[i, "zone"] = zone_name
        people.at[i, "last_update"] = datetime.utcnow().isoformat()
    return people


# --------------------------
# Sidebar controls
# --------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    st.session_state.demo_mode = st.checkbox("Demo mode (simulate movement)", value=st.session_state.demo_mode)
    demo_speed = st.slider("Demo speed", 1, 10, 3)
    auto_refresh = st.checkbox("Auto refresh", value=False)
    refresh_seconds = st.slider("Refresh every (sec)", 1, 15, 6)

    st.markdown("---")
    with st.expander("📂 Upload People CSV"):
        st.caption("CSV needs: id,name,lat,lon | optional: photo_url,risk_level")
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            try:
                dfu = pd.read_csv(up)
                if not {"id", "name", "lat", "lon"}.issubset(dfu.columns):
                    st.error("CSV missing required cols")
                else:
                    dfu = dfu.astype({"lat": float, "lon": float})
                    dfu["risk_level"] = dfu.get("risk_level", "Medium")
                    dfu["photo_url"] = dfu.get("photo_url", "https://i.pravatar.cc/80")
                    st.session_state.people = dfu
                    st.success("Loaded people from CSV")
            except Exception as e:
                st.error(f"CSV error: {e}")

    with st.expander("🗺️ Manage Zones"):
        rm_idx = None
        for idx, z in enumerate(st.session_state.zones):
            cols = st.columns([3, 1])
            with cols[0]:
                st.session_state.zones[idx]["active"] = st.checkbox(z["name"], value=z.get("active", True), key=f"zone_{idx}")
            with cols[1]:
                if st.button("Remove", key=f"rm_{idx}"):
                    rm_idx = idx
        if rm_idx is not None:
            st.session_state.zones.pop(rm_idx)
            st.experimental_set_query_params(_t=time.time())
            st.success("Zone removed")
            st.stop()

        new_name = st.text_input("New Zone name")
        new_poly = st.text_area("Polygon lat,lon (1 per line)", "19.07,72.86\n19.07,72.90\n19.05,72.90\n19.05,72.86")
        if st.button("Add zone"):
            try:
                pts = [[float(a), float(b)] for a, b in (line.split(",") for line in new_poly.strip().splitlines())]
                if len(pts) < 3:
                    st.warning("Need ≥3 points")
                else:
                    st.session_state.zones.append({"name": new_name or f"Zone {len(st.session_state.zones)+1}", "polygon": pts, "active": True})
                    st.success("Zone added")
            except Exception as e:
                st.error(f"Add zone failed: {e}")

    with st.expander("👤 Suspect Profiles"):
        for _, row in st.session_state.people.iterrows():
            st.markdown(
                f"<div class='card'><div style='display:flex;gap:10px;align-items:center;'><img src='{row.get('photo_url')}' width='56'/><div><b>{row['name']}</b> <span class='small'>({row['id']})</span><br>{risk_badge(row.get('risk_level'))}</div></div></div>",
                unsafe_allow_html=True,
            )

# --------------------------
# Update positions
# --------------------------
st.session_state.people = simulate_and_update(st.session_state.people, st.session_state.demo_mode, speed=demo_speed * 0.0005)

# --------------------------
# Tabs layout (single set)
# --------------------------
tab1, tab2, tab3 = st.tabs(["🗺️ Map & Tracking", "🚨 Alerts", "📸 Face Detection Demo"])

with tab1:
    if st.session_state.last_flash:
        st.markdown(f"<div class='flash'>🚨 {st.session_state.last_flash}</div>", unsafe_allow_html=True)

    people_df = st.session_state.people.copy()
    people_df["color"] = people_df.apply(lambda r: ([255, 0, 0] if r.get("status") == "RESTRICTED" else ([255, 120, 0] if str(r.get("risk_level")).lower().startswith("h") else [0, 128, 255])), axis=1)

    poly_data = [{"polygon": [[lo, la] for la, lo in z["polygon"]], "name": z["name"]} for z in st.session_state.zones if z.get("active", True)]

    layers = []
    if poly_data:
        layers.append(pdk.Layer("PolygonLayer", data=poly_data, get_polygon="polygon", get_fill_color=[240, 80, 80, 60], get_line_color=[200, 30, 30]))
    layers.append(pdk.Layer("ScatterplotLayer", data=people_df.to_dict(orient="records"), get_position="[lon,lat]", get_radius=80, get_fill_color="color"))
    layers.append(pdk.Layer("TextLayer", data=people_df.to_dict(orient="records"), get_position="[lon,lat]", get_text="name", get_size=14, get_color=[20, 20, 20], get_alignment_baseline="top"))

    if len(people_df) > 0:
        view_lat, view_lon, zoom = float(people_df.lat.mean()), float(people_df.lon.mean()), 10
    else:
        view_lat, view_lon, zoom = 22.97, 78.65, 4

    st.pydeck_chart(pdk.Deck(map_style="road", initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=zoom), layers=layers, tooltip={"text": "{name} | {status}"}), use_container_width=True)
    st.dataframe(people_df[["id", "name", "lat", "lon", "status", "zone", "risk_level", "last_update"]].reset_index(drop=True))

with tab2:
    if not st.session_state.alerts:
        st.success("✅ No alerts.")
    else:
        for a in st.session_state.alerts[:50]:
            css = "alert-enter" if a["event"] == "ENTERED" else "alert-exit"
            st.markdown(f"<div class='{css}'> {a['ts']} — <b>{a['name']}</b> ({a['id']}) {a['event']} <i>{a['zone']}</i></div>", unsafe_allow_html=True)
        st.download_button("Download Alerts CSV", pd.DataFrame(st.session_state.alerts).to_csv(index=False), file_name="alerts.csv")

with tab3:
    st.info("Demo face detection only (no recognition). Uses OpenCV Haar cascade.")
    face_demo = st.checkbox("Enable face detection", value=False)
    if face_demo:
        cam_img = st.camera_input("Camera")
        if cam_img and PIL_OK and OPENCV_OK:
            frame = np.array(Image.open(cam_img))
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            st.image(frame, caption=f"Faces: {len(faces)}")
        elif face_demo:
            st.warning("Camera face demo requires both PIL and OpenCV installed in the environment.")

# --------------------------
# Auto refresh logic
# --------------------------
if auto_refresh:
    # Use a timestamp to only rerun after refresh_seconds have passed
    last = st.session_state.get("_last_auto", 0)
    if time.time() - last > refresh_seconds:
        st.session_state["_last_auto"] = time.time()
        st.experimental_set_query_params(_t=time.time())
        st.experimental_rerun()

