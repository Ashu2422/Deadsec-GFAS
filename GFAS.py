from __future__ import annotations
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit import st_folium

import folium
from streamlit_folium 
from PIL import Image

# ---- Page Config ----
st.set_page_config(page_title=" Geo-Fencing Alert System", layout="wide")

# ---- Terms & Conditions Pop-up ----
if "agreed" not in st.session_state:
    st.session_state.agreed = False

if not st.session_state.agreed:
    st.markdown("<h2 style='text-align:center;'>📜 Terms & Conditions</h2>", unsafe_allow_html=True)
    st.info("Scroll through the terms below and then click Agree to proceed:")

    # Create scrollable area
    terms_text = """
1. As we don’t have real-time authentication of GPS tracking, local CCTV, or traffic signal CCTV, this is just a demo prototype.
2. All data displayed is simulated or uploaded by the user and does not represent real-time tracking.
3. This system is designed for educational & hackathon purposes only.
4. We do not take responsibility for misuse or misinterpretation of the demo system.
5. The demo may include dummy users, random coordinates, and AI-generated photos.
6. No personal data is collected or stored in this demo system.
7. Alerts, zones, and suspect database shown are only mock data.
8. Do not rely on this prototype for any security, safety, or emergency use.
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

    # Display scrollable text area
    scrolled = st.text_area("Terms and Conditions", terms_text, height=250)

    # Enable Agree button only if user scrolled (heuristic: check if last line visible)
    if scrolled.strip().endswith("proceed."):
        if st.button("✅ I Agree and Proceed"):
            st.session_state.agreed = True
            st.rerun()
    else:
        st.button("✅ I Agree and Proceed", disabled=True)

    st.stop()

# ---- Logo and Title ----
team_logo = Image.open(r"C:\Users\Ashish\OneDrive\Attachments\Desktop\hackathon\logo.jpg")
team_name = "💀Deadsec<br>Presenting💀"

col1, col2 = st.columns([1, 3])
with col1:
    st.image(team_logo, width=200)
with col2:
    st.markdown(f"""
    <style>
    .wipe-container {{
        overflow: hidden;
        white-space: nowrap;
        display: inline-block;
        font-size: 80px;
        font-weight: bold;
        font-family: 'Courier New', Courier, monospace;
        animation: wipe 5s forwards;
        color: #000000;
    }}

    @keyframes wipe {{
        0% {{ width: 0; }}
        100% {{ width: 100%; }}
    }}
    </style>

    <div class="wipe-container">{team_name}</div>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; margin-top:20px;'><u><i><b>🚨 Geo-Fencing Alert System(DEMO)</b></i></u></h1>", unsafe_allow_html=True)

# (Rest of your original Geo-Fencing code continues below here…)

# ---- Dark Mode Toggle ----
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

mode = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
st.session_state.dark_mode = mode

if mode:
    dark_theme_css = """
    <style>
        body { background-color: #121212; color: #e0e0e0; }
        .stApp { background-color: #121212; color: #e0e0e0; }
        .st-emotion-cache-1wbqy5l { background-color: #1e1e1e !important; }
        h1, h2, h3, h4, h5, h6, p, label { color: #e0e0e0 !important; }
        .st-emotion-cache-79elbk { background-color: #1e1e1e !important; }
    </style>
    """
    st.markdown(dark_theme_css, unsafe_allow_html=True)

# ---- Sidebar ----
st.sidebar.title("⚙️ Controls")
refresh_rate = st.sidebar.slider("Auto-refresh rate (sec)", 5, 60, 15)

with st.sidebar.expander("📂 Suspect Database"):
    st.write("Upload or manage suspect profiles.")
    uploaded_file = st.file_uploader("Upload suspect CSV", type=["csv"])

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["📍 Map View", "🚨 Alerts Feed", "📊 Suspect Database"])

# ---- Dummy Data ----
suspects = pd.DataFrame({
    "Name": ["John Doe", "Ali Khan", "Maria Silva"],
    "Risk": ["High", "Medium", "Low"],
    "Last Seen": ["Zone A", "Zone B", "Zone C"]
})

alerts = [
    {"time": "2025-08-25 18:30", "name": "John Doe", "zone": "Restricted Area 1", "risk": "High"},
    {"time": "2025-08-25 18:40", "name": "Ali Khan", "zone": "Restricted Area 2", "risk": "Medium"},
]

# ---- Map Tab ----
with tab1:
    st.subheader("Live Tracking Map")
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    for i, row in suspects.iterrows():
        folium.Marker(
            location=[20 + i, 78 + i],
            popup=f"{row['Name']} ({row['Risk']})",
            tooltip=row['Last Seen'],
            icon=folium.Icon(color="red" if row['Risk']=="High" else "blue")
        ).add_to(m)
    st_folium(m, width=900, height=500)

# ---- Alerts Feed Tab ----
with tab2:
    st.subheader("🚨 Alerts Feed")
    for alert in alerts:
        st.markdown(f"""
        <div style='padding:10px; border-radius:10px; margin:5px; 
        background-color:{'#3a3a3a' if mode else '#f8d7da'};'>
        <b>{alert['time']}</b> - <span style='color:red'>{alert['name']}</span> entered <b>{alert['zone']}</b> | Risk: <b>{alert['risk']}</b>
        </div>
        """, unsafe_allow_html=True)
    st.download_button("Download Alerts Log", pd.DataFrame(alerts).to_csv(index=False), "alerts_log.csv")

# ---- Suspect Database Tab ----
with tab3:
    st.subheader("📊 Suspect Database")
    if uploaded_file:
        suspects = pd.read_csv(uploaded_file)
        st.success("Uploaded successfully!")
    st.dataframe(suspects, use_container_width=True)

# ---- Auto Refresh ----
placeholder = st.empty()
for i in range(refresh_rate):
    placeholder.text(f"Refreshing in {refresh_rate-i} sec...")
    time.sleep(1)

from shapely.geometry import Point, Polygon

# Optional deps
try:
    import cv2
    OPENCV_OK = True
except Exception:
    OPENCV_OK = False

try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

# --------------------------
# App config & global styles
# --------------------------
st.set_page_config(page_title="🚨 Geo-Fencing Alert System", layout="wide")
# ---- Page Config ----
st.set_page_config(page_title="Geo-Fencing Alert System", layout="wide")



st.markdown(f"""

""", unsafe_allow_html=True)

# ---- Dark Mode Toggle ----
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


st.markdown("""
<style>
.stApp { max-width: 1350px; margin: auto; }
.flash { animation: blinker 1s linear infinite; color:#d90429; font-weight:700; }
@keyframes blinker { 50% { opacity: 0; } }

.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; }
.badge-low    { background:#e8f5e9; color:#2e7d32; }
.badge-medium { background:#fff8e1; color:#f57f17; }
.badge-high   { background:#ffebee; color:#c62828; }

.card { border:1px solid #eee; border-radius:16px; padding:12px; margin-bottom:10px; }
.card img { border-radius:12px; }
.small { font-size:12px; color:#666; }
.alert-enter { background:#ffebee; padding:8px; border-radius:12px; margin:4px 0; color:#c62828; }
.alert-exit  { background:#fff8e1; padding:8px; border-radius:12px; margin:4px 0; color:#f57f17; }
</style>
""", unsafe_allow_html=True)



# --------------------------
# Session state defaults
# --------------------------
def _init_people() -> pd.DataFrame:
    return pd.DataFrame([
        {"id": "U1", "name": "Asha", "lat": 19.0760, "lon": 72.8777, "risk_level": "Medium", "photo_url": "https://i.pravatar.cc/80?img=5"},
        {"id": "U2", "name": "Rahul", "lat": 28.6139, "lon": 77.2090, "risk_level": "High", "photo_url": "https://i.pravatar.cc/80?img=11"},
    ])

if "people" not in st.session_state:
    st.session_state.people = _init_people()

if "zones" not in st.session_state:
    st.session_state.zones = [
        {"name": "Mumbai Square", "polygon": [[19.07, 72.86],[19.07, 72.90],[19.05, 72.90],[19.05, 72.86]], "active": True},
        {"name": "Delhi Square", "polygon": [[28.62, 77.20],[28.62, 77.22],[28.60, 77.22],[28.60, 77.20]], "active": True},
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
    if risk.startswith("h"): return f"<span class='badge badge-high'>Risk: High</span>"
    if risk.startswith("l"): return f"<span class='badge badge-low'>Risk: Low</span>"
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
        people.at[i, "lat"] = lat; people.at[i, "lon"] = lon
        people.at[i, "status"] = status; people.at[i, "zone"] = zone_name
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
                if not {"id","name","lat","lon"}.issubset(dfu.columns):
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
            cols = st.columns([3,1])
            with cols[0]:
                st.session_state.zones[idx]["active"] = st.checkbox(z["name"], value=z.get("active", True), key=f"zone_{idx}")
            with cols[1]:
                if st.button("Remove", key=f"rm_{idx}"): rm_idx = idx
        if rm_idx is not None:
            st.session_state.zones.pop(rm_idx); st.experimental_set_query_params(_t=time.time()); st.success("Zone removed"); st.stop()

        new_name = st.text_input("New Zone name")
        new_poly = st.text_area("Polygon lat,lon (1 per line)", "19.07,72.86\n19.07,72.90\n19.05,72.90\n19.05,72.86")
        if st.button("Add zone"):
            try:
                pts = [[float(a), float(b)] for a,b in (line.split(",") for line in new_poly.strip().splitlines())]
                if len(pts)<3: st.warning("Need ≥3 points")
                else:
                    st.session_state.zones.append({"name": new_name or f"Zone {len(st.session_state.zones)+1}", "polygon": pts, "active": True})
                    st.success("Zone added")
            except Exception as e: st.error(f"Add zone failed: {e}")

    with st.expander("👤 Suspect Profiles"):
        for _, row in st.session_state.people.iterrows():
            st.markdown(f"<div class='card'><div style='display:flex;gap:10px;align-items:center;'><img src='{row.get('photo_url')}' width='56'/><div><b>{row['name']}</b> <span class='small'>({row['id']})</span><br>{risk_badge(row.get('risk_level'))}</div></div></div>", unsafe_allow_html=True)

# --------------------------
# Update positions
# --------------------------
st.session_state.people = simulate_and_update(st.session_state.people, st.session_state.demo_mode, speed=demo_speed*0.0005)

# --------------------------
# Tabs layout
# --------------------------
tab1, tab2, tab3 = st.tabs(["🗺️ Map & Tracking", "🚨 Alerts", "📸 Face Detection Demo"])

with tab1:
    if st.session_state.last_flash:
        st.markdown(f"<div class='flash'>🚨 {st.session_state.last_flash}</div>", unsafe_allow_html=True)

    people_df = st.session_state.people.copy()
    people_df["color"] = people_df.apply(lambda r: ([255,0,0] if r.get("status")=="RESTRICTED" else ([255,120,0] if str(r.get("risk_level")).lower().startswith("h") else [0,128,255])), axis=1)

    poly_data = [{"polygon": [[lo,la] for la,lo in z["polygon"]], "name": z["name"]} for z in st.session_state.zones if z.get("active",True)]

    layers = []
    if poly_data:
        layers.append(pdk.Layer("PolygonLayer", data=poly_data, get_polygon="polygon", get_fill_color=[240,80,80,60], get_line_color=[200,30,30]))
    layers.append(pdk.Layer("ScatterplotLayer", data=people_df.to_dict(orient="records"), get_position="[lon,lat]", get_radius=80, get_fill_color="color"))
    layers.append(pdk.Layer("TextLayer", data=people_df.to_dict(orient="records"), get_position="[lon,lat]", get_text="name", get_size=14, get_color=[20,20,20], get_alignment_baseline="top"))

    if len(people_df)>0: view_lat,view_lon,zoom=float(people_df.lat.mean()),float(people_df.lon.mean()),10
    else: view_lat,view_lon,zoom=22.97,78.65,4

    st.pydeck_chart(pdk.Deck(map_style="road", initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=zoom), layers=layers, tooltip={"text":"{name} | {status}"}), use_container_width=True)
    st.dataframe(people_df[["id","name","lat","lon","status","zone","risk_level","last_update"]].reset_index(drop=True))

with tab2:
    if not st.session_state.alerts: st.success("✅ No alerts.")
    else:
        for a in st.session_state.alerts[:50]:
            css = "alert-enter" if a["event"]=="ENTERED" else "alert-exit"
            st.markdown(f"<div class='{css}'> {a['ts']} — <b>{a['name']}</b> ({a['id']}) {a['event']} <i>{a['zone']}</i></div>", unsafe_allow_html=True)
        st.download_button("Download Alerts CSV", pd.DataFrame(st.session_state.alerts).to_csv(index=False), "alerts.csv")

with tab3:
    st.info("Demo face detection only (no recognition). Uses OpenCV Haar cascade.")
    face_demo = st.checkbox("Enable face detection", value=False)
    if face_demo:
        cam_img = st.camera_input("Camera")
        if cam_img and PIL_OK and OPENCV_OK:
            frame = np.array(Image.open(cam_img))
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml").detectMultiScale(gray,1.1,5)
            for (x,y,w,h) in faces: cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            st.image(frame, caption=f"Faces: {len(faces)}")

# --------------------------
# Auto refresh
# --------------------------
if auto_refresh:
    if time.time()-st.session_state.get("_last_auto",0) > refresh_seconds:
        st.session_state["_last_auto"] = time.time()
        st.experimental_set_query_params(_t=time.time())
        st.rerun()

