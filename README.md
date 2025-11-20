# Farm-to-Market Route Optimization

Interactive Streamlit web app for optimizing farm-to-market transport routes with real-time weather integration, vehicle pooling, and cost estimation for Dehradun region.

## Features

1. **Real Road Routing** — Uses MapMyIndia & OSRM APIs to fetch actual road routes with realistic geometry
2. **Weather-Based Rerouting** — Analyzes weather conditions (precipitation, wind, temperature) and prioritizes routes by safety scores
3. **Vehicle Pooling/Clustering** — Create or join transport pools to share vehicle capacity and reduce costs
4. **Cost Estimation** — Calculate solo vs pooled transport costs with savings breakdown
5. **Pool Booking** — Confirm participation in pools with one-click booking (session-based)

## Project Structure

```
d:\PY_PBL\
├── .git/                 # Git repository
├── .gitignore            # Git ignore rules
└── Farm2Road/
    ├── main.py           # Main Streamlit application (all-in-one)
    ├── requirements.txt  # Python dependencies
    ├── README.md         # This file
    ├── .streamlit/       # Streamlit config (secrets.toml)
    └── cache/            # API response cache (runtime)
```

## Libraries (one-line purpose each)

- streamlit — build interactive web UI and sidebar controls
- folium / streamlit-folium — render interactive maps inside Streamlit
- requests — call external APIs (MapMyIndia, OSRM, Open-Meteo)
- matplotlib / seaborn — generate charts and analytics visuals
- pandas — read/format table and CSV data
- math / python stdlib — Haversine distance, geometry helpers

## How to run locally (short)

1. Open PowerShell, navigate to project root:

```powershell
cd D:\PY_PBL
```

2. Create venv & install deps:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r Farm2Road\requirements.txt
```

3. Start the app:

```powershell
streamlit run Farm2Road\main.py
```

4. Open the Local URL shown by Streamlit in your browser.

## Notes

- All application logic is in `main.py` (single-file architecture).
- MapMyIndia API key is in `.streamlit/secrets.toml`; replace with your own key for production.
- Pool bookings are stored in Streamlit session state only (no database or payment integration).
