# Streamlit Deployment Guide
## Oil Dependency & Price Shock Transmission Dashboard

### Prerequisites
```bash
pip install streamlit plotly pandas numpy
```

### Run Locally
```bash
cd "/Volumes/T7/DS Projects/Oil Dependency Price Transmission"
streamlit run app.py
```
The dashboard opens at `http://localhost:8501`.

### Deploy to Streamlit Community Cloud (Free)
1. Push your project to a **GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **New App** → select your repo → set main file to `app.py`.
4. Click **Deploy**. Your dashboard is now live with a public URL.

### Deploy to Streamlit on a Custom Server
```bash
# Install on any Linux/Mac server
pip install streamlit

# Run in background
nohup streamlit run app.py --server.port 8501 --server.headless true &
```

### Project Structure for Deployment
```
Oil Dependency Price Transmission/
├── app.py                    ← Streamlit dashboard
├── Data/
│   └── raw_market_data.csv   ← Must exist before running
├── Scripts/
│   └── fetch_real_data.py    ← Run this first to generate data
├── Models/                   ← MATLAB analysis (offline)
└── requirements.txt          ← Python dependencies
```

### requirements.txt
```
streamlit
plotly
pandas
numpy
yfinance
```
