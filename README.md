# DiDi Nova - Allocation Engine OS
Streamlit app: forecast (ARIMA + bias + anomalies + weather) and budget allocation.

Files: `nova_app.py` (app), `nova_config.py` (queries / dictionaries).
Local run: put `bq_key.json` next to `nova_app.py`, then `streamlit run nova_app.py`.
Cloud: add the service account under `[gcp_service_account]` in the app's Secrets. Never commit `bq_key.json`.
