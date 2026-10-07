"""
DiDi Nova: Configuration & Modular Loading Engine
Automatically loads JSON dictionaries and maintains the BigQuery models and SQL.
"""

import json
import os
import streamlit as st

# =====================================================================
# 1. DYNAMIC JSON LOADER
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DICT_DIR = os.path.join(BASE_DIR, 'dictionaries')

@st.cache_data
def load_json_dictionary(filename):
    filepath = os.path.join(DICT_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        st.error(f"⚠️ Error: No se encontró el archivo '{filename}' en '{DICT_DIR}'.")
        return {}
    except json.JSONDecodeError:
        st.error(f"⚠️ Error de formato: '{filename}' no es un JSON válido. Revisa comas y comillas.")
        return {}

# Load externalized dictionaries dynamically
CITIES_DICT = load_json_dictionary('cities.json')
COORDINATES = load_json_dictionary('coordinates.json')
# Structure: {"City": {"Intensity": {"FloodRisk": {"trips": float|null, "calls": float|null, ...}}}}
# null / missing values are treated as 1.0 by nova_app.py
WEATHER_IMPACT_DICT = load_json_dictionary('weather_impact.json')

# =====================================================================
# 2. BIGQUERY MODELS & SQL ARCHITECTURE
# =====================================================================
ARIMA_MODEL = "`valid-sol-477221-e8.nova.trips_arima_baseline`"
GMV_MODEL = "`valid-sol-477221-e8.nova.gmv_arima_baseline`"
CALLS_MODEL = "`valid-sol-477221-e8.nova.calls_arima_baseline`"
EYEBALLS_MODEL = "`valid-sol-477221-e8.nova.eyeballs_arima_baseline`"
SUPPLY_MODEL = "`valid-sol-477221-e8.nova.supply_arima_baseline`"
REAL_TABLE = "`didi_db.Daily DB 100268`"
# Burn source of truth (city_id, city_name, date_value, drv_expan_mktp, pax_expan_mktp, gmv). Confirm the dataset name.
BURN_TABLE = "`didi_db.Burn SoT`"

def get_history_query(city_id):
    return f"""
        SELECT DATE(date_value) as date, 
               trips as trips_real, calls as calls_real, eyeballs as eyeballs_real, supply_hours as tsh_real, gmv as gmv_real,
               (trips / NULLIF(calls, 0)) * 100 as cr_real, 
               (calls / NULLIF(eyeballs, 0)) * 100 as ecr_real,
               (supply_hours / NULLIF(eyeballs, 0)) as sdr_real
        FROM {REAL_TABLE}
        WHERE CAST(city_id AS STRING) = '{city_id}' AND product = 'Managed Products'
        ORDER BY date_value DESC LIMIT 45
    """

def get_forecast_query(city_id):
    return f"""
        WITH trips AS (SELECT DATE(forecast_timestamp) as date, forecast_value as trips_forecast FROM ML.FORECAST(MODEL {ARIMA_MODEL}, STRUCT(14 AS horizon)) WHERE CAST(city_id AS STRING) = '{city_id}'),
        gmv AS (SELECT DATE(forecast_timestamp) as date, forecast_value as gmv_forecast FROM ML.FORECAST(MODEL {GMV_MODEL}, STRUCT(14 AS horizon)) WHERE CAST(city_id AS STRING) = '{city_id}'),
        calls AS (SELECT DATE(forecast_timestamp) as date, forecast_value as calls_forecast FROM ML.FORECAST(MODEL {CALLS_MODEL}, STRUCT(14 AS horizon)) WHERE CAST(city_id AS STRING) = '{city_id}'),
        eyeballs AS (SELECT DATE(forecast_timestamp) as date, forecast_value as eyeballs_forecast FROM ML.FORECAST(MODEL {EYEBALLS_MODEL}, STRUCT(14 AS horizon)) WHERE CAST(city_id AS STRING) = '{city_id}'),
        supply AS (SELECT DATE(forecast_timestamp) as date, forecast_value as tsh_forecast FROM ML.FORECAST(MODEL {SUPPLY_MODEL}, STRUCT(14 AS horizon)) WHERE CAST(city_id AS STRING) = '{city_id}')
        
        SELECT t.date, EXTRACT(ISOWEEK FROM t.date) as iso_week, EXTRACT(DAYOFWEEK FROM t.date) as dow, 
            t.trips_forecast, COALESCE(g.gmv_forecast, 150000) as gmv_forecast, c.calls_forecast, e.eyeballs_forecast, s.tsh_forecast,
            (t.trips_forecast / NULLIF(c.calls_forecast, 0)) * 100 as cr_forecast, (c.calls_forecast / NULLIF(e.eyeballs_forecast, 0)) * 100 as ecr_forecast,
            (s.tsh_forecast / NULLIF(e.eyeballs_forecast, 0)) as sdr_forecast
        FROM trips t LEFT JOIN gmv g ON t.date = g.date LEFT JOIN calls c ON t.date = c.date LEFT JOIN eyeballs e ON t.date = e.date LEFT JOIN supply s ON t.date = s.date
        ORDER BY t.date ASC
    """


def get_burn_query(city_id, start_date, end_date):
    """Raw money burned per day (DRV = drv_expan_mktp, PAX = pax_expan_mktp) and the GMV used to express it as a %."""
    return f"""
        SELECT DATE(date_value) AS date,
               SUM(drv_expan_mktp) AS drv_burn, SUM(pax_expan_mktp) AS pax_burn, SUM(gmv) AS gmv
        FROM {BURN_TABLE}
        WHERE CAST(city_id AS STRING) = '{city_id}'
          AND DATE(date_value) BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY 1
        ORDER BY 1
    """


def get_calibration_query(city_id, days=150):
    """Daily funnel joined with daily burn: used to learn how strongly CR reacts to DRV burn and ECR to PAX burn."""
    return f"""
        WITH d AS (
            SELECT DATE(date_value) AS date, SUM(trips) AS trips, SUM(calls) AS calls, SUM(eyeballs) AS eyeballs
            FROM {REAL_TABLE}
            WHERE CAST(city_id AS STRING) = '{city_id}' AND product = 'Managed Products'
              AND DATE(date_value) >= DATE_SUB(CURRENT_DATE(), INTERVAL {int(days)} DAY)
            GROUP BY 1
        ),
        b AS (
            SELECT DATE(date_value) AS date, SUM(drv_expan_mktp) AS drv_burn, SUM(pax_expan_mktp) AS pax_burn, SUM(gmv) AS b_gmv
            FROM {BURN_TABLE}
            WHERE CAST(city_id AS STRING) = '{city_id}'
              AND DATE(date_value) >= DATE_SUB(CURRENT_DATE(), INTERVAL {int(days)} DAY)
            GROUP BY 1
        )
        SELECT d.date, d.trips, d.calls, d.eyeballs, b.drv_burn, b.pax_burn, b.b_gmv
        FROM d JOIN b ON d.date = b.date
        ORDER BY d.date
    """