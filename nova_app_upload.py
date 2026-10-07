"""
DiDi Nova - Allocation Engine OS (Enterprise Edition)
=====================================================
Page flow : Command Center (top bar + popovers, no sidebar) -> KPI cards -> Alert Center
            -> Forecast charts -> Weather detail -> Allocation Engine (only if budget > 0)
Engines   : Rolling Bias Corrector | Weather tree (7 Packages) + Auto-Calibration | Universal Anomaly Engine (WoW / Wo2W)
            | Allocation Engine v4 (marginal-return optimizer, explainable)
Units     : The UI always speaks in %. All maths run on real numbers (money, fractions); conversion happens only at
            pct_to_frac / frac_to_pct and plan_week_budget.
Mode      : Decoupled. The Allocation Engine only runs when the weekly budget is above 0.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import copy
import altair as alt
from datetime import timedelta, datetime
import requests
from google.cloud import bigquery

# Import externalized configs
from nova_config import (
    CITIES_DICT, COORDINATES, WEATHER_IMPACT_DICT, REAL_TABLE,
    get_history_query, get_forecast_query, get_burn_query, get_calibration_query
)

# =====================================================================
# MODULE 1: PAGE CONFIG, STYLES & BIGQUERY CLIENT
# =====================================================================
st.set_page_config(page_title="DiDi Nova - Master", layout="wide", page_icon="🧠", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* No sidebar: the whole control surface lives in the Command Center */
    section[data-testid="stSidebar"], div[data-testid="collapsedControl"], div[data-testid="stSidebarCollapsedControl"] { display: none !important; }
    .block-container { padding-top: 2rem; }

    div[data-testid="metric-container"] { background-color: #1E1E1E; border-radius: 10px; padding: 15px; box-shadow: 2px 2px 10px rgba(0,0,0,0.2); }
    div.stAlert { border-radius: 10px; }

    /* Command Center: a separate console panel at the top of the page */
    .st-key-nova_cc,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.nova-cc-marker) {
        background-color: rgba(28, 134, 238, 0.07);
        border: 1px solid rgba(28, 134, 238, 0.45);
        border-top: 4px solid #1C86EE;
        border-radius: 8px;
        padding-bottom: 4px;
    }
    /* Top nav: big, even buttons */
    div[data-testid="stPopover"] button {
        width: 100%; min-height: 3.2rem; font-size: 1.02rem; font-weight: 700; border-radius: 10px;
        border: 1px solid rgba(28, 134, 238, 0.35);
    }
    div[data-testid="stPopover"] button:hover { border-color: #1C86EE; background: rgba(28, 134, 238, 0.10); }
    .nova-cc-title { font-size: 1.7rem; font-weight: 800; line-height: 1.1; margin-top: 0.15rem; }
    .nova-cc-sub { opacity: 0.7; font-size: 0.82rem; }
    .nova-status { font-size: 0.85rem; opacity: 0.85; margin: 0.35rem 0 0.9rem 0; line-height: 1.5; }

    /* Tables inside Alert Center banners */
    table.nova-tbl { width: 100%; border-collapse: collapse; font-size: 0.84rem; margin-top: 6px; }
    table.nova-tbl th { text-align: left; font-weight: 600; opacity: 0.7; padding: 3px 10px 3px 0; border-bottom: 1px solid rgba(128,128,128,0.35); }
    table.nova-tbl td { padding: 3px 10px 3px 0; }

    /* Alert Center: severity is encoded in the left rule */
    .nova-alert { border-left: 5px solid; border-radius: 6px; padding: 10px 16px; margin-bottom: 10px; }
    .nova-alert .nova-alert-title { font-weight: 700; }
    .nova-alert .nova-alert-body { font-size: 0.9rem; opacity: 0.9; margin-top: 3px; }
    .nova-alert.critical { background: rgba(220, 20, 60, 0.12); border-color: #DC143C; }
    .nova-alert.warning { background: rgba(255, 165, 0, 0.12); border-color: #FFA500; }
    .nova-alert.info { background: rgba(28, 134, 238, 0.12); border-color: #1C86EE; }
    .nova-alert.ok { background: rgba(46, 160, 67, 0.12); border-color: #2EA043; }
    </style>
""", unsafe_allow_html=True)

def _make_bq_client():
    """Cloud: service account from st.secrets['gcp_service_account']. Local: bq_key.json next to this file."""
    try:
        if "gcp_service_account" in st.secrets:
            from google.oauth2 import service_account
            info = dict(st.secrets["gcp_service_account"])
            creds = service_account.Credentials.from_service_account_info(info)
            return bigquery.Client(credentials=creds, project=info.get("project_id"))
    except Exception:
        pass  # no secrets configured (local run): fall back to the key file
    if os.path.exists("bq_key.json"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "bq_key.json"
    return bigquery.Client()


client = None
try:
    client = _make_bq_client()
except Exception as e:
    st.error("⚠️ BigQuery credentials not detected. App will run in limited mode.")

# =====================================================================
# MODULE 1.5: PATHS, TUNABLES & STORAGE HELPERS
# =====================================================================
TZ = "America/Mexico_City"
DATA_DIR = "nova_data"
DICT_DIR = "dictionaries"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DICT_DIR, exist_ok=True)

WEATHER_DICT_FILE = os.path.join(DICT_DIR, "weather_impact.json")
OVERRIDES_FILE = os.path.join(DATA_DIR, "scheduled_overrides.json")   # Anomaly Engine (key = future date)
LEDGER_FILE = os.path.join(DATA_DIR, "arima_ledger.json")             # What ARIMA said, per city and date
CALIB_LOG_FILE = os.path.join(DATA_DIR, "calibration_log.json")       # Auto-calibration audit trail

# Volume metrics that every engine multiplies. Ratios (CR, ECR, SDR, ETR) are always re-derived from these.
FUNNEL_METRICS = ['trips', 'calls', 'eyeballs', 'tsh', 'gmv']
METRIC_LABELS = {'trips': 'Trips', 'calls': 'Calls', 'eyeballs': 'Eyeballs', 'tsh': 'Supply Hours', 'gmv': 'GMV'}

# Rolling Bias Corrector
BIAS_WINDOW_DAYS = 7          # Last N closed days compared against what ARIMA predicted
BIAS_MIN_OBS = 3              # Minimum days with a stored ARIMA prediction to trust the bias
BIAS_MIN_CONSISTENCY = 0.70   # Share of days that must err in the same direction ("sustained" bias)
BIAS_CLAMP = (0.85, 1.15)     # Safety rail on the correction factor

# Weather auto-calibration: new = CALIB_KEEP * dict + CALIB_LEARN * real
CALIB_KEEP, CALIB_LEARN = 0.8, 0.2
CALIB_MIN_DIFF = 0.005        # Ignore differences smaller than this
CALIB_REAL_BOUNDS = (0.4, 2.5)  # Ignore absurd real/arima ratios (data glitches)

# Anomaly Engine: multiplier = mean of the reference day's WoW and Wo2W ratios (same weekday, 1 and 2 weeks earlier)
ANOMALY_LOOKBACK_WEEKS = 2
ANOMALY_CLAMP = (0.2, 3.0)
ANOMALY_MAX_DAYS = 30
FORECAST_HORIZON_DAYS = 14    # ML.FORECAST horizon

HEAT_THRESHOLD_C = 35.0


def today_mx():
    return pd.Timestamp.now(tz=TZ).tz_localize(None).normalize()


def load_json(filepath, default=None):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {} if default is None else default


def _json_default(o):
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (pd.Timestamp, datetime)): return o.strftime('%Y-%m-%d')
    raise TypeError(f"Not JSON serializable: {type(o)}")


def save_json(filepath, data):
    """Atomic write: a crash mid-write can never leave a half-written dictionary behind."""
    tmp_path = f"{filepath}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=_json_default)
    os.replace(tmp_path, filepath)


def safe_mult(val, default=1.0):
    """Null-safe multiplier: None / missing / NaN / non-numeric / <= 0  ->  1.0 (neutral)."""
    try:
        if val is None: return default
        v = float(val)
        if not np.isfinite(v) or v <= 0: return default
        return v
    except (TypeError, ValueError):
        return default


def recompute_ratios(df):
    """Re-derive every ratio from the volume columns (call after ANY multiplication)."""
    df['cr_forecast'] = (df['trips_forecast'] / df['calls_forecast'].replace(0, np.nan)) * 100
    df['ecr_forecast'] = (df['calls_forecast'] / df['eyeballs_forecast'].replace(0, np.nan)) * 100
    df['sdr_forecast'] = df['tsh_forecast'] / df['eyeballs_forecast'].replace(0, np.nan)
    df['etr_forecast'] = (df['trips_forecast'] / df['eyeballs_forecast'].replace(0, np.nan)) * 100
    return df


def scale_rows(df, mask, mults):
    for m in FUNNEL_METRICS:
        df.loc[mask, f'{m}_forecast'] = df.loc[mask, f'{m}_forecast'] * mults.get(m, 1.0)


# =====================================================================
# MODULE 2: DATA INGESTION & WEATHER PIPELINE
# =====================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_weather_forecast(lat, lon, city_name):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&daily=temperature_2m_max,precipitation_probability_max,precipitation_sum"
           f"&hourly=precipitation_probability,precipitation,relative_humidity_2m,temperature_2m"
           f"&timezone=America%2FMexico_City&past_days=21&forecast_days=14")
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            df_daily = pd.DataFrame({
                'date': pd.to_datetime(data['daily']['time']),
                'Temp_Max': data['daily']['temperature_2m_max'],
                'rain_prob': data['daily']['precipitation_probability_max'],
                'rain_mm': data['daily']['precipitation_sum']
            })
            df_daily['Temp_Max'] = df_daily['Temp_Max'].ffill().bfill()
            df_daily = df_daily.fillna(0)

            def classify_rain(mm):
                if mm == 0: return 'No Rain', 0
                elif mm <= 2.5: return 'Ligera (>0-2.5mm)', 25
                elif mm <= 7.6: return 'Moderada (2.5-7.6mm)', 50
                elif mm <= 50.0: return 'Fuerte (7.6-50mm)', 75
                else: return 'Tormenta (>50mm)', 100

            res = df_daily['rain_mm'].apply(classify_rain)
            df_daily['intensity_cat'] = [x[0] for x in res]
            df_daily['intensity_score'] = [x[1] for x in res]

            df_hourly = pd.DataFrame({
                'date_time': pd.to_datetime(data['hourly']['time']),
                'rain_prob': data['hourly']['precipitation_probability'],
                'rain_mm': data['hourly']['precipitation'],
                'humidity': data['hourly']['relative_humidity_2m'],
                'temp': data['hourly']['temperature_2m']
            })
            df_hourly['temp'] = df_hourly['temp'].ffill().bfill()
            df_hourly = df_hourly.fillna(0)
            df_hourly['date'] = df_hourly['date_time'].dt.strftime('%Y-%m-%d')
            df_hourly['hour'] = df_hourly['date_time'].dt.hour
            df_hourly['rain_3h_acc'] = df_hourly['rain_mm'].rolling(window=3, min_periods=1).sum()

            def calculate_flood_risk(row):
                if row['rain_mm'] == 0: return 'No'
                threshold_heavy = 15 if city_name == "Monterrey" else 20
                threshold_moderate = 8 if city_name == "Monterrey" else 10
                if row['rain_3h_acc'] >= threshold_heavy or (row['rain_mm'] >= 10 and row['humidity'] >= 85): return 'Yes'
                elif row['rain_3h_acc'] >= threshold_moderate or (row['rain_mm'] >= 5 and row['humidity'] >= 75): return 'Maybe'
                return 'No'

            df_hourly['flood_risk'] = df_hourly.apply(calculate_flood_risk, axis=1)
            risk_map = {'No': 0, 'Maybe': 1, 'Yes': 2}
            risk_inv = {0: 'No', 1: 'Maybe', 2: 'Yes'}
            df_hourly['Risk_Num'] = df_hourly['flood_risk'].map(risk_map)
            max_risk_daily = df_hourly.groupby('date')['Risk_Num'].max().reset_index()
            max_risk_daily['flood_risk'] = max_risk_daily['Risk_Num'].map(risk_inv)

            max_risk_daily['date'] = pd.to_datetime(max_risk_daily['date'])
            df_daily = pd.merge(df_daily, max_risk_daily[['date', 'flood_risk']], on='date', how='left')
            df_daily['flood_risk'] = df_daily['flood_risk'].fillna('No')

            hoy = today_mx()
            df_hourly['is_past'] = df_hourly['date_time'] < hoy

            df_hourly['Chart_Value'] = np.where(df_hourly['is_past'], np.clip(df_hourly['rain_mm'] * 10, 0, 100), df_hourly['rain_prob'])
            df_hourly['Label_Tooltip'] = np.where(df_hourly['is_past'], df_hourly['rain_mm'].round(1).astype(str) + " mm", df_hourly['rain_prob'].astype(int).astype(str) + "%")

            return df_daily, df_hourly
    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        st.error(f"⚠️ Open-Meteo connection error: {e}. Please retry or disable the weather tree.")
    return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=3600)
def get_city_data(city_id):
    if client is None:
        raise RuntimeError("BigQuery client is not available (check bq_key.json).")

    query_history = get_history_query(city_id)
    df_historical = client.query(query_history).to_dataframe()
    df_historical['date'] = pd.to_datetime(df_historical['date'])
    df_historical = df_historical.sort_values('date').reset_index(drop=True)
    df_historical['etr_real'] = (df_historical['trips_real'] / df_historical['eyeballs_real'].replace(0, np.nan)) * 100

    query_forecast = get_forecast_query(city_id)
    df_forecast = client.query(query_forecast).to_dataframe()
    df_forecast['date'] = pd.to_datetime(df_forecast['date'])
    df_forecast['etr_forecast'] = (df_forecast['trips_forecast'] / df_forecast['eyeballs_forecast'].replace(0, np.nan)) * 100

    if not df_forecast.empty:
        days_en = {2: 'Mon', 3: 'Tue', 4: 'Wed', 5: 'Thu', 6: 'Fri', 7: 'Sat', 1: 'Sun'}
        df_forecast['day_name'] = df_forecast['dow'].map(days_en)

    return df_historical, df_forecast


def week_monday(week, ref=None):
    """Monday of ISO week `week` in the year (ISO year of `ref`, +-1) that lands closest to `ref`."""
    ref = today_mx() if ref is None else ref
    iso_y = int(ref.isocalendar()[0])
    best = None
    for y in (iso_y - 1, iso_y, iso_y + 1):
        try:
            mon = pd.Timestamp(datetime.fromisocalendar(y, int(week), 1))
        except ValueError:
            continue
        if best is None or abs((mon - ref).days) < abs((best - ref).days):
            best = mon
    return best


@st.cache_data(ttl=900, show_spinner=False)
def get_past_burn(city_id, start_str, end_str):
    """Money already burned per day, straight from Burn SoT. Raises so the caller can tell the user what failed."""
    if client is None:
        raise RuntimeError("BigQuery client is not available.")
    df = client.query(get_burn_query(city_id, start_str, end_str)).to_dataframe()
    df['date'] = pd.to_datetime(df['date'])
    for c in ('drv_burn', 'pax_burn', 'gmv'):
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    safe_gmv = df['gmv'].where(df['gmv'] > 0)
    df['drv_pct'] = (df['drv_burn'] / safe_gmv * 100).fillna(0.0)
    df['pax_pct'] = (df['pax_burn'] / safe_gmv * 100).fillna(0.0)
    df['total_burn'] = df['drv_burn'] + df['pax_burn']
    df['total_pct'] = df['drv_pct'] + df['pax_pct']
    return df.sort_values('date').reset_index(drop=True)


@st.cache_data(ttl=21600, show_spinner=False)
def get_response_betas(city_id):
    """How strongly CR reacts to DRV burn and ECR to PAX burn in this city. Falls back to neutral priors."""
    try:
        df = client.query(get_calibration_query(city_id)).to_dataframe()
    except Exception:
        df = pd.DataFrame()
    return fit_response_slopes(df)


def unify_data(df_h, df_f):
    h = df_h.copy().rename(columns=lambda x: x.replace('_real', ''))
    f = df_f.copy().rename(columns=lambda x: x.replace('_forecast', ''))
    return pd.concat([h, f])


def calculate_daily_lags(df_historical, target_date_str):
    target_date = pd.to_datetime(target_date_str).normalize()
    actual_day = df_historical[df_historical['date'] == target_date]
    day_wow = df_historical[df_historical['date'] == (target_date - timedelta(days=7))]
    day_wo2w = df_historical[df_historical['date'] == (target_date - timedelta(days=14))]

    if actual_day.empty: return None

    def safe_val(df, col): return df[col].values[0] if not df.empty else 0

    metrics = {
        "trips": {"val": safe_val(actual_day, 'trips_real'), "wow": safe_val(day_wow, 'trips_real'), "wo2w": safe_val(day_wo2w, 'trips_real')},
        "cr": {"val": safe_val(actual_day, 'cr_real'), "wow": safe_val(day_wow, 'cr_real'), "wo2w": safe_val(day_wo2w, 'cr_real')},
        "gmv": {"val": safe_val(actual_day, 'gmv_real'), "wow": safe_val(day_wow, 'gmv_real'), "wo2w": safe_val(day_wo2w, 'gmv_real')}
    }

    for m in metrics:
        v, w, w2 = metrics[m]["val"], metrics[m]["wow"], metrics[m]["wo2w"]
        metrics[m]["delta_wow"] = ((v / w) - 1) * 100 if w > 0 else 0
        metrics[m]["delta_wo2w"] = ((v / w2) - 1) * 100 if w2 > 0 else 0

    return metrics

# =====================================================================
# MODULE 3: WEATHER DICTIONARY (NULL-SAFE) & THE 6 BUSINESS PACKAGES
# =====================================================================
# weather_impact.json layout: {"City": {"Intensity": {"FloodRisk": {"trips": float|null, "calls": float|null, ...}}}}
RAIN_LEVEL_TOKENS = {
    'light': ('ligera', 'light'),
    'moderate': ('moderada', 'moderate'),
    'heavy': ('fuerte', 'heavy', 'strong'),
    'storm': ('tormenta', 'storm'),
}
RAIN_LEVEL_LABEL = {'light': 'Light', 'moderate': 'Moderate', 'heavy': 'Heavy', 'storm': 'Storm'}
FLOOD_ALIASES = {
    'yes': ('yes', 'si', 'sí', 'alto', 'high'),
    'maybe': ('maybe', 'quizá', 'quizas', 'quizás', 'medio', 'medium'),
    'no': ('no', 'none', 'bajo', 'low'),
}
MULT_ALIASES = {
    'trips': ('trips',), 'calls': ('calls',), 'eyeballs': ('eyeballs',),
    'tsh': ('tsh', 'supply_hours', 'supply'), 'gmv': ('gmv',),
}

# Peak windows (hour of the day) and the rain threshold that makes an hour "rainy" for the package tree
MORNING_PEAK = {6, 7, 8, 9, 10}
AFTERNOON_PEAK = {12, 13, 14, 15, 16}
NIGHT_PEAK = {18, 19, 20, 21, 22}
PKG_RAIN_HOUR_MM = 0.2
STORM_DAILY_MM = 50.0

PACKAGE_PRESCRIPTION = {
    0: "- (Business As Usual)",
    1: "- (Business As Usual)",
    2: "🌦️ Active Rain (Controlled Investment)",
    3: "🌧 Active Storm (During/Post Event Push)",
    4: "⛈️ Workday Storm (Time-Shift Morning Peak)",
    5: "🌙 Weekend Storm (Late-Night Push)",
    6: "🚨 Safety Lock (Operational Hazard)",
    7: "🟡 Heavy Rain Off-Peak (Hold & Monitor)",
}


def rain_level(intensity_cat):
    s = str(intensity_cat).lower()
    if 'no rain' in s or s in ('', 'none', 'nan'): return 'none'
    for lvl, tokens in RAIN_LEVEL_TOKENS.items():
        if any(t in s for t in tokens): return lvl
    return 'none'


def find_intensity_key(city_block, intensity_cat):
    if not isinstance(city_block, dict): return None
    label = str(intensity_cat).lower()
    for k in city_block:
        if str(k).lower() == label: return k
    lvl = rain_level(intensity_cat)
    if lvl == 'none': return None
    for k in city_block:
        if any(t in str(k).lower() for t in RAIN_LEVEL_TOKENS[lvl]): return k
    return None


def find_flood_key(level_block, flood_risk):
    if not isinstance(level_block, dict): return None
    label = str(flood_risk).strip().lower()
    keys = [(k, str(k).strip().lower()) for k in level_block]
    for k, kl in keys:
        if kl == label: return k
    aliases = FLOOD_ALIASES.get(label, ())
    for k, kl in keys:
        if kl in aliases: return k
    for k, kl in keys:
        if kl.startswith(label): return k
    return None


def _has_metric_keys(d):
    return isinstance(d, dict) and ('multipliers' in d or any(a in d for al in MULT_ALIASES.values() for a in al))


def resolve_weather_node(city_block, intensity_cat, flood_risk):
    """Walks City -> Intensity -> FloodRisk. Returns the multiplier node (a dict) or None. Never raises."""
    lvl_key = find_intensity_key(city_block, intensity_cat)
    if lvl_key is None: return None
    lvl_node = city_block.get(lvl_key)
    if not isinstance(lvl_node, dict): return None
    if _has_metric_keys(lvl_node): return lvl_node  # flat layout (no flood layer)
    flood_key = find_flood_key(lvl_node, flood_risk)
    if flood_key is None: return None
    node = lvl_node.get(flood_key)
    return node if isinstance(node, dict) else None


def mult_source(node):
    """Where the numbers live (supports both flat nodes and legacy {'multipliers': {...}} nodes)."""
    if isinstance(node, dict) and isinstance(node.get('multipliers'), dict):
        return node['multipliers']
    return node


def read_weather_multipliers(node):
    """STRICT null handling: missing node / missing key / null value -> 1.0 for every metric."""
    src = mult_source(node)
    out = {}
    for m, aliases in MULT_ALIASES.items():
        raw = None
        if isinstance(src, dict):
            for a in aliases:
                if src.get(a) is not None:
                    raw = src.get(a)
                    break
        out[m] = safe_mult(raw)
    return out


def load_weather_dict():
    w = load_json(WEATHER_DICT_FILE)
    if not isinstance(w, dict) or not w:
        w = copy.deepcopy(WEATHER_IMPACT_DICT) if isinstance(WEATHER_IMPACT_DICT, dict) else {}
    return w


CITY_ALIASES = {'cdmx': 'mexicocity', 'ciudaddemexico': 'mexicocity', 'df': 'mexicocity'}


def _norm_city(name):
    import unicodedata
    s = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode().lower()
    s = ''.join(ch for ch in s if ch.isalnum())
    return CITY_ALIASES.get(s, s)


def city_key(w_dict, city_name):
    """Key of the dictionary that belongs to this city, ignoring accents and aliases (Cancún = Cancun, CDMX = Mexico City). None if absent."""
    if not isinstance(w_dict, dict): return None
    if city_name in w_dict: return city_name
    target = _norm_city(city_name)
    for k in w_dict:
        if _norm_city(k) == target: return k
    return None


def get_city_weather_block(w_dict, city_name):
    """The city's own block. A city that is not in the dictionary gets an empty block (neutral multipliers), never another city's numbers."""
    k = city_key(w_dict, city_name)
    block = w_dict.get(k) if k is not None else None
    return block if isinstance(block, dict) else {}


def peak_windows(rainy_hours):
    hrs = rainy_hours or []
    return [name for name, hit in (("AM peak", any(h in MORNING_PEAK for h in hrs)),
                                   ("PM peak", any(h in AFTERNOON_PEAK for h in hrs)),
                                   ("night peak", any(h in NIGHT_PEAK for h in hrs))) if hit]


def classify_package(level, flood_risk, rain_mm_day, rainy_hours, is_weekend):
    """
    Decision tree for the packages. It always returns an answer (no gaps). Evaluation order:
      P6  flood risk Yes OR daily rain > 50 mm                       (hazard override, always first)
      P1  light / moderate rain, no peak-hour rain                   (BAU)
      P2  light / moderate rain during a peak hour
      P3  heavy rain during a daytime peak (AM or PM)
      P4  heavy / storm rain at night on a weekday
      P5  heavy / storm rain at night on a weekend
      P7  heavy / storm rain outside every peak, or timing unknown   (hold and monitor)
    `rainy_hours=None` means the hourly timing is unavailable for that day.
    Returns (package_number, alert_text).
    """
    if level == 'none':
        return 0, ""

    hours_known = rainy_hours is not None
    hrs = rainy_hours or []
    morning = any(h in MORNING_PEAK for h in hrs)
    afternoon = any(h in AFTERNOON_PEAK for h in hrs)
    night = any(h in NIGHT_PEAK for h in hrs)
    day_peak = morning or afternoon
    is_peak = day_peak or night

    if flood_risk == 'Yes' or rain_mm_day > STORM_DAILY_MM: pkg = 6
    elif level in ('light', 'moderate') and not is_peak: pkg = 1
    elif level in ('light', 'moderate') and is_peak: pkg = 2
    elif level == 'heavy' and day_peak: pkg = 3
    elif level in ('heavy', 'storm') and night and not is_weekend: pkg = 4
    elif level in ('heavy', 'storm') and night and is_weekend: pkg = 5
    else: pkg = 7

    if pkg in (0, 1):
        return pkg, ""

    windows = peak_windows(hrs)
    window_txt = " + ".join(windows) if windows else ("off-peak hours" if hours_known else "unknown hours")
    alert = f"{RAIN_LEVEL_LABEL[level]} rain ({rain_mm_day:.1f} mm) during {window_txt}; flood risk: {flood_risk}."
    return pkg, alert


def add_default_climate_cols(df):
    out = df.copy()
    out['rain_mm'] = 0.0
    out['intensity_cat'] = "No Rain"
    out['flood_risk'] = "No"
    out['package'] = 0
    out['prescription'] = PACKAGE_PRESCRIPTION[0]
    out['alert'] = ""
    out['rain_windows'] = ""
    for m in FUNNEL_METRICS: out[f'mult_{m}'] = 1.0
    return out


def apply_climate(df, df_w_daily, df_w_hourly, w_dict, city_name):
    """Merges weather, multiplies the funnel and injects `package`, `prescription` and `alert` columns."""
    out = df.merge(df_w_daily[['date', 'rain_mm', 'intensity_cat', 'flood_risk']], on='date', how='left')
    out['rain_mm'] = out['rain_mm'].fillna(0.0)
    out['intensity_cat'] = out['intensity_cat'].fillna('No Rain')
    out['flood_risk'] = out['flood_risk'].fillna('No')
    city_block = get_city_weather_block(w_dict, city_name)

    meta = []
    for _, r in out.iterrows():
        level = rain_level(r['intensity_cat'])
        mults = {m: 1.0 for m in FUNNEL_METRICS}
        pkg, alert, windows_txt = 0, "", ""
        if level != 'none':
            date_str = r['date'].strftime('%Y-%m-%d')
            rainy_hours = None  # None = no hourly rows for this day
            if not df_w_hourly.empty:
                day_rows = df_w_hourly[df_w_hourly['date'] == date_str]
                if not day_rows.empty:
                    rainy_hours = day_rows[day_rows['rain_mm'] >= PKG_RAIN_HOUR_MM]['hour'].tolist()
            mults = read_weather_multipliers(resolve_weather_node(city_block, r['intensity_cat'], r['flood_risk']))
            pkg, alert = classify_package(level, r['flood_risk'], float(r['rain_mm']), rainy_hours, r['date'].dayofweek >= 5)
            windows_txt = " + ".join(peak_windows(rainy_hours)) or ("off-peak" if rainy_hours is not None else "unknown")
        row = {f'mult_{m}': v for m, v in mults.items()}
        row.update({'package': pkg, 'prescription': PACKAGE_PRESCRIPTION[pkg], 'alert': alert, 'rain_windows': windows_txt})
        meta.append(row)

    out = pd.concat([out, pd.DataFrame(meta, index=out.index)], axis=1)
    for m in FUNNEL_METRICS:
        out[f'{m}_forecast'] = out[f'{m}_forecast'] * out[f'mult_{m}']
    return recompute_ratios(out)


# =====================================================================
# MODULE 3.5: WEATHER DICTIONARY AUTO-CALIBRATION (SILENT, IDEMPOTENT)
# =====================================================================
def auto_calibrate_weather_dict(city_name, df_historical, ledger_city, df_w_daily):
    """
    If it rained yesterday, compares the real multiplier (REAL_TABLE / raw ARIMA base) with the JSON and nudges it:
        new = 0.8 * dict + 0.2 * real        (a null / missing dict value is treated as 1.0)
    Runs once per city per day. Never raises and never blocks the UI. Returns a message when something changed.
    """
    try:
        if df_w_daily.empty: return None
        yesterday = today_mx() - timedelta(days=1)
        y_str = yesterday.strftime('%Y-%m-%d')
        log = load_json(CALIB_LOG_FILE)
        if log.get(city_name) == y_str: return None

        w_row = df_w_daily[df_w_daily['date'] == yesterday]
        if w_row.empty: return None
        intensity, flood = w_row['intensity_cat'].iloc[0], w_row['flood_risk'].iloc[0]
        if rain_level(intensity) == 'none': return None

        h_row = df_historical[df_historical['date'] == yesterday]
        base = ledger_city.get(y_str)
        if h_row.empty or not base: return None

        w_dict = load_weather_dict()
        city_block = get_city_weather_block(w_dict, city_name)  # Strict: never calibrate one city with another city's data
        node = resolve_weather_node(city_block, intensity, flood) if city_block else None
        if node is None: return None

        src = mult_source(node)
        changes = []
        for m in FUNNEL_METRICS:
            real_val, base_val = h_row[f'{m}_real'].iloc[0], base.get(m)
            if pd.isna(real_val) or not base_val or base_val <= 0: continue
            real_mult = float(real_val) / float(base_val)
            if not (CALIB_REAL_BOUNDS[0] <= real_mult <= CALIB_REAL_BOUNDS[1]): continue

            key = next((a for a in MULT_ALIASES[m] if a in src), MULT_ALIASES[m][0])
            old = safe_mult(src.get(key))  # null -> 1.0
            if abs(real_mult - old) <= CALIB_MIN_DIFF: continue
            new = round(CALIB_KEEP * old + CALIB_LEARN * real_mult, 4)
            src[key] = new
            changes.append(f"{METRIC_LABELS[m]} {old:.3f} to {new:.3f}")

        log[city_name] = y_str
        if changes:
            save_json(WEATHER_DICT_FILE, w_dict)
            audit = log.get('_changes', [])
            audit.append({'city': city_name, 'rain_day': y_str, 'bucket': f"{intensity} / flood {flood}", 'changes': changes})
            log['_changes'] = audit[-50:]
        save_json(CALIB_LOG_FILE, log)
        return (f"Weather dictionary auto-calibrated for {city_name} ({intensity}, flood {flood}): " + ", ".join(changes)) if changes else None
    except Exception:
        return None


# =====================================================================
# MODULE 4: ROLLING BIAS CORRECTOR (ARIMA MEMORY)
# =====================================================================
def update_arima_ledger(city_name, df_arima_all):
    """
    ML.FORECAST can only look forward, so we keep a ledger of what ARIMA predicted for each date.
    Future dates are refreshed on every run; once a date is in the past its prediction is frozen.
    """
    ledger = load_json(LEDGER_FILE)
    city_led = ledger.get(city_name, {})
    today = today_mx()
    changed = False

    for _, r in df_arima_all.iterrows():
        d = r['date']
        key = d.strftime('%Y-%m-%d')
        snap = {m: float(r[f'{m}_forecast']) for m in FUNNEL_METRICS if pd.notna(r.get(f'{m}_forecast'))}
        if snap and (d >= today or key not in city_led):
            if city_led.get(key) != snap:
                city_led[key] = snap
                changed = True

    cutoff = (today - timedelta(days=45)).strftime('%Y-%m-%d')
    for k in [k for k in city_led if k < cutoff]:
        del city_led[k]
        changed = True

    if changed:
        ledger[city_name] = city_led
        save_json(LEDGER_FILE, ledger)
    return city_led


def compute_bias_factors(df_historical, ledger_city):
    """
    Per metric: EMA of (REAL / what ARIMA said) over the last 7 closed days.
    A factor > 1 means ARIMA has been under-estimating. It is only applied when the bias is sustained
    (enough observations, same direction most days) and it is clamped to BIAS_CLAMP.
    """
    window = df_historical.sort_values('date').tail(BIAS_WINDOW_DAYS)
    factors, diag = {}, []
    for m in FUNNEL_METRICS:
        ratios = []
        for _, r in window.iterrows():
            snap = ledger_city.get(r['date'].strftime('%Y-%m-%d'))
            real = r.get(f'{m}_real')
            if snap and pd.notna(real) and real > 0 and snap.get(m, 0) > 0:
                ratios.append(float(real) / snap[m])

        factor, n = 1.0, len(ratios)
        if n >= BIAS_MIN_OBS:
            s = pd.Series(ratios)
            if max((s < 1).mean(), (s > 1).mean()) >= BIAS_MIN_CONSISTENCY:
                ema = s.ewm(span=BIAS_WINDOW_DAYS, adjust=False).mean().iloc[-1]
                factor = float(np.clip(ema, BIAS_CLAMP[0], BIAS_CLAMP[1]))
        factors[m] = factor
        diag.append({'metric': m, 'n': n, 'factor': factor})
    return factors, diag


def apply_bias(df, factors):
    """Applied to future days BEFORE climate and anomalies."""
    out = df.copy()
    for m in FUNNEL_METRICS:
        out[f'{m}_forecast'] = out[f'{m}_forecast'] * factors.get(m, 1.0)
    return recompute_ratios(out)


# =====================================================================
# MODULE 5: UNIVERSAL ANOMALY ENGINE (replaces Externalities & Black Swans)
# =====================================================================
def purge_past_overrides():
    """Reads scheduled_overrides.json and drops every anomaly whose date has already passed."""
    data = load_json(OVERRIDES_FILE)
    today = today_mx()
    kept = {}
    for k, v in data.items():
        d = pd.to_datetime(k, errors='coerce')
        if pd.notna(d) and d >= today:
            kept[k] = v
    if len(kept) != len(data):
        save_json(OVERRIDES_FILE, kept)
    return kept


def expand_dates(val):
    """st.date_input returns a date, or a tuple of 1-2 dates in range mode."""
    if isinstance(val, (tuple, list)):
        if len(val) == 2: return list(pd.date_range(val[0], val[1]))
        if len(val) == 1: return [pd.Timestamp(val[0])]
        return []
    return [pd.Timestamp(val)]


def _safe_ratio(num, den):
    try:
        if num is None or den is None or pd.isna(num) or pd.isna(den) or float(den) <= 0:
            return None
        return float(num) / float(den)
    except (TypeError, ValueError):
        return None


def derive_ratio_mults(mults):
    """How the funnel ratios move when the volumes move: CR = trips/calls, ECR = calls/eyeballs, ETR = trips/eyeballs, SDR = TSH/eyeballs."""
    return {
        'cr': mults['trips'] / mults['calls'],
        'ecr': mults['calls'] / mults['eyeballs'],
        'etr': mults['trips'] / mults['eyeballs'],
        'sdr': mults['tsh'] / mults['eyeballs'],
    }


def compute_anomaly_multipliers(city_id, ref_days):
    """
    For every reference day and each of trips / calls / eyeballs / TSH (and GMV):
      WoW  = reference day / same weekday 1 week earlier
      Wo2W = reference day / same weekday 2 weeks earlier
    The multiplier replayed on the future day is the mean of the ratios that exist (clamped). CR, ECR, ETR and SDR
    are never stored as inputs: they are re-derived from the volumes when the anomaly is applied.
    """
    if client is None: raise RuntimeError("BigQuery client is not available.")
    needed = set()
    for rd in ref_days:
        for k in range(ANOMALY_LOOKBACK_WEEKS + 1):
            needed.add((rd - timedelta(days=7 * k)).strftime('%Y-%m-%d'))
    in_list = ", ".join(f"'{d}'" for d in sorted(needed))
    q = f"""
        SELECT DATE(date_value) AS date, SUM(trips) AS trips, SUM(calls) AS calls, SUM(eyeballs) AS eyeballs,
               SUM(supply_hours) AS tsh, SUM(gmv) AS gmv
        FROM {REAL_TABLE}
        WHERE CAST(city_id AS STRING) = '{city_id}' AND product = 'Managed Products'
          AND DATE(date_value) IN ({in_list})
        GROUP BY 1
    """
    df = client.query(q).to_dataframe()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    result = {}
    for rd in ref_days:
        if rd not in df.index:
            raise ValueError(f"No data in BigQuery for the reference date {rd.strftime('%Y-%m-%d')}.")
        wow, wo2w, mults = {}, {}, {}
        for m in FUNNEL_METRICS:
            ref_val = df.loc[rd, m]
            wow[m] = _safe_ratio(ref_val, df[m].get(rd - timedelta(days=7)))
            wo2w[m] = _safe_ratio(ref_val, df[m].get(rd - timedelta(days=14)))
            avail = [r for r in (wow[m], wo2w[m]) if r is not None]
            mults[m] = float(np.clip(np.mean(avail), ANOMALY_CLAMP[0], ANOMALY_CLAMP[1])) if avail else 1.0
        if all(wow[m] is None and wo2w[m] is None for m in FUNNEL_METRICS):
            raise ValueError(f"No baseline weeks found before {rd.strftime('%Y-%m-%d')}.")
        result[rd.strftime('%Y-%m-%d')] = {'wow': wow, 'wo2w': wo2w, 'multipliers': mults, 'derived': derive_ratio_mults(mults)}
    return result


def save_scheduled_anomaly(city_name, city_id, future_days, ref_days):
    computed = compute_anomaly_multipliers(city_id, ref_days)
    data = load_json(OVERRIDES_FILE)
    for fd, rd in zip(future_days, ref_days):
        node = computed[rd.strftime('%Y-%m-%d')]
        data.setdefault(fd.strftime('%Y-%m-%d'), {})[city_name] = {
            "reference_date": rd.strftime('%Y-%m-%d'),
            "saved_on": today_mx().strftime('%Y-%m-%d'),
            "multipliers": node['multipliers'],
            "wow": node['wow'], "wo2w": node['wo2w'], "derived": node['derived'],
        }
    save_json(OVERRIDES_FILE, data)


def remove_scheduled_anomalies(city_name, date_strs):
    data = load_json(OVERRIDES_FILE)
    for d in date_strs:
        if isinstance(data.get(d), dict):
            data[d].pop(city_name, None)
            if not data[d]: data.pop(d)
    save_json(OVERRIDES_FILE, data)


def apply_scheduled_overrides(df, city_name, overrides):
    """Injects every scheduled anomaly that falls inside the forecast horizon into the funnel."""
    out = df.copy()
    out['has_anomaly'] = False
    applied = []
    for date_str, by_city in sorted(overrides.items()):
        node = by_city.get(city_name) if isinstance(by_city, dict) else None
        d = pd.to_datetime(date_str, errors='coerce')
        if not isinstance(node, dict) or pd.isna(d): continue
        mask = out['date'] == d
        if not mask.any(): continue
        raw = node.get('multipliers')
        raw = raw if isinstance(raw, dict) else {}
        mults = {m: safe_mult(raw.get(m)) for m in FUNNEL_METRICS}
        scale_rows(out, mask, mults)
        out.loc[mask, 'has_anomaly'] = True
        applied.append({'date': d, 'reference_date': node.get('reference_date', '?'), 'mults': mults, 'derived': derive_ratio_mults(mults)})
    return recompute_ratios(out), applied


def build_adjustment_flags(df_clim, df_w_daily, climate_on):
    """One row per forecast day that carries an adjustment, for the markers on the charts."""
    heat_dates = set()
    if df_w_daily is not None and not df_w_daily.empty:
        heat_dates = set(df_w_daily.loc[df_w_daily['Temp_Max'] > HEAT_THRESHOLD_C, 'date'])
    rows = []
    for _, r in df_clim.iterrows():
        icons, tips = [], []
        pkg = int(r.get('package', 0))
        if climate_on and pkg == 6:
            icons.append("🚨"); tips.append("Package 6 safety lock")
        elif climate_on and any(abs(r[f'mult_{m}'] - 1.0) > 1e-9 for m in FUNNEL_METRICS):
            icons.append("🌧️"); tips.append(f"Climate adjustment applied to the forecast ({r['intensity_cat']}" + (f", package {pkg}" if pkg else "") + ")")
        elif climate_on and pkg >= 2:
            icons.append("☔"); tips.append(f"Rain package {pkg}: budget action only, the forecast is unchanged (no multipliers for this rain bucket)")
        if bool(r.get('has_anomaly', False)):
            icons.append("⚡"); tips.append("Saved anomaly applied")
        if r['date'] in heat_dates:
            icons.append("🔥"); tips.append("Heatwave > 35C")
        if icons:
            rows.append({'date': r['date'], 'icon': "".join(icons), 'tip': " | ".join(tips)})
    return pd.DataFrame(rows, columns=['date', 'icon', 'tip'])


# =====================================================================
# MODULE 6: FORECAST PIPELINE (ARIMA -> Bias -> Anomalies -> Climate)
# =====================================================================
def build_forecast_pipeline(params, df_historical, df_arima_all, df_w_daily, df_w_hourly):
    """
    Returns a dict with every stage so the UI can chart them:
      df_org   : raw ARIMA, future days only
      df_bias  : after the Rolling Bias Corrector
      df_anom  : after scheduled anomalies
      df_clim  : final adjusted forecast (also carries package / prescription / alert)
    """
    city_name = params['city_name']
    last_real = df_historical['date'].max()

    # Days already covered by actuals are not "forecast": they only feed the ledger.
    df_org = df_arima_all[df_arima_all['date'] > last_real].copy().reset_index(drop=True)
    if df_org.empty:
        return None
    for m in FUNNEL_METRICS:
        df_org[f'{m}_forecast'] = df_org[f'{m}_forecast'].astype(float)
    df_org = recompute_ratios(df_org)

    # 1) Rolling Bias Corrector
    ledger_city = update_arima_ledger(city_name, df_arima_all)
    factors, bias_diag = compute_bias_factors(df_historical, ledger_city)
    df_bias = apply_bias(df_org, factors)

    # 2) Anomaly Engine (past anomalies are always purged from the JSON)
    overrides = purge_past_overrides()
    if params.get('use_anomalies', True):
        df_anom, anomalies_applied = apply_scheduled_overrides(df_bias, city_name, overrides)
    else:
        df_anom, anomalies_applied = df_bias.copy(), []
        df_anom['has_anomaly'] = False

    # 3) Climate tree (the dictionary is auto-calibrated first so today's run already uses the learning)
    calib_msg = auto_calibrate_weather_dict(city_name, df_historical, ledger_city, df_w_daily)
    if params.get('clima', True) and not df_w_daily.empty:
        df_clim = apply_climate(df_anom, df_w_daily, df_w_hourly, load_weather_dict(), city_name)
    else:
        df_clim = add_default_climate_cols(df_anom)

    return {
        "df_org": df_org, "df_bias": df_bias, "df_anom": df_anom, "df_clim": df_clim,
        "bias_factors": factors, "bias_diag": bias_diag,
        "anomalies_applied": anomalies_applied, "calib_msg": calib_msg,
    }

# =====================================================================
# MODULE 7: ALLOCATION ENGINE v4 (MARGINAL-RETURN OPTIMIZER)
# ---------------------------------------------------------------------
# UNITS RULE: everything inside this module is a plain number. Rates are fractions (0.48, not 48),
# budgets are money. Percentages only exist at the UI boundary (pct_to_frac / frac_to_pct).
#
# MODEL (per remaining day i, per channel k in {DRV, PAX}):
#   * Every channel has a relative "leak" g_ik (how far the day sits below its healthy reference, as a % of that reference).
#       DRV leak = CR shortfall + RHO_SUPPLY * supply-per-eyeball shortfall      (fulfillment)
#       PAX leak = ECR shortfall + RHO_EYEBALL * eyeball shortfall               (conversion / acquisition)
#   * Money recovers the leak with diminishing returns:  r(u) = g * (1 - exp(-beta * u / g)),  u = spend / day GMV
#       beta = relative lift per 1.0 of GMV burned when the leak is wide open (prior, or fitted from Burn SoT history).
#   * Incremental trips ~ day trips * r.  Maximizing total trips for a fixed budget gives equal marginal return everywhere,
#     solved exactly by water-filling (closed form per node + a bisection on the price of money).
#   * Weather packages, targets and trend only change how much a node is WORTH (multipliers), never the physics.
# =====================================================================
import math

ENGINE_BETA_PRIOR = 1.5      # relative lift per 100% of GMV burned (1% burn ~ 1.5% lift). Neutral: same for DRV and PAX.
BETA_FIT_MIN_DAYS = 45       # fewer joined days of (burn, funnel) than this -> keep the prior
BETA_SHRINK_T2 = 9.0         # shrinkage strength: weight on the fit = t^2 / (t^2 + 9)
BETA_CLIP = (0.25, 4.0)      # fitted beta stays within [0.25x, 4x] of the prior

G_MIN = 0.02                 # every day carries a 2% stretch opportunity on top of its measured leak (the budget always has a home)
G_FLOOR = 0.005              # a day performing well above its reference can fall to this, but never to zero
SURPLUS_CLIP = -0.10         # being 10%+ better than the reference does not reduce the need any further
RHO_SUPPLY = 0.5             # a supply-per-eyeball shortfall counts half as much as a CR shortfall (leading indicator)
RHO_EYEBALL = 0.35           # an eyeball shortfall counts 35% as much as an ECR shortfall (acquisition value)
SAFE_CAP_FRAC = 0.08         # safe mode: a day can burn at most 8% of that day's GMV (DRV + PAX together)
TREND_KAPPA = 2.0            # trend deterioration (relative, 4 weeks) -> value multiplier slope
TREND_CLIP = (-0.15, 0.30)
CALLS_TARGET_GAIN = 2.0      # weekly calls shortfall of 10% -> PAX value x1.2
CALLS_MULT_CLAMP = (0.6, 2.0)

# (weight on past behaviour inside each day's state, weight of the 4-week trend multiplier)
SIGNAL_MODES = {"forecast": (0.0, 0.0), "blend": (0.4, 0.5), "past": (1.0, 1.0)}
SIGNAL_LABELS = {"forecast": "Forecast", "blend": "Blend (forecast + recent trend)", "past": "Past trend only"}

# How a weather package changes the WORTH of a channel on that day and the next one. Reviewed by humans: edit freely.
PACKAGE_STANCE = {
    0: dict(drv=1.00, pax=1.00, next_drv=1.00, next_pax=1.00, note="No weather adjustment."),
    1: dict(drv=1.00, pax=1.00, next_drv=1.00, next_pax=1.00, note="Light/moderate rain off-peak: business as usual."),
    2: dict(drv=1.10, pax=1.00, next_drv=1.00, next_pax=1.00, note="Active rain: mild supply stress, drivers are worth 10% more."),
    3: dict(drv=1.25, pax=0.90, next_drv=1.15, next_pax=1.15, note="Active storm: push supply during the event (+25% DRV, PAX -10% because demand cannot convert without cars), then both channels +15% the next day."),
    4: dict(drv=0.85, pax=0.85, next_drv=1.35, next_pax=1.00, note="Workday storm at night: hold back today (-15%) and shift value to tomorrow's morning peak (DRV +35%)."),
    5: dict(drv=1.20, pax=1.00, next_drv=1.00, next_pax=1.00, note="Weekend storm at night: late-night driver push (DRV +20%)."),
    6: dict(drv=0.00, pax=0.30, next_drv=1.20, next_pax=1.20, note="Safety lock: no driver incentives while the hazard lasts, passenger spend at 30%, recover with +20% the next day."),
    7: dict(drv=1.05, pax=1.00, next_drv=1.00, next_pax=1.00, note="Heavy rain outside peaks: hold and monitor (DRV +5%)."),
}


# ---------------------------------------------------------------- unit boundary
def pct_to_frac(p):
    """UI percentage (e.g. 6 or 90) -> fraction (0.06, 0.90). The ONLY place percentages enter the maths."""
    return float(p) / 100.0


def frac_to_pct(f):
    return float(f) * 100.0


# ---------------------------------------------------------------- funnel helpers (all fractions)
def funnel_fractions(df, suffix):
    """From raw volumes derive E (eyeballs), e (ECR), c (CR), sdr and T (trips). `suffix` is 'real' or 'forecast'."""
    E = df[f'eyeballs_{suffix}'].astype(float)
    C = df[f'calls_{suffix}'].astype(float)
    T = df[f'trips_{suffix}'].astype(float)
    S = df[f'tsh_{suffix}'].astype(float)
    return pd.DataFrame({
        'date': pd.to_datetime(df['date']).to_numpy(),
        'E': E.to_numpy(),
        'e': (C / E.replace(0, np.nan)).to_numpy(),
        'c': (T / C.replace(0, np.nan)).to_numpy(),
        'sdr': (S / E.replace(0, np.nan)).to_numpy(),
        'T': T.to_numpy(),
    })


def weekday_reference(hist_f, window=28):
    """
    Healthy reference per weekday = mean of the last 4 same weekdays (falls back to the overall mean).
    Also returns the 'persistence' state = mean of the last 2 same weekdays.
    """
    h = hist_f.sort_values('date').tail(window).copy()
    h['dow'] = pd.to_datetime(h['date']).dt.dayofweek
    keys = ['E', 'e', 'c', 'sdr']
    overall = h[keys].mean()
    by = h.groupby('dow')[keys].mean()
    cnt = h.groupby('dow').size()
    last2 = h.groupby('dow')[keys].apply(lambda g: g.tail(2).mean())

    def ref(dow):
        return by.loc[dow] if (dow in by.index and cnt[dow] >= 2) else overall

    def persist(dow):
        return last2.loc[dow] if (dow in last2.index and cnt[dow] >= 2) else ref(dow)

    return ref, persist


def weekly_trends(hist_f, weeks=4):
    """
    Slope of 4 consecutive 7-day blocks per metric, as a relative decline over the window (+ = deteriorating).
    DRV trend = supply-per-eyeball and CR. PAX trend = ECR and eyeballs.
    """
    h = hist_f.sort_values('date').tail(weeks * 7).reset_index(drop=True)
    if len(h) < int(weeks * 7 * 0.75):
        return {'drv': 0.0, 'pax': 0.0, 'declines': {}}
    chunks = np.array_split(np.arange(len(h)), weeks)
    x = np.arange(weeks)
    declines = {}
    for k in ('E', 'e', 'c', 'sdr'):
        v = np.array([h.loc[idx, k].mean() for idx in chunks], dtype=float)
        if (not np.all(np.isfinite(v))) or v.mean() <= 0:
            declines[k] = 0.0
            continue
        slope = np.polyfit(x, v, 1)[0]
        declines[k] = float(-slope * (weeks - 1) / v.mean())
    drv = float(np.clip(0.5 * (declines['sdr'] + declines['c']), *TREND_CLIP))
    pax = float(np.clip(0.5 * (declines['e'] + declines['E']), *TREND_CLIP))
    return {'drv': drv, 'pax': pax, 'declines': declines}


def _stance(pkg):
    try:
        return PACKAGE_STANCE.get(int(pkg), PACKAGE_STANCE[0])
    except (TypeError, ValueError):
        return PACKAGE_STANCE[0]


# ---------------------------------------------------------------- response-slope calibration (Burn SoT history)
def fit_response_slopes(df_cal, prior=ENGINE_BETA_PRIOR):
    """
    Semi-elasticity of CR to DRV burn and of ECR to PAX burn:  log(metric) ~ weekday + trend + own burn + other burn.
    df_cal columns: date, trips, calls, eyeballs, drv_burn, pax_burn, b_gmv (money, same-day).
    A fit is only trusted when positive and t >= 1, and it is shrunk toward the prior: w = t^2 / (t^2 + 9).
    Otherwise the neutral prior is returned. Never raises.
    """
    out = {k: dict(beta=prior, source='prior', raw=None, t=None, n=0) for k in ('drv', 'pax')}
    try:
        d = df_cal.copy()
        d['date'] = pd.to_datetime(d['date'])
        for c in ('trips', 'calls', 'eyeballs', 'b_gmv', 'drv_burn', 'pax_burn'):
            d[c] = pd.to_numeric(d[c], errors='coerce')
        d = d.dropna(subset=['trips', 'calls', 'eyeballs', 'b_gmv'])
        d = d[(d.trips > 0) & (d.calls > 0) & (d.eyeballs > 0) & (d.b_gmv > 0)].sort_values('date').reset_index(drop=True)
        n = len(d)
        if n < BETA_FIT_MIN_DAYS:
            return out
        d['e'] = d.calls / d.eyeballs
        d['c'] = d.trips / d.calls
        d['xd'] = d.drv_burn.fillna(0.0) / d.b_gmv
        d['xp'] = d.pax_burn.fillna(0.0) / d.b_gmv
        dow = pd.get_dummies(d.date.dt.dayofweek, drop_first=True).astype(float)
        base = np.column_stack([np.ones(n), dow.to_numpy(), np.arange(n) / n])
        idx = base.shape[1]
        for ch, ycol, own, other in (('drv', 'c', 'xd', 'xp'), ('pax', 'e', 'xp', 'xd')):
            if d[own].std() < 1e-4:
                continue
            y = np.log(d[ycol].to_numpy())
            X = np.column_stack([base, d[own].to_numpy(), d[other].to_numpy()])
            coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
            if rank < X.shape[1]:
                continue
            resid = y - X @ coef
            dof = n - X.shape[1]
            s2 = float(resid @ resid) / dof
            cov = s2 * np.linalg.inv(X.T @ X)
            b, se = float(coef[idx]), math.sqrt(max(cov[idx, idx], 0.0))
            t = b / se if se > 0 else 0.0
            out[ch].update(raw=b, t=t, n=n)
            if b > 0 and t >= 1.0:
                w = t * t / (t * t + BETA_SHRINK_T2)
                beta = float(np.clip(w * b + (1 - w) * prior, prior * BETA_CLIP[0], prior * BETA_CLIP[1]))
                out[ch].update(beta=beta, source='calibrated')
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- optimizer
def _x_at(price, a, k):
    """Spend that equalizes marginal return to `price`: x = k * ln(a / price) where a > price, else 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.where(a > price, k * np.log(np.maximum(a, 1e-300) / price), 0.0)
    return np.where(np.isfinite(x), np.maximum(x, 0.0), 0.0)


def _capped_day(price, a_i, k_i, cap_i):
    x = _x_at(price, a_i, k_i)
    if cap_i is None or x.sum() <= cap_i:
        return x
    lo, hi = price, float(a_i.max())
    for _ in range(80):  # the daily cap carries its own shadow price on top of the budget price
        mid = math.sqrt(lo * hi)
        if _x_at(mid, a_i, k_i).sum() > cap_i:
            lo = mid
        else:
            hi = mid
    return _x_at(hi, a_i, k_i)


def _allocate_joint(a, k, caps, budget):
    """Both channels compete for one budget. a, k: (n, 2). caps: (n,) joint daily cap or None. Returns (x, leftover)."""
    n = a.shape[0]
    amax = float(a.max())
    if amax <= 0 or budget <= 0:
        return np.zeros_like(a), float(max(budget, 0.0))

    def at(price):
        return np.vstack([_capped_day(price, a[i], k[i], None if caps is None else caps[i]) for i in range(n)])

    lo, hi = amax * 1e-30, amax
    x_lo = at(lo)
    if x_lo.sum() < budget:
        return x_lo, float(budget - x_lo.sum())
    for _ in range(120):
        mid = math.sqrt(lo * hi)
        if at(mid).sum() > budget:
            lo = mid
        else:
            hi = mid
    x = at(hi)
    if x.sum() > 0:
        x = x * (budget / x.sum())
    return x, 0.0


def _allocate_channel(a_c, k_c, caps_c, budget):
    """One channel, own budget (used by the forced split). a_c, k_c, caps_c: (n,)."""
    amax = float(a_c.max())
    if amax <= 0 or budget <= 0:
        return np.zeros_like(a_c), float(max(budget, 0.0))

    def at(price):
        x = _x_at(price, a_c, k_c)
        return x if caps_c is None else np.minimum(x, caps_c)

    lo, hi = amax * 1e-30, amax
    x_lo = at(lo)
    if x_lo.sum() < budget:
        return x_lo, float(budget - x_lo.sum())
    for _ in range(120):
        mid = math.sqrt(lo * hi)
        if at(mid).sum() > budget:
            lo = mid
        else:
            hi = mid
    x = at(hi)
    if x.sum() > 0:
        x = x * (budget / x.sum())
    return x, 0.0


def _recovery(x, g, G, beta):
    """Relative recovery of a leak g when spending x on a day with GMV G."""
    u = np.where(G > 0, x / np.where(G > 0, G, 1.0), 0.0)
    return g * (1.0 - np.exp(-beta * u / g))


def run_allocation_engine(fcst_week, df_hist, budget_usd, mode="forecast", cr_target=0.0, calls_target=0.0,
                          weekly_calls=0.0, force_drv_share=None, safe_mode=True, betas=None):
    """
    fcst_week     : remaining forecast days of the target week (adjusted forecast, with 'package' and 'gmv_forecast').
    df_hist       : actuals (columns *_real).
    budget_usd    : money still to spend this week.
    cr_target     : FRACTION (0.90 = 90%) or 0. Becomes the reference CR of the driver leak (below it -> more DRV, above -> less).
    calls_target  : weekly calls (absolute) or 0. Scales the passenger value (below target -> more PAX, above -> less).
    weekly_calls  : calls of the whole week (actual so far + forecast) to compare with calls_target.
    force_drv_share: FRACTION of the remaining budget forced to DRV (0.10 = 10/90) or None. Day shape is still optimized.
    """
    n = len(fcst_week)
    if n == 0 or budget_usd is None or budget_usd <= 0:
        return None
    betas = betas or {'drv': ENGINE_BETA_PRIOR, 'pax': ENGINE_BETA_PRIOR}
    w_state, w_trend = SIGNAL_MODES.get(mode, SIGNAL_MODES['forecast'])

    fw = fcst_week.reset_index(drop=True)
    fc = funnel_fractions(fw, 'forecast')
    hist_f = funnel_fractions(df_hist, 'real')
    ref_fn, persist_fn = weekday_reference(hist_f)
    trend = weekly_trends(hist_f)
    G = fw['gmv_forecast'].astype(float).to_numpy()
    T = fc['T'].to_numpy()
    pkgs = [int(p) for p in fw['package']] if 'package' in fw.columns else [0] * n

    calls_mult = 1.0
    if calls_target and calls_target > 0 and weekly_calls and weekly_calls > 0:
        rel = (calls_target - weekly_calls) / calls_target
        calls_mult = float(np.clip(1.0 + CALLS_TARGET_GAIN * rel, *CALLS_MULT_CLAMP))

    def leaks_of(s, ref, c_ref):
        """Signed shortfalls vs the healthy reference (+ = leak, - = better than normal) and the two channel opportunities."""
        comp = {
            'cr': (c_ref - s['c']) / c_ref if c_ref > 0 else 0.0,
            'sdr': 1.0 - s['sdr'] / ref['sdr'] if ref['sdr'] > 0 else 0.0,
            'ecr': (ref['e'] - s['e']) / ref['e'] if ref['e'] > 0 else 0.0,
            'eb': 1.0 - s['E'] / ref['E'] if ref['E'] > 0 else 0.0,
        }
        comp = {key: max(SURPLUS_CLIP, val) for key, val in comp.items()}
        g_drv = max(G_FLOOR, G_MIN + comp['cr'] + RHO_SUPPLY * comp['sdr'])
        g_pax = max(G_FLOOR, G_MIN + comp['ecr'] + RHO_EYEBALL * comp['eb'])
        return comp, g_drv, g_pax

    rows = []
    for i in range(n):
        dow = pd.Timestamp(fc.loc[i, 'date']).dayofweek
        ref, per = ref_fn(dow), persist_fn(dow)
        fval = {}
        for key in ('E', 'e', 'c', 'sdr'):
            f = fc.loc[i, key]
            fval[key] = ref[key] if not np.isfinite(f) else float(f)

        def state(w):
            return {key: (1.0 - w) * fval[key] + w * float(per[key]) for key in ('E', 'e', 'c', 'sdr')}

        s = state(w_state)                      # what the engine uses
        s_fc, s_pa = state(0.0), state(1.0)     # the two pure views, kept for the explanations
        c_ref = cr_target if cr_target and cr_target > 0 else float(ref['c'])
        comp, g_drv, g_pax = leaks_of(s, ref, c_ref)
        _, g_drv_fc, g_pax_fc = leaks_of(s_fc, ref, c_ref)
        _, g_drv_pa, g_pax_pa = leaks_of(s_pa, ref, c_ref)

        sp = _stance(pkgs[i])
        m_drv, m_pax = sp['drv'], sp['pax']
        carried = False
        if i > 0 and pkgs[i] != 6:
            prev = _stance(pkgs[i - 1])
            if prev['next_drv'] != 1.0 or prev['next_pax'] != 1.0:
                carried = True
            m_drv *= prev['next_drv']
            m_pax *= prev['next_pax']
        t_drv = t_pax = 1.0
        if w_trend > 0:
            t_drv = max(0.5, 1.0 + w_trend * TREND_KAPPA * trend['drv'])
            t_pax = max(0.5, 1.0 + w_trend * TREND_KAPPA * trend['pax'])
            m_drv *= t_drv
            m_pax *= t_pax
        m_pax *= calls_mult

        rows.append(dict(
            date=pd.Timestamp(fc.loc[i, 'date']), E=s['E'], e=s['e'], c=s['c'], sdr=s['sdr'],
            E_fc=s_fc['E'], e_fc=s_fc['e'], c_fc=s_fc['c'], sdr_fc=s_fc['sdr'],
            E_pa=s_pa['E'], e_pa=s_pa['e'], c_pa=s_pa['c'], sdr_pa=s_pa['sdr'],
            E_ref=float(ref['E']), e_ref=float(ref['e']), c_ref=float(c_ref), sdr_ref=float(ref['sdr']),
            leak_cr=comp['cr'], leak_sdr=comp['sdr'], leak_ecr=comp['ecr'], leak_eb=comp['eb'],
            g_drv=g_drv, g_pax=g_pax, g_drv_fc=g_drv_fc, g_pax_fc=g_pax_fc, g_drv_pa=g_drv_pa, g_pax_pa=g_pax_pa,
            trend_m_drv=t_drv, trend_m_pax=t_pax,
            m_drv=m_drv, m_pax=m_pax, carried=carried, package=pkgs[i], T=float(T[i]), G=float(G[i]),
        ))
    d = pd.DataFrame(rows)

    beta = np.array([betas['drv'], betas['pax']], dtype=float)
    g = d[['g_drv', 'g_pax']].to_numpy()
    m = d[['m_drv', 'm_pax']].to_numpy()
    Gv = d['G'].to_numpy()[:, None]
    Tv = d['T'].to_numpy()[:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(Gv > 0, Tv * beta[None, :] * m / np.where(Gv > 0, Gv, 1.0), 0.0)   # marginal trips per money at zero spend
        k = np.where(beta[None, :] > 0, g * Gv / beta[None, :], 0.0)                      # width of the leak in money
    cap_total = SAFE_CAP_FRAC * d['G'].to_numpy() if safe_mode else None

    leftover = 0.0
    if force_drv_share is not None:
        s_drv = float(np.clip(force_drv_share, 0.0, 1.0))
        x = np.zeros_like(a)
        for ch, share in ((0, s_drv), (1, 1.0 - s_drv)):
            caps_c = None if cap_total is None else cap_total * share
            xc, left = _allocate_channel(a[:, ch], k[:, ch], caps_c, budget_usd * share)
            x[:, ch] = xc
            leftover += left
    else:
        x, leftover = _allocate_joint(a, k, cap_total, float(budget_usd))

    x_drv, x_pax = x[:, 0], x[:, 1]
    r_drv = _recovery(x_drv, d['g_drv'].to_numpy(), d['G'].to_numpy(), beta[0])
    r_pax = _recovery(x_pax, d['g_pax'].to_numpy(), d['G'].to_numpy(), beta[1])
    d['x_drv'], d['x_pax'] = x_drv, x_pax
    d['x_total'] = x_drv + x_pax
    d['drv_pct'] = np.where(d['G'] > 0, d['x_drv'] / d['G'] * 100.0, 0.0)
    d['pax_pct'] = np.where(d['G'] > 0, d['x_pax'] / d['G'] * 100.0, 0.0)
    d['burn_pct'] = d['drv_pct'] + d['pax_pct']
    d['split_drv'] = np.where(d['x_total'] > 0, d['x_drv'] / d['x_total'], 0.0)
    spent = float(d['x_total'].sum())
    d['share'] = d['x_total'] / spent if spent > 0 else 0.0
    d['fair_share'] = fc['E'].to_numpy() / fc['E'].sum() if fc['E'].sum() > 0 else 0.0
    d['gain_drv'] = d['T'] * r_drv
    d['gain_pax'] = d['T'] * r_pax
    d['gain'] = d['gain_drv'] + d['gain_pax']

    pool_drv = float((d['T'] * d['g_drv']).sum())
    pool_pax = float((d['T'] * d['g_pax']).sum())
    closed = float(d['gain'].sum() / (pool_drv + pool_pax)) if (pool_drv + pool_pax) > 0 else 0.0
    summary = dict(
        budget=float(budget_usd), spent=spent, leftover=float(leftover),
        drv_total=float(x_drv.sum()), pax_total=float(x_pax.sum()),
        macro_drv=float(x_drv.sum() / spent) if spent > 0 else 0.0,
        leak_share_drv=pool_drv / (pool_drv + pool_pax) if (pool_drv + pool_pax) > 0 else 0.5,
        gain=float(d['gain'].sum()), cpit=(spent / float(d['gain'].sum())) if d['gain'].sum() > 0 else float('nan'),
        gain_pct=float(d['gain'].sum() / d['T'].sum() * 100.0) if d['T'].sum() > 0 else 0.0,
        closed=closed, mode=mode, w_state=w_state, w_trend=w_trend, calls_mult=calls_mult, cr_target=float(cr_target or 0.0), calls_target=float(calls_target or 0.0),
        weekly_calls=float(weekly_calls or 0.0), force_drv_share=force_drv_share, safe_mode=bool(safe_mode),
        trend=trend, betas=dict(betas), caps_binding=bool(cap_total is not None and np.any(d['x_total'].to_numpy() >= cap_total * 0.999)),
    )
    return {'days': d, 'summary': summary}


# ---------------------------------------------------------------- explanations
def signal_note(summary):
    """What the chosen signal source means for THIS plan (shown above the allocation)."""
    mode, tr = summary.get('mode', 'forecast'), summary.get('trend', {'drv': 0.0, 'pax': 0.0})
    ws, wt = SIGNAL_MODES.get(mode, SIGNAL_MODES['forecast'])
    if mode == 'forecast':
        return ("Signal: FORECAST. Each day is sized by what the adjusted forecast expects (CR, ECR, supply, eyeballs). "
                "Recent history only sets the healthy reference for that weekday. The 4-week trend is not used.")
    trend_txt = (f"4-week trend: supply/CR {tr['drv'] * 100:+.1f}%, conversion/eyeballs {tr['pax'] * 100:+.1f}% deterioration "
                 f"(positive = getting worse)")
    if mode == 'blend':
        return (f"Signal: BLEND. Inside every day the state is {int((1 - ws) * 100)}% forecast + {int(ws * 100)}% recent behaviour "
                f"(last 2 same weekdays), and the 4-week trend moves each channel's value at {int(wt * 100)}% weight. {trend_txt}.")
    return ("Signal: PAST ONLY. The forecast state is ignored: each day is read from recent behaviour (last 2 same weekdays) and the full 4-week trend. "
            f"Weather packages still apply because they describe what is coming. {trend_txt}.")


def explain_day(r, summary, weather_txt="", pkg_prescription=""):
    """Plain-language reasoning for one day (r = one row of result['days']). The reading changes with the signal source."""
    mode = summary.get('mode', 'forecast')
    ws, wt = SIGNAL_MODES.get(mode, SIGNAL_MODES['forecast'])
    lines = []
    pkg = int(r['package'])
    if pkg > 0:
        lines.append(f"**Weather:** {weather_txt}. Package {pkg}: {pkg_prescription}. {PACKAGE_STANCE[pkg]['note']}")
    elif weather_txt:
        lines.append(f"**Weather:** {weather_txt}. No package applies.")
    if r['carried']:
        lines.append("**Carry-over:** yesterday's weather package passes value to this day (recovery / time-shift).")

    def pp(a, b):
        return f"{(a - b) * 100:+.1f}pp"

    # --- how this day is being read
    if mode == 'forecast':
        lines.append(f"**Reading (forecast):** the forecast expects CR {r['c_fc']*100:.1f}% and ECR {r['e_fc']*100:.1f}% on this day, versus a healthy "
                     f"{r['c_ref']*100:.1f}% / {r['e_ref']*100:.1f}%.")
        dd, dp = r['g_drv_pa'] - r['g_drv'], r['g_pax_pa'] - r['g_pax']
        if abs(dd) < 0.01 and abs(dp) < 0.01:
            lines.append("**Cross-check with the past:** recent behaviour on this weekday agrees with the forecast, so the call is not sensitive to the signal.")
        else:
            lines.append(f"**Cross-check with the past:** recent same-weekday behaviour (CR {r['c_pa']*100:.1f}%, ECR {r['e_pa']*100:.1f}%) would size the driver "
                         f"opportunity at {r['g_drv_pa']*100:.1f}% (forecast: {r['g_drv']*100:.1f}%) and the passenger one at {r['g_pax_pa']*100:.1f}% "
                         f"(forecast: {r['g_pax']*100:.1f}%). Switch the signal source to see that plan.")
    elif mode == 'past':
        lines.append(f"**Reading (past only):** recent same-weekday behaviour is CR {r['c_pa']*100:.1f}% and ECR {r['e_pa']*100:.1f}%, versus a healthy "
                     f"{r['c_ref']*100:.1f}% / {r['e_ref']*100:.1f}%. The forecast is not used for this.")
        lines.append(f"**What the forecast would have said:** CR {r['c_fc']*100:.1f}% / ECR {r['e_fc']*100:.1f}% (driver opportunity {r['g_drv_fc']*100:.1f}% vs "
                     f"{r['g_drv']*100:.1f}% now, passenger {r['g_pax_fc']*100:.1f}% vs {r['g_pax']*100:.1f}% now).")
        lines.append(f"**Trend effect:** the 4-week trend multiplies driver value by x{r['trend_m_drv']:.2f} and passenger value by x{r['trend_m_pax']:.2f}.")
    else:
        lines.append(f"**Reading (blend {int((1 - ws) * 100)}% forecast / {int(ws * 100)}% recent):** CR forecast {r['c_fc']*100:.1f}%, recent {r['c_pa']*100:.1f}%, "
                     f"blended {r['c']*100:.1f}%. ECR forecast {r['e_fc']*100:.1f}%, recent {r['e_pa']*100:.1f}%, blended {r['e']*100:.1f}%. "
                     f"Healthy: {r['c_ref']*100:.1f}% / {r['e_ref']*100:.1f}%.")
        dd, dp = r['g_drv_pa'] - r['g_drv_fc'], r['g_pax_pa'] - r['g_pax_fc']
        if abs(dd) >= 0.01 or abs(dp) >= 0.01:
            who = []
            if abs(dd) >= 0.01:
                who.append(f"on drivers the forecast sees {r['g_drv_fc']*100:.1f}% and the past {r['g_drv_pa']*100:.1f}%")
            if abs(dp) >= 0.01:
                who.append(f"on passengers the forecast sees {r['g_pax_fc']*100:.1f}% and the past {r['g_pax_pa']*100:.1f}%")
            lines.append("**Where they disagree:** " + "; ".join(who) + ". The plan sits in between.")
        else:
            lines.append("**Where they disagree:** they do not, on this day.")
        lines.append(f"**Trend effect:** driver value x{r['trend_m_drv']:.2f}, passenger value x{r['trend_m_pax']:.2f} (half weight).")

    drv_bits = [f"CR {r['c']*100:.1f}% vs reference {r['c_ref']*100:.1f}% ({pp(r['c'], r['c_ref'])})"]
    if r['leak_sdr'] > 0.005:
        drv_bits.append(f"supply per eyeball {-r['leak_sdr']*100:.0f}% vs norm")
    pax_bits = [f"ECR {r['e']*100:.1f}% vs reference {r['e_ref']*100:.1f}% ({pp(r['e'], r['e_ref'])})"]
    if r['leak_eb'] > 0.005:
        pax_bits.append(f"eyeballs {-r['leak_eb']*100:.0f}% vs norm")
    lines.append(f"**Driver leak (fulfillment):** {'; '.join(drv_bits)} -> opportunity {r['g_drv']*100:.1f}% of the day (about {r['T']*r['g_drv']:,.0f} trips recoverable).")
    lines.append(f"**Passenger leak (conversion):** {'; '.join(pax_bits)} -> opportunity {r['g_pax']*100:.1f}% of the day (about {r['T']*r['g_pax']:,.0f} trips recoverable).")

    pull = r['share'] - r['fair_share']
    if abs(pull) < 0.01:
        why = "in line with its share of eyeballs"
    elif pull > 0:
        why = f"{pull*100:.1f}pp above its eyeball share because its leaks are wider and/or it is worth more per peso"
    else:
        why = f"{-pull*100:.1f}pp below its eyeball share because it leaks less or a safety/hold rule applies"
    lines.append(f"**Allocation:** ${r['x_total']:,.0f} ({r['share']*100:.1f}% of the remaining budget, {why}). "
                 f"Driver ${r['x_drv']:,.0f} / Passenger ${r['x_pax']:,.0f} ({r['split_drv']*100:.0f}/{(1-r['split_drv'])*100:.0f}), "
                 f"burn {r['burn_pct']:.2f}% of the day's GMV.")
    if summary.get('force_drv_share') is not None:
        lines.append(f"**Forced split:** the week is locked at {summary['force_drv_share']*100:.0f}/{(1-summary['force_drv_share'])*100:.0f} (DRV/PAX); "
                     "Nova still decides which days get more inside each channel.")
    if r['gain'] > 0:
        lines.append(f"**Modelled effect:** about +{r['gain']:,.0f} trips ({r['gain']/r['T']*100:.1f}% of the day).")
    return "\n\n".join(lines)


def week_insight(summary):
    """One short paragraph for the city owner."""
    s = summary
    src = {'forecast': "From the forecast, ", 'blend': "Blending forecast and recent behaviour, ", 'past': "Following recent behaviour only, "}.get(s['mode'], "")
    bits = [f"{src}Nova puts {s['macro_drv']*100:.0f}% of the remaining ${s['spent']:,.0f} on drivers and {(1-s['macro_drv'])*100:.0f}% on passengers."]
    if s.get('force_drv_share') is not None:
        bits.append("That split was forced by you; without it the leaks alone would suggest "
                    f"{s['leak_share_drv']*100:.0f}% drivers.")
    else:
        bits.append(f"{s['leak_share_drv']*100:.0f}% of the recoverable trips sit in fulfillment (CR and supply) and "
                    f"{(1-s['leak_share_drv'])*100:.0f}% in conversion (ECR and eyeballs).")
    if s['mode'] != 'forecast':
        tr = s['trend']
        bits.append(f"Recent 4-week trend: supply/CR {tr['drv']*100:+.1f}%, conversion/eyeballs {tr['pax']*100:+.1f}% deterioration "
                    f"({'blended with' if s['mode']=='blend' else 'replacing'} the forecast).")
    if s['cr_target'] > 0:
        bits.append(f"CR target {s['cr_target']*100:.0f}% is the reference for the driver leak.")
    if s['calls_target'] > 0:
        bits.append(f"Calls target {s['calls_target']:,.0f} vs projection {s['weekly_calls']:,.0f} scales passenger value x{s['calls_mult']:.2f}.")
    return " ".join(bits)
# =====================================================================
# MODULE 8: PURE HELPERS (no Streamlit): weeks, units, tables, statuses
# =====================================================================
import html as _html

VOL_COLS = ['trips', 'calls', 'eyeballs', 'tsh', 'cr', 'ecr', 'sdr', 'etr']
TYPE_ACT = '1. Actuals (History)'
TYPE_ORG = '2. Forecast (Organic ARIMA)'
TYPE_ANOM = '3. Forecast (Anomaly Engine)'
TYPE_ADJ = '4. Forecast (Adjusted)'
CHART_H = 420
ANOM_METRIC_ROWS = [('trips', 'Trips'), ('calls', 'Calls'), ('eyeballs', 'Eyeballs'), ('tsh', 'Supply hours (TSH)')]
ANOM_DERIVED_ROWS = [('cr', 'CR = trips / calls'), ('ecr', 'ECR = calls / eyeballs'),
                     ('etr', 'ETR = trips / eyeballs'), ('sdr', 'SDR = supply hours / eyeballs')]


def _day_lbl(d):
    return pd.Timestamp(d).strftime('%a %m-%d')


def _wow(cur, prev):
    return ((cur / prev) - 1) * 100 if prev else 0.0


def esc(x):
    return _html.escape(str(x), quote=False)


def html_table(headers, rows):
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="nova-tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def week_bounds(week, ref=None):
    """(Monday, Sunday) of ISO week `week`, in the year that lands closest to `ref` (today)."""
    mon = week_monday(week, ref)
    return mon, mon + timedelta(days=6)


def chart_window(week, ref=None):
    """
    Four full weeks ending on the Sunday of the selected week.
      ongoing week : 3 weeks of history + the selected week (history so far + the days still to come)
      next week    : 2 weeks of history, the ongoing week (history + forecast) and the selected week (forecast)
    """
    mon, sun = week_bounds(week, ref)
    return mon - timedelta(days=21), sun


def week_labels(start, end, ref=None):
    """One label per Monday-Sunday block of the chart window: W39, W40, Current (W41), W42."""
    ref = today_mx() if ref is None else ref
    rows = []
    cur = pd.Timestamp(start)
    while cur <= pd.Timestamp(end):
        wk = int(cur.isocalendar()[1])
        is_cur = cur <= ref <= cur + timedelta(days=6)
        rows.append({'mid': cur + timedelta(days=3, hours=12), 'label': f"Current (W{wk})" if is_cur else f"W{wk}"})
        cur += timedelta(days=7)
    return pd.DataFrame(rows, columns=['mid', 'label'])


def week_slice(df, week, ref=None, date_col='date'):
    mon, sun = week_bounds(week, ref)
    return df[(df[date_col] >= mon) & (df[date_col] <= sun)]


def agg_funnel(m):
    """Weekly funnel from SUMMED volumes (a ratio of sums, not the mean of daily ratios). Rates in %."""
    if m is None or m.empty:
        return None
    E, C, T, S = (float(m[c].sum()) for c in ('eyeballs', 'calls', 'trips', 'tsh'))
    return dict(eyeballs=E, calls=C, trips=T, tsh=S, gmv=float(m['gmv'].sum()),
                cr=T / C * 100 if C else 0.0, ecr=C / E * 100 if E else 0.0,
                etr=T / E * 100 if E else 0.0, sdr=S / E if E else 0.0)


def plan_week_budget(budget_pct, week_gmv, spent_usd):
    """
    UNIT BOUNDARY for the budget. budget_pct is what the user typed (6 means 6%).
    Money = (6 / 100) x GMV of the week (actual so far + forecast). Everything returned as money AND as % of week GMV.
    """
    total = pct_to_frac(budget_pct) * float(week_gmv)
    spent = float(spent_usd)
    remaining = max(0.0, total - spent)
    g = float(week_gmv)
    return dict(total_usd=total, spent_usd=spent, remaining_usd=remaining,
                budget_pct=float(budget_pct),
                spent_pct=frac_to_pct(spent / g) if g > 0 else 0.0,
                remaining_pct=frac_to_pct(remaining / g) if g > 0 else 0.0)


def build_past_burn_frame(hist_week, burn_df=None, manual=None):
    """
    Burn of the days already played, one row per actual day of the week.
      burn_df : Burn SoT frame (date, drv_burn, pax_burn, gmv, drv_pct, pax_pct) -> automatic
      manual  : {dow: (drv_pct, pax_pct)} typed by the user
    UNITS: the budget is a % of the Daily DB GMV, so the money already spent is ALSO (burn % x that day's Daily DB GMV).
    The raw Burn SoT money and Burn SoT GMV are kept (sot_usd, sot_gmv) only to reconcile the two sources.
    """
    rows = []
    burn_by_date = {}
    if burn_df is not None and not burn_df.empty:
        burn_by_date = {pd.Timestamp(r['date']): r for _, r in burn_df.iterrows()}
    for _, h in hist_week.sort_values('date').iterrows():
        d = pd.Timestamp(h['date'])
        g = float(h['gmv_real'])
        base = dict(date=d, day_gmv=g, sot_usd=0.0, sot_gmv=0.0, missing=False)
        if manual is not None:
            dp, pp = manual.get(str(d.dayofweek), (0.0, 0.0))
            dp, pp = float(dp or 0.0), float(pp or 0.0)
            rows.append({**base, 'drv_pct': dp, 'pax_pct': pp, 'source': 'Manual'})
        elif d in burn_by_date:
            b = burn_by_date[d]
            rows.append({**base, 'drv_pct': float(b['drv_pct']), 'pax_pct': float(b['pax_pct']), 'source': 'Burn SoT',
                         'sot_usd': float(b['drv_burn']) + float(b['pax_burn']), 'sot_gmv': float(b['gmv'])})
        else:
            rows.append({**base, 'drv_pct': 0.0, 'pax_pct': 0.0, 'source': 'Burn SoT', 'missing': True})
    cols = ['date', 'day_gmv', 'drv_pct', 'pax_pct', 'source', 'missing', 'sot_usd', 'sot_gmv']
    out = pd.DataFrame(rows, columns=cols)
    out['drv_usd'] = out['drv_pct'].map(pct_to_frac) * out['day_gmv']
    out['pax_usd'] = out['pax_pct'].map(pct_to_frac) * out['day_gmv']
    out['total_usd'] = out['drv_usd'] + out['pax_usd']
    out['total_pct'] = out['drv_pct'] + out['pax_pct']
    out['gmv_ratio'] = np.where(out['day_gmv'] > 0, out['sot_gmv'] / out['day_gmv'].where(out['day_gmv'] > 0), np.nan)
    return out


def gmv_mismatch_note(past, tol=0.05):
    """Text when Burn SoT's GMV does not match the Daily DB GMV (duplicated rows, other product scope...). '' if all fine."""
    if past is None or past.empty:
        return ""
    bad = past[(past['source'] == 'Burn SoT') & (~past['missing']) & past['gmv_ratio'].notna() & ((past['gmv_ratio'] - 1.0).abs() > tol)]
    if bad.empty:
        return ""
    r = bad.iloc[0]
    return (f"Burn SoT's GMV on {_day_lbl(r['date'])} is ${r['sot_gmv']:,.0f}, {r['gmv_ratio']:.2f}x the Daily DB GMV (${r['day_gmv']:,.0f}); "
            f"{len(bad)} played day(s) differ by more than {tol * 100:.0f}%. Nova keeps the burn % from Burn SoT and converts it with the Daily DB GMV, "
            f"the same base as your weekly budget, so the % you see and the money counted always agree. Raw Burn SoT money on that day: ${r['sot_usd']:,.0f}. "
            "A ratio close to 2.00x usually means duplicated rows in Burn SoT; a different ratio usually means a different product scope.")


def fmt_change(ratio):
    """Ratio of day vs baseline -> '+23.4%'. Missing baseline -> '-'."""
    if ratio is None or (isinstance(ratio, float) and not np.isfinite(ratio)):
        return "-"
    return f"{(float(ratio) - 1.0) * 100:+.1f}%"


def anomaly_status(date_str, today, runtime, anoms_on_now=True):
    """Where a saved anomaly stands. `runtime` is what the last dashboard run saw (horizon / toggle)."""
    d = pd.to_datetime(date_str, errors='coerce')
    if pd.isna(d): return "Invalid date"
    if d < today: return "Expired (past)"
    if not runtime: return "Saved"
    if not runtime.get('anoms_on', True): return "Saved (anomalies toggle is OFF)"
    hmax = pd.to_datetime(runtime.get('horizon_max'), errors='coerce')
    if pd.notna(hmax) and d > hmax: return "Waiting (beyond forecast horizon)"
    return "Active (in the forecast)"


def list_saved_anomalies(overrides, today, runtime=None, city=None):
    """Flat list of saved anomalies (all cities, or one) with status, multipliers, WoW/Wo2W and derived ratio multipliers."""
    rows = []
    for d, by_city in sorted((overrides or {}).items()):
        if not isinstance(by_city, dict): continue
        for c, node in by_city.items():
            if city is not None and c != city: continue
            if not isinstance(node, dict): continue
            raw = node.get('multipliers') if isinstance(node.get('multipliers'), dict) else {}
            mults = {m: safe_mult(raw.get(m)) for m in FUNNEL_METRICS}
            rows.append(dict(
                date=d, city=c, reference=node.get('reference_date', '?'), saved_on=node.get('saved_on', '?'),
                status=anomaly_status(d, today, runtime), multipliers=mults, derived=derive_ratio_mults(mults),
                wow=node.get('wow') if isinstance(node.get('wow'), dict) else {},
                wo2w=node.get('wo2w') if isinstance(node.get('wo2w'), dict) else {},
            ))
    return rows


def short_reason(r, mode='forecast'):
    """One-line 'why' for a day of the allocation table. Wording follows the signal source."""
    src = {'forecast': "fcst ", 'past': "recent ", 'blend': "blend "}.get(mode, "")
    bits = []
    if int(r['package']) > 0: bits.append(f"P{int(r['package'])}")
    if r['carried']: bits.append("carry-over")
    if r['leak_cr'] > 0.02: bits.append(f"{src}CR gap")
    if r['leak_sdr'] > 0.05: bits.append(f"{src}low supply")
    if r['leak_ecr'] > 0.02: bits.append(f"{src}ECR gap")
    if r['leak_eb'] > 0.05: bits.append(f"{src}low eyeballs")
    if mode != 'forecast':
        if r['trend_m_drv'] >= 1.05: bits.append("supply trend falling")
        if r['trend_m_pax'] >= 1.05: bits.append("conversion trend falling")
    return ", ".join(bits) if bits else "baseline"


def signal_comparison(results):
    """Side-by-side of the three signal sources. results: {mode: engine result}. Returns (DataFrame, notes list)."""
    modes = [m for m in ('forecast', 'blend', 'past') if results.get(m) is not None]
    if not modes:
        return pd.DataFrame(), []
    dates = results[modes[0]]['days']['date'].tolist()
    rows = [{"Day": "WEEK (remaining)", **{SIGNAL_LABELS[m]: f"DRV {results[m]['summary']['macro_drv'] * 100:.0f}% / PAX {(1 - results[m]['summary']['macro_drv']) * 100:.0f}%" for m in modes}}]
    notes = []
    for i, d in enumerate(dates):
        row = {"Day": _day_lbl(d)}
        for m in modes:
            r = results[m]['days'].iloc[i]
            row[SIGNAL_LABELS[m]] = (f"{r['burn_pct']:.2f}% ({r['split_drv'] * 100:.0f}/{(1 - r['split_drv']) * 100:.0f})" if r['x_total'] > 0 else "-")
        rows.append(row)
        if len(modes) > 1:
            burns = [results[m]['days'].iloc[i]['burn_pct'] for m in modes]
            splits = [results[m]['days'].iloc[i]['split_drv'] for m in modes]
            hi, lo = modes[int(np.argmax(burns))], modes[int(np.argmin(burns))]
            if min(burns) > 0 and (max(burns) / min(burns) - 1) >= 0.25:
                notes.append(f"{_day_lbl(d)}: {SIGNAL_LABELS[hi]} burns {max(burns):.2f}% vs {min(burns):.2f}% with {SIGNAL_LABELS[lo]}.")
            elif (max(splits) - min(splits)) >= 0.10:
                notes.append(f"{_day_lbl(d)}: the DRV share moves between {min(splits) * 100:.0f}% and {max(splits) * 100:.0f}% depending on the signal.")
    return pd.DataFrame(rows), notes


def package_reason(row):
    pkg = int(row['package'])
    if pkg == 0: return "No rain expected, so no weather action."
    if pkg == 1: return "Rain is light or moderate and does not touch the AM, PM or night peak windows, so business as usual."
    return row['alert'] or PACKAGE_STANCE.get(pkg, {}).get('note', '')


# =====================================================================
# MODULE 8b: VISUAL RENDERING (UI)
# =====================================================================
def render_day_diagnostics(df_historical, date_str, city_name, lat, lon):
    st.subheader(f"🕵️ Forensic Analysis - {city_name}")
    st.caption(f"Snapshot of target date: **{date_str}**")
    metrics = calculate_daily_lags(df_historical, date_str)
    if not metrics:
        st.error(f"⚠️ No operational historical data found for {date_str}.")
    else:
        st.markdown("### 📊 Performance vs Past (WoW and Wo2W)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Actual Trips", f"{int(metrics['trips']['val']):,}", f"{metrics['trips']['delta_wow']:+.1f}% WoW | {metrics['trips']['delta_wo2w']:+.1f}% Wo2W")
        c2.metric("Efficiency (CR)", f"{metrics['cr']['val']:.1f}%", f"{metrics['cr']['val'] - metrics['cr']['wow']:+.1f}pp WoW | {metrics['cr']['val'] - metrics['cr']['wo2w']:+.1f}pp Wo2W")
        c3.metric("Generated GMV", f"${int(metrics['gmv']['val']):,}", f"{metrics['gmv']['delta_wow']:+.1f}% WoW | {metrics['gmv']['delta_wo2w']:+.1f}% Wo2W")
        st.divider()

    df_climate_daily, _ = get_weather_forecast(lat, lon, city_name)
    if not df_climate_daily.empty:
        day_climate = df_climate_daily[df_climate_daily['date'] == pd.to_datetime(date_str).normalize()]
        if not day_climate.empty:
            temp = day_climate['Temp_Max'].values[0]
            status = day_climate['intensity_cat'].values[0]
            mm_rain = day_climate['rain_mm'].values[0]
            emoji = "⛈️" if "Tormenta" in status else ("🌧️" if "Fuerte" in status else ("🌦️" if "Moderada" in status or "Ligera" in status else "☀️"))
            st.markdown(f"### 🌡️ Satellite Weather Report ({date_str})")
            col1, col2 = st.columns(2)
            col1.metric("Max Temperature", f"{temp:.1f}°C")
            col2.metric("Precipitation / Status", f"{emoji} {status}", f"Accumulated: {mm_rain:.1f} mm", delta_color="off" if mm_rain > 0 else "normal")


def _forecast_plot_frame(df, type_label):
    out = df[['date'] + [c + '_forecast' for c in VOL_COLS]].copy()
    out.columns = ['date'] + VOL_COLS
    out['Type'] = type_label
    return out


def build_plot_frames(df_historical, df_org, df_anom, df_clim):
    """All the series of the continuous chart in one long frame (lines hooked to the last actual day)."""
    df_h = df_historical[['date'] + [c + '_real' for c in VOL_COLS]].copy()
    df_h.columns = ['date'] + VOL_COLS
    df_h['Type'] = TYPE_ACT
    df_f = _forecast_plot_frame(df_org, TYPE_ORG)
    df_c = _forecast_plot_frame(df_clim, TYPE_ADJ)

    df_e = pd.DataFrame()
    if 'has_anomaly' in df_anom.columns and df_anom['has_anomaly'].any():
        a_dates = df_anom.loc[df_anom['has_anomaly'], 'date']
        lo, hi = a_dates.min() - timedelta(days=1), a_dates.max() + timedelta(days=1)
        df_e = _forecast_plot_frame(df_anom[(df_anom['date'] >= lo) & (df_anom['date'] <= hi)], TYPE_ANOM)

    last_hist_date = df_h['date'].max()
    last_hist_row = df_h[df_h['date'] == last_hist_date]
    series = []
    for frame, label in ((df_f, TYPE_ORG), (df_e, TYPE_ANOM), (df_c, TYPE_ADJ)):
        if frame.empty: continue
        future = frame[frame['date'] > last_hist_date]
        if label != TYPE_ANOM and not last_hist_row.empty:
            connect = last_hist_row.copy()
            connect['Type'] = label
            future = pd.concat([connect, future], ignore_index=True)
        series.append(future)
    df_plot = pd.concat([df_h] + series, ignore_index=True)
    df_plot['date'] = pd.to_datetime(df_plot['date'])
    return df_plot


def render_timeline_charts(df_historical, df_org, df_anom, df_clim, week, flags=None):
    df_plot = build_plot_frames(df_historical, df_org, df_anom, df_clim)
    start, end = chart_window(week)
    df_plot = df_plot[(df_plot['date'] >= start) & (df_plot['date'] <= end)].copy()
    if df_plot.empty:
        st.info("No data inside the chart window for the selected week.")
        return

    df_mondays = pd.DataFrame({'date': pd.date_range(start, end, freq='7D')})
    week_lines = alt.Chart(df_mondays).mark_rule(color='gray', strokeDash=[5, 5], opacity=0.5, strokeWidth=1.5).encode(x='date:T')
    labels = alt.Chart(week_labels(start, end)).mark_text(baseline='top', fontWeight='bold', fontSize=12, opacity=0.75).encode(
        x='mid:T', y=alt.value(2), text='label:N')
    flag_layer = None
    if flags is not None and not flags.empty:
        fl = flags[(flags['date'] >= start) & (flags['date'] <= end)]
        if not fl.empty:
            flag_layer = alt.Chart(fl).mark_text(baseline='bottom', fontSize=16).encode(
                x='date:T', y=alt.value(CHART_H - 4), text='icon:N', tooltip=[alt.Tooltip('date:T', title='Day'), alt.Tooltip('tip:N', title='Adjustment')])
    x_axis = alt.X('date:T', title='', scale=alt.Scale(domain=[start - timedelta(hours=12), end + timedelta(hours=12)]),
                   axis=alt.Axis(format='%a %d', labelAngle=-45, tickCount='day'))

    domain_vals = [TYPE_ACT, TYPE_ORG, TYPE_ANOM, TYPE_ADJ]
    range_vals = ['#A9A9A9', '#A9A9A9', '#DC143C', '#1C86EE']
    dash_scale = alt.Scale(domain=domain_vals, range=[[0], [5, 5], [5, 5], [0]])
    opacity_scale = alt.Scale(domain=domain_vals, range=[1.0, 0.6, 0.8, 1.0])

    def create_chart(metric, title, color_override=None):
        rng = ['#A9A9A9', color_override, '#DC143C', '#1C86EE'] if color_override else range_vals
        scale = alt.Scale(domain=domain_vals, range=rng)
        line = alt.Chart(df_plot).mark_line(point=alt.OverlayMarkDef(size=60, opacity=1), strokeWidth=3).encode(
            x=x_axis, y=alt.Y(f'{metric}:Q', title=title, scale=alt.Scale(zero=False)),
            color=alt.Color('Type:N', scale=scale),
            strokeDash=alt.StrokeDash('Type:N', scale=dash_scale, legend=None),
            opacity=alt.Opacity('Type:N', scale=opacity_scale, legend=None),
            tooltip=['date:T', alt.Tooltip(f'{metric}:Q', format='.2f'), 'Type:N']
        )
        layers = [week_lines, line, labels] + ([flag_layer] if flag_layer is not None else [])
        return alt.layer(*layers).properties(height=CHART_H)

    st.caption("Adjusted forecast = organic ARIMA x rolling bias correction x scheduled anomalies x climate tree. "
               f"Window: {start.strftime('%b %d')} to {end.strftime('%b %d')} (always ends on a Sunday).")
    t1, t2, t3, t4 = st.tabs(["🌪️ Demand Funnel", "🚦 Conversion Funnel", "⏱️ Supply Funnel", "🚗 Overview"])
    with t1:
        st.altair_chart(create_chart('eyeballs', 'Eyeballs', '#1f77b4'), use_container_width=True)
        st.altair_chart(create_chart('calls', 'Calls', '#2ca02c'), use_container_width=True)
    with t2:
        st.altair_chart(create_chart('ecr', 'Eyeball-to-Call Rate (%)', '#9467bd'), use_container_width=True)
        st.altair_chart(create_chart('cr', 'Conversion Rate (%)', '#00CED1'), use_container_width=True)
    with t3:
        st.altair_chart(create_chart('tsh', 'Total Supply Hours'), use_container_width=True)
        st.altair_chart(create_chart('sdr', 'Supply/Demand Ratio', '#e377c2'), use_container_width=True)
    with t4:
        st.altair_chart(create_chart('trips', 'Total Trips', '#FF7F50'), use_container_width=True)
        st.altair_chart(create_chart('etr', 'End-to-End Conv. Rate (ETR %)', '#FF1493'), use_container_width=True)
    if flags is not None and not flags.empty:
        st.caption("Icons along the bottom of each chart mark adjusted days (hover for the reason): "
                   "🌧️ forecast changed by climate · ☔ rain package only (forecast unchanged) · ⚡ saved anomaly · 🚨 safety lock (package 6) · 🔥 heatwave.")


def render_weather_detail(df_climate, df_hourly, city_name, input_week, df_pkg=None):
    """Daily context on the left, hourly detail for one chosen day on the right. One selector flips both."""
    st.subheader(f"🌦️ 4. Weather Detail for {city_name}")
    today = today_mx()

    view = st.radio("Weather metric", ["🌧️ Rain", "🌡️ Temperature"], horizontal=True, label_visibility="collapsed", key="wx_view")
    show_rain = view.endswith("Rain")

    dfc = df_climate.copy()
    mon, sun = week_bounds(input_week)
    df_filtered = dfc[(dfc['date'] >= mon - timedelta(days=14)) & (dfc['date'] <= sun)].copy()
    if df_filtered.empty:
        df_filtered = dfc
    df_filtered['Type'] = np.where(df_filtered['date'] >= today, 'Forecast', 'History (Fact)')

    df_mondays = df_filtered[df_filtered['date'].dt.dayofweek == 0][['date']].drop_duplicates()
    week_lines = alt.Chart(df_mondays).mark_rule(color='gray', strokeDash=[5, 5], opacity=0.5, strokeWidth=1.5).encode(x='date:T')
    x_axis = alt.X('date:T', title='', axis=alt.Axis(format='%a %d', labelAngle=-45, tickCount='day'))
    base = alt.Chart(df_filtered).encode(x=x_axis)

    col_daily, col_hourly = st.columns([3, 2])

    # ---- Daily context ----
    with col_daily:
        if show_rain:
            bars = base.mark_bar(opacity=0.7, cornerRadiusEnd=4, size=20).encode(
                y=alt.Y('intensity_score:Q', title='Rain Intensity (0-100)'),
                color=alt.condition(alt.datum.Type == 'History (Fact)', alt.value('#B0C4DE'), alt.value('#1C86EE')),
                tooltip=['date:T', 'Type:N', 'intensity_cat:N', 'rain_mm:Q']
            )
            layers = [bars, week_lines]
            df_future = df_filtered[df_filtered['Type'] == 'Forecast']
            if not df_future.empty:
                prob_line = alt.Chart(df_future).mark_line(color='#000080', strokeWidth=3, strokeDash=[5, 5], point=alt.OverlayMarkDef(size=60)).encode(
                    x=x_axis, y=alt.Y('rain_prob:Q', title='Rain Intensity (0-100)'), tooltip=['date:T', 'rain_prob:Q'])
                layers.append(prob_line)
            st.altair_chart(alt.layer(*layers).properties(height=320), use_container_width=True)
            st.caption("Bars: rain intensity (light blue = observed, blue = forecast). Dashed line: forecast rain probability (%).")
        else:
            temp_line = base.mark_line(strokeWidth=4, point=alt.OverlayMarkDef(size=80, opacity=1)).encode(
                y=alt.Y('Temp_Max:Q', title='Max Temp (°C)', scale=alt.Scale(zero=False)),
                color=alt.Color('Type:N', scale=alt.Scale(domain=['History (Fact)', 'Forecast'], range=['#A9A9A9', '#FF4500']), legend=None),
                strokeDash=alt.condition(alt.datum.Type == 'Forecast', alt.value([5, 5]), alt.value([0])),
                tooltip=['date:T', 'Temp_Max:Q', 'Type:N']
            )
            layers = [week_lines, temp_line]
            if df_filtered['Temp_Max'].max() >= HEAT_THRESHOLD_C - 8:
                heat_rule = alt.Chart(pd.DataFrame({'t': [HEAT_THRESHOLD_C]})).mark_rule(color='#DC143C', strokeDash=[6, 4], strokeWidth=2).encode(
                    y=alt.Y('t:Q', title='Max Temp (°C)'))
                layers.append(heat_rule)
            st.altair_chart(alt.layer(*layers).properties(height=320), use_container_width=True)
            st.caption(f"Red line: {HEAT_THRESHOLD_C:.0f}°C heatwave threshold used by the Alert Center.")

    # ---- Hourly detail ----
    with col_hourly:
        valid_dates = df_filtered['date'].dt.strftime('%Y-%m-%d').tolist()
        today_str = today.strftime('%Y-%m-%d')
        default_idx = valid_dates.index(today_str) if today_str in valid_dates else 0
        selected_day = st.selectbox("Day to inspect (hourly)", valid_dates, index=default_idx, key="wx_day")
        df_h = df_hourly[df_hourly['date'] == selected_day].copy()
        day_row = df_filtered[df_filtered['date'] == pd.to_datetime(selected_day)]

        if df_h.empty:
            st.info("No hourly data for that day.")
        else:
            is_past = pd.to_datetime(selected_day) < today
            hour_scale = alt.Scale(domain=[-0.5, 23.5])
            x_hour = alt.X('hour:Q', title='Hour of the day', scale=hour_scale, axis=alt.Axis(values=list(range(0, 24, 3)), labelAngle=0))
            windows = pd.DataFrame({'start': [5.5, 11.5, 17.5], 'end': [10.5, 16.5, 22.5]})
            shade = alt.Chart(windows).mark_rect(opacity=0.10, color='#FFA500').encode(
                x=alt.X('start:Q', title='Hour of the day', scale=hour_scale), x2='end:Q')

            if show_rain:
                y_col = 'rain_mm' if is_past else 'rain_prob'
                y_title = 'Actual rain (mm)' if is_past else 'Rain probability (%)'
                bars_h = alt.Chart(df_h).mark_bar(size=12, opacity=0.85, color='#B0C4DE' if is_past else '#1C86EE').encode(
                    x=x_hour, y=alt.Y(f'{y_col}:Q', title=y_title),
                    tooltip=[alt.Tooltip('hour:Q', title='Hour'), alt.Tooltip('rain_prob:Q', title='Prob (%)'), alt.Tooltip('rain_mm:Q', title='Rain (mm)', format='.1f')]
                )
                st.altair_chart(alt.layer(shade, bars_h).properties(height=260), use_container_width=True)
            else:
                temp_h = alt.Chart(df_h).mark_line(color='#FF4500', strokeWidth=3, point=alt.OverlayMarkDef(size=50)).encode(
                    x=x_hour, y=alt.Y('temp:Q', title='Temperature (°C)', scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip('hour:Q', title='Hour'), alt.Tooltip('temp:Q', title='Temp (°C)', format='.1f')]
                )
                layers_h = [shade, temp_h]
                if df_h['temp'].max() >= HEAT_THRESHOLD_C - 8:
                    layers_h.append(alt.Chart(pd.DataFrame({'t': [HEAT_THRESHOLD_C]})).mark_rule(color='#DC143C', strokeDash=[6, 4], strokeWidth=2).encode(
                        y=alt.Y('t:Q', title='Temperature (°C)')))
                st.altair_chart(alt.layer(*layers_h).properties(height=260), use_container_width=True)
            st.caption("Shaded bands: AM peak (6-10h), PM peak (12-16h) and night peak (18-22h).")

        if not day_row.empty:
            r = day_row.iloc[0]
            st.caption(f"{r['intensity_cat']}, {r['rain_mm']:.1f} mm, flood risk {r['flood_risk']}, max {r['Temp_Max']:.1f}°C.")
        if df_pkg is not None and not df_pkg.empty:
            pkg_row = df_pkg[df_pkg['date'] == pd.to_datetime(selected_day)]
            if not pkg_row.empty:
                p = pkg_row.iloc[0]
                label = f"Package {int(p['package'])}: " if int(p['package']) > 0 else ""
                st.info(f"🧭 Nova prescription: {label}{p['prescription']}\n\nWhy: {package_reason(p)}")


def render_kpi_cards(m_t, m_1, w_target, city_name):
    st.subheader(f"📊 1. Market Weather & Funnel Diagnostics - W{w_target} ({city_name})")
    a, b = agg_funnel(m_t), agg_funnel(m_1)
    if a is None or b is None:
        st.info(f"There is not enough data to compare W{w_target} against the previous week.")
        return
    if len(m_t) < 7:
        st.caption(f"Only {len(m_t)} of 7 days of W{w_target} exist in the data and forecast, so the totals are partial.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Eyeballs", f"{int(a['eyeballs']):,}", f"{_wow(a['eyeballs'], b['eyeballs']):+.1f}% WoW")
    c2.metric("Calls", f"{int(a['calls']):,}", f"{_wow(a['calls'], b['calls']):+.1f}% WoW")
    c3.metric("Trips", f"{int(a['trips']):,}", f"{_wow(a['trips'], b['trips']):+.1f}% WoW")
    c4.metric("Supply Hours", f"{int(a['tsh']):,}", f"{_wow(a['tsh'], b['tsh']):+.1f}% WoW")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("ECR (Conversion)", f"{a['ecr']:.1f}%", f"{a['ecr'] - b['ecr']:+.1f}pp WoW")
    c6.metric("SDR (Balance)", f"{a['sdr']:.2f}", f"{a['sdr'] - b['sdr']:+.2f} WoW")
    c7.metric("CR (Fulfillment)", f"{a['cr']:.1f}%", f"{a['cr'] - b['cr']:+.1f}pp WoW")
    c8.metric("Projected GMV", f"${int(a['gmv']):,}", f"{_wow(a['gmv'], b['gmv']):+.1f}% WoW")


def rows_bias_active(ctx):
    return any(abs(f - 1.0) >= 0.005 for f in ctx['bias_factors'].values())


def build_alerts(params, ctx, m_t, df_w_daily, weather_ok):
    """Nova's reading of weather, temperature, anomalies and ARIMA drift, most severe first. Bodies are small HTML tables."""
    alerts = []
    df_clim = ctx['df_clim']
    climate_on = params['clima'] and weather_ok

    PKG_LEVEL = {6: 'critical', 3: 'warning', 4: 'warning', 5: 'warning', 2: 'info', 7: 'info'}
    LEVEL_RANK = {'critical': 3, 'warning': 2, 'info': 1}

    def package_group():
        """One compact banner for every weather package; the per-package tables live in an expander under it."""
        parts, details, worst = [], [], 'info'
        for pkg in (6, 3, 4, 5, 2, 7):
            sub = df_clim[df_clim['package'] == pkg]
            if sub.empty: continue
            lvl = PKG_LEVEL[pkg]
            if LEVEL_RANK[lvl] > LEVEL_RANK[worst]: worst = lvl
            parts.append(f"<b>P{pkg}</b> {', '.join(_day_lbl(d) for d in sub['date'])}")
            rows = [[f"<b>{_day_lbl(r['date'])}</b>", f"{esc(RAIN_LEVEL_LABEL.get(rain_level(r['intensity_cat']), r['intensity_cat']))} - {r['rain_mm']:.1f} mm",
                     esc(r['rain_windows'] or '-'), esc(r['flood_risk']), esc(PACKAGE_STANCE[pkg]['note'])] for _, r in sub.iterrows()]
            details.append(f"<b>Package {pkg}: {esc(PACKAGE_PRESCRIPTION[pkg])}</b>" +
                           html_table(["Day", "Rain", "Hits", "Flood risk", "What Nova does with the budget"], rows))
        if not parts: return
        n_days = int((df_clim['package'].isin(list(PKG_LEVEL))).sum())
        alerts.append({'level': worst, 'title': f"📦 Weather packages: {n_days} day(s) need an action",
                       'body': " · ".join(parts), 'details': details, 'details_label': "Package details (day, rain, peak hits, what Nova does)"})

    if not weather_ok:
        alerts.append({'level': 'warning', 'title': '⚠️ Weather feed unavailable',
                       'body': 'Open-Meteo did not answer. Rain adjustments, heat alerts and the weather detail are skipped in this run.'})

    if weather_ok:
        hot = df_w_daily[(df_w_daily['date'] >= today_mx()) & (df_w_daily['Temp_Max'] > HEAT_THRESHOLD_C)]
        if not hot.empty:
            rows = [[f"<b>{_day_lbl(r['date'])}</b>", f"{r['Temp_Max']:.1f}°C", f"+{r['Temp_Max'] - HEAT_THRESHOLD_C:.1f}°C over the threshold"] for _, r in hot.iterrows()]
            alerts.append({'level': 'critical',
                           'title': "🔥 Heatwave Detectada: Riesgo de colapso de supply por calor extremo. Usa el Anomaly Engine para inyectar datos históricos.",
                           'body': html_table(["Day", "Max temp", "Gap"], rows)})

    if climate_on:
        package_group()
        mcols = [f'mult_{m}' for m in FUNNEL_METRICS]
        adj = df_clim[(df_clim[mcols] != 1.0).any(axis=1)]
        if not adj.empty:
            rows = [[f"<b>{_day_lbl(r['date'])}</b>", esc(r['intensity_cat']), f"{r['rain_mm']:.1f} mm", f"P{int(r['package'])}",
                     f"x{r['mult_eyeballs']:.3f}", f"x{r['mult_calls']:.3f}", f"x{r['mult_trips']:.3f}", f"x{r['mult_tsh']:.3f}"] for _, r in adj.iterrows()]
            alerts.append({'level': 'info', 'title': f"🌧️ Climate shock: elasticity applied to the organic forecast on {len(adj)} day(s)",
                           'body': html_table(["Day", "Rain", "Amount", "Package", "Eyeballs", "Calls", "Trips", "Supply hrs"], rows)})

        # Rain is forecast but nothing multiplied it: say WHY instead of staying silent
        city_block = get_city_weather_block(load_weather_dict(), params['city_name'])
        neutral = df_clim[(df_clim['intensity_cat'].map(rain_level) != 'none') & ((df_clim[mcols] == 1.0).all(axis=1))]
        if not neutral.empty:
            rows = []
            for _, r in neutral.iterrows():
                node = resolve_weather_node(city_block, r['intensity_cat'], r['flood_risk']) if city_block else None
                why = ("city not in the weather dictionary" if not city_block else
                       "no dictionary entry for this rain bucket" if node is None else "dictionary cell is empty (null) for this bucket")
                rows.append([f"<b>{_day_lbl(r['date'])}</b>", esc(r['intensity_cat']), esc(r['flood_risk']), why])
            alerts.append({'level': 'info', 'title': f"ℹ️ Rain forecast but no climate multiplier on {len(neutral)} day(s) (forecast left as organic)",
                           'body': html_table(["Day", "Rain bucket", "Flood risk", "Why"], rows)})

    applied = ctx['anomalies_applied']
    if applied:
        rows = []
        for a in applied:
            mm, dd = a['mults'], a['derived']
            rows.append([f"<b>{_day_lbl(a['date'])}</b>", esc(a['reference_date']),
                         f"x{mm['trips']:.2f}", f"x{mm['calls']:.2f}", f"x{mm['eyeballs']:.2f}", f"x{mm['tsh']:.2f}",
                         f"x{dd['cr']:.2f}", f"x{dd['ecr']:.2f}", f"x{dd['etr']:.2f}", f"x{dd['sdr']:.2f}"])
        word = "anomaly" if len(applied) == 1 else "anomalies"
        alerts.append({'level': 'warning', 'title': f"⚡ Anomaly Engine: {len(applied)} scheduled {word} injected into the funnel",
                       'body': html_table(["Day", "Reference", "Trips", "Calls", "Eyeballs", "TSH", "CR", "ECR", "ETR", "SDR"], rows)})

    diag = {d['metric']: d for d in ctx['bias_diag']}
    rows = []
    for m, f in ctx['bias_factors'].items():
        if abs(f - 1.0) >= 0.005:
            rows.append([esc(METRIC_LABELS[m]), "under-estimating" if f > 1 else "over-estimating", f"{abs(f - 1) * 100:.1f}%", f"{diag[m]['n']} days", f"x{f:.3f}"])
    if rows:
        alerts.append({'level': 'info', 'title': '📈 Rolling Bias Corrector is active',
                       'body': html_table(["Metric", "ARIMA has been", "By", "Over the last", "Future days scaled by"], rows)})

    if not rows_bias_active(ctx):
        n_obs = max([d['n'] for d in ctx['bias_diag']] or [0])
        alerts.append({'level': 'info', 'title': '📈 Rolling Bias Corrector: not correcting',
                       'body': f"ARIMA memory has {n_obs} closed day(s) for this city (needs {BIAS_MIN_OBS}, all leaning the same way at least "
                               f"{int(BIAS_MIN_CONSISTENCY * 100)}% of the time). It fills up as days pass; a restart of the server empties it."})

    a = agg_funnel(m_t)
    if a is not None:
        if params["target_cr"] > 0 and a['cr'] < params["target_cr"]:
            alerts.append({'level': 'warning', 'title': '⚠️ Target CR at risk',
                           'body': f"Projected CR ({a['cr']:.1f}%) is {params['target_cr'] - a['cr']:.1f} pp below the target ({params['target_cr']:g}%)."})
        if params["target_calls"] > 0 and a['calls'] < params["target_calls"]:
            alerts.append({'level': 'warning', 'title': '⚠️ Target Calls at risk',
                           'body': f"Projection ({int(a['calls']):,}) is missing {int(params['target_calls'] - a['calls']):,} calls to reach the target ({int(params['target_calls']):,})."})
    return alerts


def render_alert_center(alerts):
    st.subheader("🚨 2. Alert Center")
    if not alerts:
        alerts = [{'level': 'ok', 'title': '✅ All clear',
                   'body': 'No weather hazards, heat, scheduled anomalies or ARIMA drift detected in the forecast horizon.'}]
    for a in alerts:
        st.markdown(f'<div class="nova-alert {a["level"]}"><div class="nova-alert-title">{a["title"]}</div><div class="nova-alert-body">{a["body"]}</div></div>', unsafe_allow_html=True)
        if a.get('details'):
            with st.expander(a.get('details_label', 'Details')):
                for block in a['details']:
                    st.markdown(block, unsafe_allow_html=True)


def render_anomaly_flash(flash, ctx, params):
    """Feedback right after saving an anomaly: did it land in the forecast the user is looking at?"""
    dates = [pd.Timestamp(d) for d in flash['dates']]
    day_txt = ", ".join(_day_lbl(d) for d in dates)
    if flash['city'] != params['city_name']:
        msg = f"⚡ Anomaly saved for **{flash['city']}** ({day_txt}). You are viewing {params['city_name']}, so switch city to see its effect."
        st.info(msg)
        st.toast(f"Anomaly saved for {flash['city']}.", icon="⚡")
        return
    applied = {pd.Timestamp(a['date']) for a in ctx['anomalies_applied']}
    horizon_max = ctx['df_org']['date'].max()
    ok = [d for d in dates if d in applied]
    if not params.get('use_anomalies', True):
        st.warning(f"⚡ Anomaly saved for {day_txt}, but **Apply scheduled anomalies is OFF** so it is not in this forecast.")
    elif len(ok) == len(dates):
        st.success(f"⚡ Anomaly saved and **applied** to the forecast: {day_txt} (see the Alert Center and the ⚡ icons on the charts).")
        st.toast("Anomaly saved and applied.", icon="⚡")
    else:
        late = [d for d in dates if d not in applied]
        st.warning(f"⚡ Anomaly saved, but {', '.join(_day_lbl(d) for d in late)} is outside the forecast horizon (last forecast day: "
                   f"{_day_lbl(horizon_max)}). It will apply automatically once the forecast reaches it.")


def render_allocation_engine(params, ctx, df_historical, m_t, w_target):
    """Allocation Engine: weekly DRV/PAX split, daily plan, reasons and past burn. Only called when budget > 0."""
    st.subheader("💊 5. Allocation Engine (Budget Manager)")
    mon, sun = week_bounds(w_target)
    df_clim = ctx['df_clim']
    hist_week = df_historical[(df_historical['date'] >= mon) & (df_historical['date'] <= sun)]
    fcst_week = df_clim[(df_clim['date'] >= mon) & (df_clim['date'] <= sun)].copy().reset_index(drop=True)
    week_gmv = float(m_t['gmv'].sum()) if not m_t.empty else 0.0

    # ---------------- past burn (automatic from Burn SoT, manual only as a fallback)
    past = pd.DataFrame(columns=['date', 'drv_usd', 'pax_usd', 'drv_pct', 'pax_pct', 'source', 'missing', 'total_usd', 'total_pct'])
    if not hist_week.empty:
        if params.get('manual_burn'):
            past = build_past_burn_frame(hist_week, manual=params.get('manual_past') or {})
        else:
            try:
                burn_df = get_past_burn(params['city'], mon.strftime('%Y-%m-%d'), hist_week['date'].max().strftime('%Y-%m-%d'))
                past = build_past_burn_frame(hist_week, burn_df=burn_df)
            except Exception as e:
                st.warning(f"⚠️ Could not read Burn SoT ({e}). Past burn is counted as 0. Turn on the manual override in 💰 Budget & Allocation to type it yourself.")
                past = build_past_burn_frame(hist_week, burn_df=pd.DataFrame())
    spent_usd = float(past['total_usd'].sum()) if not past.empty else 0.0
    plan = plan_week_budget(params['budget'], week_gmv, spent_usd)

    if week_gmv <= 0:
        st.info("There is no GMV for this week yet, so the budget cannot be converted to money.")
        return

    # ---------------- budget header (always in %, with the money behind it)
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Weekly budget", f"{plan['budget_pct']:.2f}% of GMV", f"${plan['total_usd']:,.0f}", delta_color="off")
    h2.metric("Already burned", f"{plan['spent_pct']:.2f}%", f"${plan['spent_usd']:,.0f}", delta_color="off")
    h3.metric("Left to allocate", f"{plan['remaining_pct']:.2f}%", f"${plan['remaining_usd']:,.0f}", delta_color="off")
    h4.metric("Week GMV (actual + forecast)", f"${week_gmv:,.0f}")

    if not past.empty:
        note = gmv_mismatch_note(past)
        if note:
            st.warning("⚠️ Burn SoT and Daily DB disagree on GMV. " + note)
        with st.expander(f"🧾 Burn already spent this week ({past['source'].iloc[0]})", expanded=True):
            show = pd.DataFrame({
                "Day": past['date'].map(_day_lbl),
                "DRV burn (%)": past['drv_pct'].map(lambda v: f"{v:.2f}%"),
                "PAX burn (%)": past['pax_pct'].map(lambda v: f"{v:.2f}%"),
                "Total burn (%)": past['total_pct'].map(lambda v: f"{v:.2f}%"),
                "Split DRV/PAX": [f"{d / t * 100:.0f}/{p / t * 100:.0f}" if t > 0 else "-" for d, p, t in zip(past['drv_usd'], past['pax_usd'], past['total_usd'])],
                "Day GMV ($)": past['day_gmv'].map(lambda v: f"${v:,.0f}"),
                "Spent ($)": past['total_usd'].map(lambda v: f"${v:,.0f}"),
            })
            st.dataframe(show, hide_index=True, use_container_width=True)
            if past['missing'].any():
                st.caption("Days with no Burn SoT row are counted as 0: " + ", ".join(past.loc[past['missing'], 'date'].map(_day_lbl)) + ".")

    if fcst_week.empty:
        st.success("✅ The week has ended (or has no forecast days). No future budget to allocate.")
        return
    if plan['remaining_usd'] <= 0:
        st.warning("The weekly budget is already fully spent, so there is nothing left to allocate.")
        return

    # ---------------- run the engine (money in, money out; % only at this boundary)
    try:
        beta_info = get_response_betas(params['city'])
        betas = {k: float(beta_info[k]['beta']) for k in ('drv', 'pax')}
    except Exception:
        beta_info, betas = None, None
    res = run_allocation_engine(
        fcst_week, df_historical, plan['remaining_usd'], mode=params['signal'],
        cr_target=pct_to_frac(params['target_cr']), calls_target=float(params['target_calls']),
        weekly_calls=float(m_t['calls'].sum()),
        force_drv_share=pct_to_frac(params['split']) if params['force'] else None,
        safe_mode=params['safe_mode'], betas=betas)
    if res is None:
        st.info("The engine had nothing to allocate.")
        return
    d, s = res['days'], res['summary']

    # ---------------- weekly split and effect
    drv_all = float(past['drv_usd'].sum()) + s['drv_total'] if not past.empty else s['drv_total']
    pax_all = float(past['pax_usd'].sum()) + s['pax_total'] if not past.empty else s['pax_total']
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Split for the remaining days", f"DRV {s['macro_drv'] * 100:.0f}% / PAX {(1 - s['macro_drv']) * 100:.0f}%",
              f"{frac_to_pct(s['spent'] / week_gmv):.2f}% of week GMV (${s['spent']:,.0f})", delta_color="off")
    k2.metric("Whole-week split (past + plan)", f"DRV {drv_all / (drv_all + pax_all) * 100:.0f}% / PAX {pax_all / (drv_all + pax_all) * 100:.0f}%" if (drv_all + pax_all) > 0 else "-",
              f"DRV {frac_to_pct(drv_all / week_gmv):.2f}% + PAX {frac_to_pct(pax_all / week_gmv):.2f}% of GMV", delta_color="off")
    k3.metric("Modelled extra trips", f"+{s['gain']:,.0f}", f"+{s['gain_pct']:.2f}% of the remaining days", delta_color="off")
    k4.metric("Cost per extra trip", f"${s['cpit']:,.2f}" if np.isfinite(s['cpit']) else "-", f"{s['closed'] * 100:.0f}% of the opportunity closed", delta_color="off")
    st.info("📡 " + signal_note(s))
    st.info("🧠 " + week_insight(s))
    if s['leftover'] > 1:
        st.warning(f"${s['leftover']:,.0f} ({frac_to_pct(s['leftover'] / week_gmv):.2f}% of week GMV) could not be placed because the daily cap "
                   f"({SAFE_CAP_FRAC * 100:.0f}% of each day's GMV) is binding. Turn off Safe Capping or lower the budget to use it.")
    elif s['caps_binding']:
        st.caption(f"Safe Capping is binding on at least one day (max {SAFE_CAP_FRAC * 100:.0f}% of that day's GMV).")

    # ---------------- week table: played days + plan
    total_week_money = s['spent'] + spent_usd
    rows, chart_rows = [], []
    for _, r in past.iterrows():
        t = r['total_usd']
        rows.append({"Day": _day_lbl(r['date']), "Status": "Actual", "Why": "Burn SoT" if r['source'] == 'Burn SoT' else "manual entry",
                     "Split DRV/PAX": f"{r['drv_usd'] / t * 100:.0f}/{r['pax_usd'] / t * 100:.0f}" if t > 0 else "-",
                     "DRV burn (%)": f"{r['drv_pct']:.2f}%", "PAX burn (%)": f"{r['pax_pct']:.2f}%", "Total burn (%)": f"{r['total_pct']:.2f}%",
                     "Share of week spend": f"{t / total_week_money * 100:.1f}%" if total_week_money > 0 else "-", "Extra trips": "-"})
        chart_rows += [dict(Day=_day_lbl(r['date']), Channel='DRV', Burn=r['drv_pct'], Status='Actual', order=r['date']),
                       dict(Day=_day_lbl(r['date']), Channel='PAX', Burn=r['pax_pct'], Status='Actual', order=r['date'])]
    for _, r in d.iterrows():
        rows.append({"Day": _day_lbl(r['date']), "Status": "Plan", "Why": short_reason(r, s['mode']),
                     "Split DRV/PAX": f"{r['split_drv'] * 100:.0f}/{(1 - r['split_drv']) * 100:.0f}" if r['x_total'] > 0 else "-",
                     "DRV burn (%)": f"{r['drv_pct']:.2f}%", "PAX burn (%)": f"{r['pax_pct']:.2f}%", "Total burn (%)": f"{r['burn_pct']:.2f}%",
                     "Share of week spend": f"{r['x_total'] / total_week_money * 100:.1f}%" if total_week_money > 0 else "-", "Extra trips": f"+{r['gain']:,.0f}"})
        chart_rows += [dict(Day=_day_lbl(r['date']), Channel='DRV', Burn=r['drv_pct'], Status='Plan', order=r['date']),
                       dict(Day=_day_lbl(r['date']), Channel='PAX', Burn=r['pax_pct'], Status='Plan', order=r['date'])]
    st.markdown("#### 📅 Week plan (burn as % of each day's GMV)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    df_c = pd.DataFrame(chart_rows)
    order = list(dict.fromkeys(df_c.sort_values('order')['Day']))
    bars = alt.Chart(df_c).mark_bar().encode(
        x=alt.X('Day:N', sort=order, title='', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Burn:Q', title="Burn (% of the day's GMV)"),
        color=alt.Color('Channel:N', scale=alt.Scale(domain=['DRV', 'PAX'], range=['#1C86EE', '#FF7F50'])),
        opacity=alt.condition(alt.datum.Status == 'Actual', alt.value(0.45), alt.value(1.0)),
        tooltip=['Day:N', 'Channel:N', alt.Tooltip('Burn:Q', format='.2f'), 'Status:N'])
    st.altair_chart(bars.properties(height=260), use_container_width=True)
    st.caption("Faded bars are money already spent (Burn SoT). Solid bars are Nova's plan for the days that remain.")

    # ---------------- explanations
    st.markdown("#### 🔍 Why this allocation?")
    labels = [f"{_day_lbl(r['date'])}  -  ${r['x_total']:,.0f}  ({r['split_drv'] * 100:.0f}/{(1 - r['split_drv']) * 100:.0f})" for _, r in d.iterrows()]
    pick = st.selectbox("Day to explain", range(len(labels)), format_func=lambda i: labels[i], key="alloc_explain_day")
    row = d.iloc[pick]
    fw = fcst_week.iloc[pick]
    weather_txt = f"{fw['intensity_cat']}, {fw['rain_mm']:.1f} mm, flood risk {fw['flood_risk']}" if str(fw['intensity_cat']) != 'No Rain' else ""
    st.markdown(explain_day(row, s, weather_txt, PACKAGE_PRESCRIPTION.get(int(row['package']), "")))

    with st.expander("🔀 How the signal source changes the plan (Forecast vs Blend vs Past)"):
        st.caption("Same budget, same targets, same weather: only the signal changes. Each cell is the day's total burn as % of its GMV and the DRV/PAX split.")
        others = {s['mode']: res}
        for m in ('forecast', 'blend', 'past'):
            if m not in others:
                others[m] = run_allocation_engine(
                    fcst_week, df_historical, plan['remaining_usd'], mode=m,
                    cr_target=pct_to_frac(params['target_cr']), calls_target=float(params['target_calls']),
                    weekly_calls=float(m_t['calls'].sum()),
                    force_drv_share=pct_to_frac(params['split']) if params['force'] else None,
                    safe_mode=params['safe_mode'], betas=betas)
        cmp_df, cmp_notes = signal_comparison(others)
        st.dataframe(cmp_df, hide_index=True, use_container_width=True)
        for n in cmp_notes:
            st.caption("• " + n)
        if not cmp_notes:
            st.caption("The three signal sources lead to a very similar plan this week.")

    with st.expander("🧮 Opportunity map (what the engine sees on each day)"):
        st.caption("Leaks are how far each part of the funnel sits below its healthy reference (same weekday, last 4 weeks, or your CR target). "
                   "Negative means the day is doing better than normal. Value x is what weather, trend and calls target add to a peso on that channel.")
        st.dataframe(pd.DataFrame({
            "Day": d['date'].map(_day_lbl),
            "CR": [f"{c * 100:.1f}% vs {rf * 100:.1f}%" for c, rf in zip(d['c'], d['c_ref'])],
            "ECR": [f"{e * 100:.1f}% vs {rf * 100:.1f}%" for e, rf in zip(d['e'], d['e_ref'])],
            "CR gap": (d['leak_cr'] * 100).map(lambda v: f"{v:+.1f}%"),
            "Supply gap": (d['leak_sdr'] * 100).map(lambda v: f"{v:+.1f}%"),
            "ECR gap": (d['leak_ecr'] * 100).map(lambda v: f"{v:+.1f}%"),
            "Eyeball gap": (d['leak_eb'] * 100).map(lambda v: f"{v:+.1f}%"),
            "DRV opportunity": (d['g_drv'] * 100).map(lambda v: f"{v:.1f}%"),
            "  by forecast / by past": [f"{a * 100:.1f}% / {b * 100:.1f}%" for a, b in zip(d['g_drv_fc'], d['g_drv_pa'])],
            "PAX opportunity": (d['g_pax'] * 100).map(lambda v: f"{v:.1f}%"),
            " by forecast / by past": [f"{a * 100:.1f}% / {b * 100:.1f}%" for a, b in zip(d['g_pax_fc'], d['g_pax_pa'])],
            "DRV value x": d['m_drv'].map(lambda v: f"{v:.2f}"),
            "PAX value x": d['m_pax'].map(lambda v: f"{v:.2f}"),
        }), hide_index=True, use_container_width=True)
        if beta_info:
            def _b(k):
                x = beta_info[k]
                return f"{x['beta']:.2f} ({x['source']}" + (f", t={x['t']:.1f}, {x['n']} days" if x.get('t') is not None else "") + ")"
            st.caption(f"Response strength (lift per 100% of GMV burned): DRV {_b('drv')}, PAX {_b('pax')}. "
                       "The prior is used whenever history cannot support a calibrated value.")
        st.caption(f"Signal source: {SIGNAL_LABELS.get(s['mode'], s['mode'])}.")


# =====================================================================
# MODULE 9: COMMAND CENTER (TOP BAR, NO SIDEBAR)
# =====================================================================
def retrain_models():
    if client is None:
        st.error("⚠️ BigQuery client is not available.")
        return
    try:
        with st.spinner("Training 5 ARIMA models... this will take a few minutes."):
            model_names = ['trips', 'gmv', 'calls', 'eyeballs', 'supply']
            columns = ['trips', 'gmv', 'calls', 'eyeballs', 'supply_hours']
            for mod, col in zip(model_names, columns):
                q = f"CREATE OR REPLACE MODEL `valid-sol-477221-e8.nova.{mod}_arima_baseline` OPTIONS(MODEL_TYPE = 'ARIMA_PLUS', TIME_SERIES_TIMESTAMP_COL = 'date_value', TIME_SERIES_DATA_COL = '{col}', TIME_SERIES_ID_COL = 'city_id', DATA_FREQUENCY = 'DAILY', HORIZON = 14, HOLIDAY_REGION = 'MX') AS SELECT date_value, CAST(city_id AS STRING) AS city_id, {col} FROM {REAL_TABLE} WHERE {col} IS NOT NULL AND date_value >= '2024-01-01' AND product = 'Managed Products'"
                client.query(q).result()
            st.cache_data.clear()
        st.success("✅ All 5 Models were updated successfully!")
        st.toast("All 5 models were updated.", icon="✅")
    except Exception as e:
        st.error(f"❌ Retraining failed: {e}")


def handle_save_anomaly(city_name, city_id, future_val, ref_val):
    future_days, ref_days = expand_dates(future_val), expand_dates(ref_val)
    today = today_mx()
    if not future_days or not ref_days:
        st.warning("Finish picking both date ranges first.")
    elif len(future_days) != len(ref_days):
        st.error(f"The future range has {len(future_days)} day(s) and the reference range has {len(ref_days)}. They must match.")
    elif len(future_days) > ANOMALY_MAX_DAYS:
        st.error(f"Pick at most {ANOMALY_MAX_DAYS} days per anomaly.")
    elif min(future_days) < today:
        st.error("Future dates must be today or later.")
    elif max(ref_days) >= today:
        st.error("Reference dates must already be in the past.")
    else:
        try:
            with st.spinner("Computing WoW and Wo2W from BigQuery..."):
                save_scheduled_anomaly(city_name, city_id, future_days, ref_days)
        except Exception as e:
            st.error(f"❌ Could not save the anomaly: {e}")
            return
        st.session_state['anom_flash'] = {'city': city_name, 'dates': [d.strftime('%Y-%m-%d') for d in future_days]}
        st.rerun()


def render_anomaly_detail(row):
    """WoW / Wo2W / multiplier for the four volumes plus the four derived ratios of one saved anomaly."""
    rows = []
    for key, label in ANOM_METRIC_ROWS:
        rows.append({"Metric": label, "WoW": fmt_change(row['wow'].get(key)), "Wo2W": fmt_change(row['wo2w'].get(key)),
                     "Applied multiplier": f"x{row['multipliers'][key]:.3f}"})
    for key, label in ANOM_DERIVED_ROWS:
        rows.append({"Metric": label, "WoW": "derived", "Wo2W": "derived", "Applied multiplier": f"x{row['derived'][key]:.3f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_command_center(current_week, current_date):
    today_dow = current_date.dayofweek
    today_date = today_mx().date()
    days_en = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    try:
        panel = st.container(border=True, key="nova_cc")  # key adds the .st-key-nova_cc class (Streamlit >= 1.39)
    except TypeError:
        panel = st.container(border=True)
    with panel:
        st.markdown('<div class="nova-cc-marker"></div><div class="nova-cc-title">🧠 Nova Command Center</div>'
                    '<div class="nova-cc-sub">Budget allocation engine powered by casually-informed AI</div>', unsafe_allow_html=True)
        legend_slot = st.empty()  # filled once every widget below has a value; lives in the header so it can never spill out of the panel
        col_city, col_budget, col_env, col_saved, col_sys = st.columns(5)

        # ---- City & Week (nothing else) ----
        with col_city.popover("🏙️ City & Week"):
            p_city_name = st.selectbox("City", list(CITIES_DICT.keys()), key="cc_city")
            p_week = int(st.number_input("Calendar Week", value=int(current_week), min_value=1, max_value=53, key="cc_week"))

        # ---- Budget & Allocation (every target lives here; 0 keeps the app in Decoupled mode) ----
        with col_budget.popover("💰 Budget & Allocation"):
            st.caption("Everything here is typed in % (or weekly calls). Nova converts it to money and real rates internally.")
            p_budget = st.number_input("Weekly budget (% of week GMV)", value=0.0, min_value=0.0, step=0.5, key="cc_budget",
                                       help="0 = Decoupled mode: only the forecast is shown. Above 0 unlocks the Allocation Engine. "
                                            "6 means 6% of the GMV of the selected week.")
            p_cr = st.number_input("CR target (%)", value=0.0, min_value=0.0, max_value=100.0, step=1.0, key="cc_cr",
                                   help="0 = off. Below target: more weight to drivers. Above target: less.")
            p_calls = st.number_input("Weekly calls target", value=0, min_value=0, step=1000, key="cc_calls",
                                      help="0 = off. Projection above target: less weight to passengers. Below: more.")
            p_signal = st.radio("Signal source", list(SIGNAL_LABELS.keys()), format_func=lambda k: SIGNAL_LABELS[k], key="cc_signal",
                                help="Forecast: what the funnel is projected to do. Blend: forecast plus the last 4 weeks' trend. "
                                     "Past trend only: ignore the forecast state and follow recent behaviour.")
            p_safe_mode = st.toggle("Safe Capping (max 8% of a day's GMV)", value=True, key="cc_safe")
            p_force = st.toggle("Force weekly DRV/PAX split", key="cc_force",
                                help="Final modifier: the week follows your split, each day still gets its own shape.")
            p_split = None
            if p_force:
                p_split = st.slider("Driver share (%)", 0, 100, 50, key="cc_split")
                st.caption(f"Forced split: DRV {p_split}% / PAX {100 - p_split}%")

            st.divider()
            st.caption("🔄 Past days of the ongoing week are read automatically from Burn SoT.")
            p_manual = st.toggle("Manual burn override (only if Burn SoT fails)", value=False, key="cc_manual")
            manual_past = {}
            if p_manual and p_week == current_week and today_dow > 0:
                for i in range(today_dow):
                    c_a, c_b = st.columns(2)
                    manual_past[str(i)] = (c_a.number_input(f"{days_en[i][:3]} DRV burn (%)", value=0.0, step=0.1, key=f"cc_mdrv_{i}"),
                                           c_b.number_input(f"{days_en[i][:3]} PAX burn (%)", value=0.0, step=0.1, key=f"cc_mpax_{i}"))

        # ---- Environment & Anomalies ----
        with col_env.popover("🌦️ Climate / Anomalies"):
            p_climate = st.toggle("Enable Climate Tree (rain adjustments and packages)", value=True, key="cc_clima")
            p_anoms = st.toggle("Apply scheduled anomalies", value=True, key="cc_anoms")
            st.divider()
            st.markdown("**⚡ New anomaly**")
            st.caption("Pick the future day(s) you expect to be unusual and the past day(s) that looked the same. Nova stores how the past day(s) moved "
                       "versus the same weekday 1 and 2 weeks earlier (trips, calls, eyeballs, supply hours) and replays that on the future day(s). "
                       "CR, ECR, ETR and SDR are derived from those volumes.")
            f_val = st.date_input("📅 Future date(s)", value=(today_date + timedelta(days=1), today_date + timedelta(days=1)),
                                  min_value=today_date, key="cc_anom_future")
            r_val = st.date_input("🕰️ Past reference date(s)", value=(today_date - timedelta(days=1), today_date - timedelta(days=1)),
                                  max_value=today_date - timedelta(days=1), key="cc_anom_ref")
            if st.button("💾 Save anomaly", key="cc_anom_save", use_container_width=True):
                handle_save_anomaly(p_city_name, CITIES_DICT[p_city_name], f_val, r_val)

        # ---- Saved Anomalies ----
        with col_saved.popover("📋 Saved Anomalies"):
            scope = st.radio("Show", ["This city", "All cities"], horizontal=True, key="sv_scope")
            runtime = st.session_state.get('nova_runtime')
            saved = list_saved_anomalies(purge_past_overrides(), today_mx(), runtime, p_city_name if scope == "This city" else None)
            if not saved:
                st.info("No saved anomalies. Create one in 🌦️ Environment & Anomalies.")
            else:
                st.dataframe(pd.DataFrame([{
                    "Date": r['date'], "City": r['city'], "Reference": r['reference'], "Status": r['status'],
                    "Trips x": round(r['multipliers']['trips'], 2), "Calls x": round(r['multipliers']['calls'], 2),
                    "Eyeballs x": round(r['multipliers']['eyeballs'], 2), "TSH x": round(r['multipliers']['tsh'], 2),
                    "CR x": round(r['derived']['cr'], 2), "ECR x": round(r['derived']['ecr'], 2),
                    "ETR x": round(r['derived']['etr'], 2), "SDR x": round(r['derived']['sdr'], 2)} for r in saved]),
                    hide_index=True, use_container_width=True)
                names = [f"{r['date']} - {r['city']}" for r in saved]
                pick = st.selectbox("Inspect WoW and Wo2W", range(len(saved)), format_func=lambda i: names[i], key="sv_pick")
                render_anomaly_detail(saved[pick])
                to_remove = st.multiselect("Remove saved anomalies", names, key="sv_remove")
                if to_remove and st.button("🗑️ Remove selected", key="sv_remove_btn", use_container_width=True):
                    by_city = {}
                    for n in to_remove:
                        dd, cc = n.split(" - ", 1)
                        by_city.setdefault(cc, []).append(dd)
                    for cc, ds in by_city.items():
                        remove_scheduled_anomalies(cc, ds)
                    st.toast("Anomalies removed.", icon="🗑️")
                    st.rerun()

        # ---- System Maintenance ----
        with col_sys.popover("⚙️ System Maintenance"):
            st.caption("Clear the cache to re-download BigQuery and weather data. Retrain to refresh the 5 ARIMA models.")
            if st.button("🧹 Clear Cache (Refresh)", key="cc_clear", use_container_width=True):
                st.cache_data.clear()
                st.success("✅ Cache cleared successfully.")
                st.toast("Cache cleared.", icon="🧹")
            if st.button("🧠 Retrain Models in BQ", key="cc_retrain", use_container_width=True):
                retrain_models()

        bits = [f"Showing <b>{esc(p_city_name)}</b>", f"week <b>{p_week}</b>",
                f"budget <b>{p_budget:g}% of GMV</b>" if p_budget > 0 else "<b>no budget</b> (forecast only)"]
        if p_budget > 0:
            bits.append(f"signal <b>{esc(SIGNAL_LABELS[p_signal])}</b>")
            if p_cr > 0: bits.append(f"CR target <b>{p_cr:g}%</b>")
            if p_calls > 0: bits.append(f"calls target <b>{int(p_calls):,}</b>")
            if p_force: bits.append(f"forced split <b>{p_split}/{100 - p_split}</b>")
            bits.append("safe capping <b>on</b>" if p_safe_mode else "safe capping <b>off</b>")
        bits.append(f"climate tree <b>{'on' if p_climate else 'off'}</b>")
        bits.append(f"anomalies <b>{'on' if p_anoms else 'off'}</b>")
        legend_slot.markdown('<div class="nova-status">' + " · ".join(bits) + '</div>', unsafe_allow_html=True)

    return {
        "city": CITIES_DICT[p_city_name], "city_name": p_city_name, "week": p_week,
        "budget": float(p_budget), "target_calls": float(p_calls), "target_cr": float(p_cr),
        "signal": p_signal, "clima": p_climate, "use_anomalies": p_anoms,
        "force": p_force, "split": p_split, "safe_mode": p_safe_mode,
        "manual_burn": bool(p_manual), "manual_past": manual_past,
    }


# =====================================================================
# MODULE 10: DASHBOARD FLOW (KPIs > Alerts > Forecast > Weather > Allocation)
# =====================================================================
def render_dashboard(params):
    city_name, city_id = params['city_name'], params['city']
    lat, lon = COORDINATES[city_name]

    with st.spinner("Downloading full funnel from BigQuery..."):
        try:
            df_historical, df_arima_all = get_city_data(city_id)
        except Exception as e:
            st.error(f"⚠️ Could not load city data: {e}")
            return
        if df_arima_all.empty:
            st.error("⚠️ The ARIMA model returned no data.")
            return

        df_w_daily, df_w_hourly = get_weather_forecast(lat, lon, city_name)
        weather_ok = not df_w_daily.empty
        ctx = build_forecast_pipeline(params, df_historical, df_arima_all, df_w_daily, df_w_hourly)

    if ctx is None:
        st.error("⚠️ The ARIMA forecast has no days after the latest actuals. Retrain the models from ⚙️ System Maintenance.")
        return
    if ctx['calib_msg']:
        st.toast(ctx['calib_msg'], icon="🧬")

    # What this run saw, so the Saved Anomalies menu can report real statuses.
    st.session_state['nova_runtime'] = {'city': city_name, 'horizon_max': ctx['df_org']['date'].max().strftime('%Y-%m-%d'),
                                        'anoms_on': bool(params.get('use_anomalies', True))}
    flash = st.session_state.pop('anom_flash', None)
    if flash:
        render_anomaly_flash(flash, ctx, params)

    df_clim = ctx['df_clim']
    w_target = params['week']
    df_all = unify_data(df_historical, df_clim)
    m_t = week_slice(df_all, w_target)
    mon, _ = week_bounds(w_target)
    m_1 = df_all[(df_all['date'] >= mon - timedelta(days=7)) & (df_all['date'] < mon)]

    # 1. KPI cards
    render_kpi_cards(m_t, m_1, w_target, city_name)

    # 2. Alert Center
    render_alert_center(build_alerts(params, ctx, m_t, df_w_daily, weather_ok))

    # 3. Forecast charts
    st.subheader("📈 3. Continuous Projection")
    flags = build_adjustment_flags(df_clim, df_w_daily, params['clima'] and weather_ok)
    render_timeline_charts(df_historical, ctx['df_org'], ctx['df_anom'], df_clim, w_target, flags)

    # 4. Weather detail
    if weather_ok:
        render_weather_detail(df_w_daily, df_w_hourly, city_name, w_target, df_clim if params['clima'] else None)

    # 5. Decoupled mode: the Allocation Engine only exists when there is a budget
    if params['budget'] > 0:
        render_allocation_engine(params, ctx, df_historical, m_t, w_target)
    else:
        st.info("🔌 Decoupled mode: showing the organic and adjusted forecast only. "
                "Enter a weekly budget in 💰 Budget & Allocation to unlock the Allocation Engine.")

    with st.expander("🕵️ Forensic Day Analysis"):
        forensic_day = st.date_input("Day to analyze", value=(today_mx() - timedelta(days=1)).date(),
                                     max_value=(today_mx() - timedelta(days=1)).date(), key="forensic_day")
        render_day_diagnostics(df_historical, forensic_day.strftime('%Y-%m-%d'), city_name, lat, lon)


# =====================================================================
# MODULE 11: MAIN APP
# =====================================================================
def main():
    current_date = pd.Timestamp.now(tz=TZ)
    current_week = int(current_date.isocalendar().week)
    params = render_command_center(current_week, current_date)
    render_dashboard(params)


if __name__ == "__main__":
    main()
