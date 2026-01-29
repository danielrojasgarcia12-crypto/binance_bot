# market/scanner_1h.py
# ETAPA 2: Scanner 1H (Profesional) + función reutilizable para ETAPA 3

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import requests
from binance.spot import Spot
from utils.indicators import ema, atr, last_n_high_low, clamp, log10_safe

MIN_QUOTE_VOLUME_USDT = 10_000_000
KLINES_LIMIT = 260
EMA_FAST = 50
EMA_SLOW = 200
ATR_PERIOD = 14
RANGE_LOOKBACK = 60
RANGE_BAND_PCT = 0.25
MIN_ATR_PCT = 0.30
MIN_EMA_SEPARATION_PCT = 0.20

EXCLUDE_SYMBOLS = {"USDCUSDT", "FDUSDUSDT", "USD1USDT", "BFUSDUSDT", "TUSDUSDT", "USDPUSDT", "BUSDUSDT"}

ATR_IDEAL_LOW = 0.35
ATR_IDEAL_HIGH = 2.50
ATR_MAX_OK = 6.00

def safe_call(fn, *args, retries: int = 5, base_sleep: float = 2.0, **kwargs):
    last_err = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            last_err = e
            wait = base_sleep * (2 ** i)
            print(f"⚠️  Red inestable ({type(e).__name__}). Reintento {i+1}/{retries} en {wait:.0f}s...")
            time.sleep(wait)
        except Exception as e:
            last_err = e
            wait = base_sleep * (2 ** i)
            print(f"⚠️  Error temporal ({type(e).__name__}). Reintento {i+1}/{retries} en {wait:.0f}s...")
            time.sleep(wait)
    raise last_err

def get_operable_usdt_pairs(client: Spot) -> list[dict]:
    exchange_info = safe_call(client.exchange_info)
    symbols_info = exchange_info.get("symbols", [])

    usdt_pairs = []
    for s in symbols_info:
        if (
            s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
            and s.get("isSpotTradingAllowed", False)
        ):
            usdt_pairs.append(s["symbol"])

    tickers = safe_call(client.ticker_24hr)
    ticker_map = {t["symbol"]: t for t in tickers}

    operables = []
    for sym in usdt_pairs:
        if sym in EXCLUDE_SYMBOLS:
            continue
        t = ticker_map.get(sym)
        if not t:
            continue
        qv = float(t.get("quoteVolume", 0))
        if qv >= MIN_QUOTE_VOLUME_USDT:
            operables.append({"symbol": sym, "quoteVolume": qv})

    operables.sort(key=lambda x: x["quoteVolume"], reverse=True)
    return operables

def fetch_klines_1h(client: Spot, symbol: str, limit: int = KLINES_LIMIT):
    kl = safe_call(client.klines, symbol=symbol, interval="1h", limit=limit)
    highs = [float(k[2]) for k in kl]
    lows  = [float(k[3]) for k in kl]
    closes= [float(k[4]) for k in kl]
    return highs, lows, closes

def classify_symbol(symbol: str, quote_volume: float, closes, highs, lows) -> dict:
    price = closes[-1]
    ema50 = ema(closes, EMA_FAST)
    ema200 = ema(closes, EMA_SLOW)

    atr_val = atr(highs, lows, closes, ATR_PERIOD)
    atr_pct = (atr_val / price) * 100
    ema_sep_pct = abs(ema50 - ema200) / price * 100

    is_range = (atr_pct < MIN_ATR_PCT) or (ema_sep_pct < MIN_EMA_SEPARATION_PCT)

    max_high, min_low = last_n_high_low(highs, lows, RANGE_LOOKBACK)
    range_size = max_high - min_low if max_high > min_low else 1e-9
    pos_in_range = (price - min_low) / range_size

    if is_range:
        if pos_in_range <= RANGE_BAND_PCT:
            state = "🟢 RANGO (posible ACUMULACIÓN)"
        elif pos_in_range >= (1 - RANGE_BAND_PCT):
            state = "🔴 RANGO (posible DISTRIBUCIÓN)"
        else:
            state = "🟡 RANGO (NO operar)"
    else:
        if price > ema50 > ema200:
            state = "🔵 TENDENCIA ALCISTA"
        elif price < ema50 < ema200:
            state = "⚫ TENDENCIA BAJISTA"
        else:
            state = "🟠 TRANSICIÓN (cautela)"

    liq = log10_safe(quote_volume)

    if "TENDENCIA" in state:
        trend_bonus = 1.0
    elif "TRANSICIÓN" in state:
        trend_bonus = -0.3
    else:
        trend_bonus = -0.8

    if atr_pct < ATR_IDEAL_LOW:
        atr_score = -1.0
    elif atr_pct <= ATR_IDEAL_HIGH:
        atr_score = 1.0
    elif atr_pct <= ATR_MAX_OK:
        atr_score = 0.2
    else:
        atr_score = -1.5

    sep = clamp(ema_sep_pct, 0.0, 8.0)
    sep_score = sep / 8.0

    score = (liq * 1.5) + (trend_bonus * 2.0) + (atr_score * 2.0) + (sep_score * 1.0)

    return {
        "symbol": symbol,
        "state": state,
        "price": price,
        "quoteVolume": quote_volume,
        "atr_pct": atr_pct,
        "ema_sep_pct": ema_sep_pct,
        "score": score,
    }

def run_scanner_1h(top_swing: int = 20, top_action: int = 10) -> dict:
    client = Spot()
    operables = get_operable_usdt_pairs(client)

    results = []
    for item in operables:
        sym = item["symbol"]
        qv = item["quoteVolume"]
        try:
            highs, lows, closes = fetch_klines_1h(client, sym)
            results.append(classify_symbol(sym, qv, closes, highs, lows))
        except Exception:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)

    swing = [r for r in results if 0.35 <= r["atr_pct"] <= 2.50 and ("TENDENCIA" in r["state"] or "ACUMULACIÓN" in r["state"])]
    action = [r for r in results if 2.50 < r["atr_pct"] <= 6.00 and ("TENDENCIA" in r["state"] or "ACUMULACIÓN" in r["state"])]

    swing.sort(key=lambda x: x["score"], reverse=True)
    action.sort(key=lambda x: x["score"], reverse=True)

    return {
        "swing": swing[:top_swing],
        "action": action[:top_action],
        "all": results,
    }

def main():
    out = run_scanner_1h(20, 10)

    print("\n=== TOP 20 (SWING / estable) ===")
    for r in out["swing"]:
        print(f"{r['symbol']:<12} {r['state']:<28} ATR%:{r['atr_pct']:<5.2f} Vol:{r['quoteVolume']/1_000_000:>7.1f}M Score:{r['score']:.2f}")

    print("\n=== TOP 10 (ACCIÓN / más movimiento) ===")
    for r in out["action"]:
        print(f"{r['symbol']:<12} {r['state']:<28} ATR%:{r['atr_pct']:<5.2f} Vol:{r['quoteVolume']/1_000_000:>7.1f}M Score:{r['score']:.2f}")

if __name__ == "__main__":
    main()
