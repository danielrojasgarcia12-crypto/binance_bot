# market/scanner_15m.py
# ETAPA 3.1: Scanner 15M (Timing pro)
# - Universo: Top 20 Swing + Top 10 Acción (desde 1H)
# - Setups: Pullback+Rechazo, Breakout+Retest, Double top/bottom (básico)
# - Sugiere Entry/SL/TP con ATR (NO ejecuta)

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import requests
from binance.spot import Spot

from utils.indicators import ema, atr, last_n_high_low

TOP_SWING = 20
TOP_ACTION = 10

KLINES_LIMIT_15M = 220
EMA_FAST_15M = 20
EMA_SLOW_15M = 50
ATR_PERIOD = 14

SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0

# Breakout + retest
BREAKOUT_LOOKBACK = 40
BREAKOUT_MARGIN_PCT = 0.08      # un poco menos estricto
RETEST_WINDOW = 6               # cuántas velas después de breakout buscamos retest
RETEST_TOL_PCT = 0.20           # tolerancia para “tocó el nivel”

# Pullback + rechazo
PULLBACK_ZONE_TOL_PCT = 0.30    # tolerancia para zona EMA
REJECTION_BODY_MIN_PCT = 0.12   # tamaño mínimo del cuerpo (en % del precio) para considerar rechazo
REJECTION_WICK_RATIO = 1.3      # wick inferior > cuerpo*ratio (para long)

# Doble techo/suelo (básico)
DT_LOOKBACK = 60
DT_TOL_PCT = 0.25

MIN_ATR_PCT_15M = 0.20
MAX_ATR_PCT_15M = 5.00
MIN_PRICE = 0.00001

EXCLUDE_SYMBOLS = {"USDCUSDT","FDUSDUSDT","USD1USDT","BFUSDUSDT","TUSDUSDT","USDPUSDT","BUSDUSDT"}

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

def fmt_price(x: float) -> str:
    if x >= 100:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}"
    if x >= 0.01:
        return f"{x:.6f}"
    return f"{x:.8f}"

def fetch_klines(client: Spot, symbol: str, interval: str, limit: int):
    kl = safe_call(client.klines, symbol=symbol, interval=interval, limit=limit)
    # kline: [open_time, open, high, low, close, volume, ...]
    opens  = [float(k[1]) for k in kl]
    highs  = [float(k[2]) for k in kl]
    lows   = [float(k[3]) for k in kl]
    closes = [float(k[4]) for k in kl]
    return opens, highs, lows, closes

def atr_pct(highs, lows, closes):
    a = atr(highs, lows, closes, ATR_PERIOD)
    return (a / closes[-1]) * 100, a

def suggest_sl_tp(entry, atr_val, side: str):
    if side == "LONG":
        sl = entry - SL_ATR_MULT * atr_val
        tp = entry + TP_ATR_MULT * atr_val
    else:
        sl = entry + SL_ATR_MULT * atr_val
        tp = entry - TP_ATR_MULT * atr_val
    return sl, tp

def detect_pullback_rejection(opens, highs, lows, closes):
    """
    Tendencia 15M (EMA20>EMA50) + precio en zona EMA + vela de rechazo (LONG).
    """
    ema20 = ema(closes, EMA_FAST_15M)
    ema50 = ema(closes, EMA_SLOW_15M)

    # tendencia local alcista (más estable que exigir price>ema20 siempre)
    if not (ema20 > ema50):
        return None

    price = closes[-1]
    zone_low = min(ema20, ema50) * (1 - PULLBACK_ZONE_TOL_PCT / 100)
    zone_high = max(ema20, ema50) * (1 + PULLBACK_ZONE_TOL_PCT / 100)

    # si el cierre está dentro de la zona, evaluamos “rechazo”
    if not (zone_low <= price <= zone_high):
        return None

    # vela actual
    o = opens[-1]
    h = highs[-1]
    l = lows[-1]
    c = closes[-1]

    body = abs(c - o)
    wick_down = min(o, c) - l
    # tamaño mínimo del cuerpo relativo al precio
    body_pct = (body / max(c, 1e-9)) * 100

    # Rechazo alcista: wick inferior grande + cierre por encima de apertura
    if c > o and wick_down > body * REJECTION_WICK_RATIO and body_pct >= REJECTION_BODY_MIN_PCT:
        return {"type": "PULLBACK_REJECTION", "side": "LONG", "entry": c}

    return None

def detect_breakout_retest(highs, lows, closes):
    """
    Breakout al alza del rango previo + retest del nivel.
    """
    max_h, _ = last_n_high_low(highs, lows, BREAKOUT_LOOKBACK)
    price = closes[-1]

    margin = max_h * (BREAKOUT_MARGIN_PCT / 100)

    # detectar breakout: algún cierre reciente por encima del nivel + margen
    breakout_idx = None
    for i in range(len(closes) - RETEST_WINDOW, len(closes)):
        if closes[i] > (max_h + margin):
            breakout_idx = i
            break
    if breakout_idx is None:
        return None

    # retest: después del breakout, el mínimo toca cerca del nivel
    level = max_h
    tol = level * (RETEST_TOL_PCT / 100)
    start = breakout_idx
    end = len(closes)
    for j in range(start, end):
        if abs(lows[j] - level) <= tol:
            # confirmación: último cierre vuelve a estar por encima del nivel
            if closes[-1] > level:
                return {"type": "BREAKOUT_RETEST", "side": "LONG", "entry": closes[-1]}
    return None

def detect_double_top_bottom_basic(highs, lows, closes):
    max_h, min_l = last_n_high_low(highs, lows, DT_LOOKBACK)
    mid = (max_h + min_l) / 2
    tol = mid * (DT_TOL_PCT / 100)
    price = closes[-1]

    if abs(highs[-1] - max_h) <= tol and price < mid:
        return {"type": "DOUBLE_TOP_BASIC", "side": "SHORT", "entry": price}
    if abs(lows[-1] - min_l) <= tol and price > mid:
        return {"type": "DOUBLE_BOTTOM_BASIC", "side": "LONG", "entry": price}
    return None

def main():
    print("⏱️ ETAPA 3 — Scanner 15M (Timing pro)")
    print(f"Analizando candidatos: Top {TOP_SWING} Swing + Top {TOP_ACTION} Acción (desde 1H)\n")

    from market.scanner_1h import run_scanner_1h
    candidates = run_scanner_1h(TOP_SWING, TOP_ACTION)
    universe = candidates["swing"] + candidates["action"]
    symbols = [x["symbol"] for x in universe if x["symbol"] not in EXCLUDE_SYMBOLS]

    client = Spot()
    setups = []

    for sym in symbols:
        try:
            opens, highs, lows, closes = fetch_klines(client, sym, "15m", KLINES_LIMIT_15M)
            price = closes[-1]
            if price < MIN_PRICE:
                continue

            a_pct, a_val = atr_pct(highs, lows, closes)
            if not (MIN_ATR_PCT_15M <= a_pct <= MAX_ATR_PCT_15M):
                continue

            s1 = detect_pullback_rejection(opens, highs, lows, closes)
            s2 = detect_breakout_retest(highs, lows, closes)
            s3 = detect_double_top_bottom_basic(highs, lows, closes)

            for s in (s1, s2, s3):
                if not s:
                    continue
                sl, tp = suggest_sl_tp(s["entry"], a_val, s["side"])
                setups.append({
                    "symbol": sym,
                    "setup": s["type"],
                    "side": s["side"],
                    "entry": s["entry"],
                    "sl": sl,
                    "tp": tp,
                    "atr_pct": a_pct,
                })
        except Exception:
            continue

    if not setups:
        print("No se detectaron setups claros en estos candidatos (esto puede pasar en mercados sin pullbacks/retests).")
        return

    # Orden por “calma” (ATR% más bajo primero)
    setups.sort(key=lambda x: x["atr_pct"])

    print("=== SETUPS DETECTADOS (15M) ===")
    for r in setups[:30]:
        print(
            f"{r['symbol']:<12} {r['setup']:<18} {r['side']:<5} "
            f"Entry:{fmt_price(r['entry']):>10} "
            f"SL:{fmt_price(r['sl']):>10} "
            f"TP:{fmt_price(r['tp']):>10} "
            f"ATR%:{r['atr_pct']:<5.2f}"
        )

    print("\nNota: Son sugerencias de timing. NO ejecuta órdenes.")

if __name__ == "__main__":
    main()
