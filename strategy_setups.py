# strategy_setups.py
# ETAPA 7 — Setup base robusto: REBOTE EMA20 (15m)
# Devuelve una lista de setups con entry/sl/tp/atr_pct

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def ema(values, period: int):
    if values is None or len(values) < period:
        return None
    k = 2 / (period + 1)
    e = sum(values[:period]) / period
    for v in values[period:]:
        e = v * k + e * (1 - k)
    return e

def atr(ohlc, period: int = 14):
    if ohlc is None or len(ohlc) < period + 1:
        return None
    trs = []
    for i in range(1, len(ohlc)):
        h = ohlc[i]["high"]
        l = ohlc[i]["low"]
        pc = ohlc[i - 1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period

def atr_pct(ohlc, period: int = 14):
    a = atr(ohlc, period)
    if a is None:
        return None
    last_close = ohlc[-1]["close"]
    if last_close <= 0:
        return None
    return (a / last_close) * 100.0

def setup_ema20_bounce(ohlc):
    """
    LONG si:
    - Precio toca zona EMA20 (distancia pequeña)
    - Cierra arriba de EMA20
    - Vela verde
    SL/TP por ATR
    """
    if not ohlc or len(ohlc) < 30:
        return []

    closes = [c["close"] for c in ohlc]
    opens = [c["open"] for c in ohlc]

    e20 = ema(closes, 20)
    if e20 is None:
        return []

    price = closes[-1]
    prev_close = closes[-2]
    last_green = closes[-1] > opens[-1]

    # Distancia a EMA20 (zona de decisión)
    dist_pct = abs((price - e20) / e20) * 100.0

    # Condiciones: venía cerca o por debajo y recupera
    near = dist_pct <= 0.50
    cross_up = (prev_close <= e20 and price > e20) or (near and price > e20)

    if not (cross_up and last_green):
        return []

    a = atr(ohlc, 14)
    ap = atr_pct(ohlc, 14)
    if a is None or ap is None:
        return []

    # SL/TP ATR-based (profesional)
    entry = price
    sl = entry - 1.2 * a
    tp = entry + 2.0 * a

    return [{
        "setup": "EMA20_BOUNCE",
        "side": "LONG",
        "entry": float(entry),
        "sl": float(sl),
        "tp": float(tp),
        "atr": float(a),
        "atr_pct": float(ap),
        "meta": {
            "ema20": float(e20),
            "dist_pct": float(dist_pct)
        }
    }]
