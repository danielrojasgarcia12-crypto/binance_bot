# config.py
# Configuración general del bot (scanner + estrategia)

# 1) Universo de monedas (15–20)
# Nota: todas en USDT para simplificar
from pathlib import Path

def load_symbols(path: str = "symbols.txt") -> list[str]:
    p = Path(path)
    if not p.exists():
        # Si no existe symbols.txt, caemos en una lista mínima segura
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

    symbols = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        symbols.append(line.upper())

    # Evitar duplicados manteniendo orden
    seen = set()
    out = []
    for s in symbols:
        if s not in seen:
            out.append(s)
            seen.add(s)

    return out

# Universo dinámico (editable sin tocar código)
SYMBOLS = load_symbols("symbols.txt")


# 2) Timeframes
TF_CONTEXT = "1h"   # contexto (tendencia general)
TF_ENTRY = "15m"    # entrada (timing)

# 3) Indicadores (valores estándar y estables)
EMA_FAST = 50
EMA_SLOW = 200
RSI_PERIOD = 14
ATR_PERIOD = 14

# 4) Filtros anti-ruido (muy importantes)
# Si el mercado está muy "muerto" (ATR bajo), no operamos
MIN_ATR_PCT = 0.30  # ATR mínimo como % del precio (0.30% es un inicio razonable)

# 5) Señales (modo asistido y luego automático)
RSI_BUY_MAX = 40    # RSI debe estar <= 40 para compras (confirmación, no gatillo)
RSI_SELL_MIN = 60   # RSI debe estar >= 60 para ventas (confirmación, no gatillo)

# 6) Riesgo (más adelante, cuando ejecutemos órdenes)
RISK_PER_TRADE_PCT = 1.0  # 1% del capital por operación (conservador)
SL_ATR_MULT = 1.5         # stop loss = 1.5 * ATR
TP_ATR_MULT = 3.0         # take profit = 3.0 * ATR (R:R aproximado 1:2)
