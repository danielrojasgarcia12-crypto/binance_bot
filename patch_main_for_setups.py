# patch_main_for_setups.py
# Parchea main.py para que scan_15m_for_symbol use el setup EMA20_BOUNCE

from pathlib import Path

ROOT = Path(__file__).resolve().parent
main_path = ROOT / "main.py"

if not main_path.exists():
    raise SystemExit("No encuentro main.py en la carpeta binance_bot.")

text = main_path.read_text(encoding="utf-8")

# Si ya está importado, no duplica
import_line = "from strategy_setups import setup_ema20_bounce\n"
if import_line not in text:
    # Inserta import al inicio (después de imports estándar)
    lines = text.splitlines(True)
    inserted = False
    out = []
    for line in lines:
        out.append(line)
        if not inserted and line.startswith("import"):
            continue
        if not inserted and (line.startswith("from") or line.strip() == ""):
            out.append(import_line)
            inserted = True
    text = "".join(out)

# Reemplaza la función scan_15m_for_symbol completa si encuentra su firma
marker = "def scan_15m_for_symbol("
start = text.find(marker)
if start == -1:
    raise SystemExit("No pude ubicar scan_15m_for_symbol en main.py. Dime si tiene otro nombre.")

# Encuentra el final de la función (naive: próximo 'def ' al inicio de línea)
rest = text[start:]
next_def = rest.find("\ndef ", 1)
if next_def == -1:
    end = len(text)
else:
    end = start + next_def

new_func = r'''
def scan_15m_for_symbol(client, symbol: str):
    """
    Scanner 15m: ahora usa setup base EMA20_BOUNCE.
    Devuelve lista de setups.
    """
    try:
        kl = client.klines(symbol, "15m", limit=60)
    except Exception:
        return []

    ohlc = []
    try:
        for k in kl:
            ohlc.append({
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
            })
    except Exception:
        return []

    # Setup base
    setups = []
    setups += setup_ema20_bounce(ohlc)

    return setups
'''.lstrip("\n")

patched = text[:start] + new_func + text[end:]
main_path.write_text(patched, encoding="utf-8")
print("✅ main.py parchado. scan_15m_for_symbol ahora usa EMA20_BOUNCE.")
