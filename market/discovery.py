# market/discovery.py
# ETAPA 1: Market Discovery
# Lee TODOS los pares USDT activos de Binance y filtra por volumen

from binance.spot import Spot

# Umbral de volumen en USDT (24h) para considerar un par "operable"
# Puedes ajustarlo luego; 10 millones es conservador
MIN_QUOTE_VOLUME_USDT = 10_000_000

def get_all_usdt_pairs(client: Spot) -> list[dict]:
    """
    Obtiene todos los pares USDT activos (trading habilitado).
    """
    exchange_info = client.exchange_info()
    symbols = exchange_info.get("symbols", [])

    usdt_pairs = []
    for s in symbols:
        if (
            s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
            and s.get("isSpotTradingAllowed", False)
        ):
            usdt_pairs.append(s)

    return usdt_pairs

def get_24h_tickers(client: Spot) -> list[dict]:
    """
    Obtiene estadísticas 24h (incluye volumen).
    """
    return client.ticker_24hr()

def filter_by_volume(usdt_pairs: list[dict], tickers_24h: list[dict]) -> list[dict]:
    """
    Filtra pares USDT por volumen 24h (quoteVolume).
    """
    # Crear un mapa symbol -> ticker
    ticker_map = {t["symbol"]: t for t in tickers_24h}

    operables = []
    for s in usdt_pairs:
        symbol = s["symbol"]
        t = ticker_map.get(symbol)
        if not t:
            continue

        # quoteVolume viene como string
        quote_volume = float(t.get("quoteVolume", 0))
        if quote_volume >= MIN_QUOTE_VOLUME_USDT:
            operables.append({
                "symbol": symbol,
                "quoteVolume": quote_volume
            })

    # Ordenar por volumen descendente
    operables.sort(key=lambda x: x["quoteVolume"], reverse=True)
    return operables

def main():
    print("🔎 ETAPA 1 — Market Discovery")
    print("Conectando a Binance (REST)...")

    client = Spot()  # Solo lectura, no requiere API key

    print("Leyendo pares USDT activos...")
    usdt_pairs = get_all_usdt_pairs(client)
    print(f"Total pares USDT activos: {len(usdt_pairs)}")

    print("Leyendo estadísticas 24h...")
    tickers_24h = get_24h_tickers(client)

    print(f"Filtrando por volumen >= {MIN_QUOTE_VOLUME_USDT:,.0f} USDT...")
    operables = filter_by_volume(usdt_pairs, tickers_24h)

    print(f"Pares USDT operables hoy: {len(operables)}\n")

    # Mostrar los primeros 15 como muestra
    print("Top 15 por volumen (24h):")
    for i, item in enumerate(operables[:15], start=1):
        print(f"{i:>2}. {item['symbol']:<10}  Volumen: {item['quoteVolume']:,.0f} USDT")

if __name__ == "__main__":
    main()
