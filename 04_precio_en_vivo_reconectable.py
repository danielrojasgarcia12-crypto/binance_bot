import json
import asyncio
import websockets
import time

SYMBOL = "btcusdt"
WS_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL}@trade"

async def escuchar_precio():
    print(f"Intentando conexión a Binance para {SYMBOL.upper()}...")

    async with websockets.connect(
        WS_URL,
        ping_interval=20,
        ping_timeout=20
    ) as ws:

        print("Conectado correctamente. Escuchando precios...")

        while True:
            mensaje = await ws.recv()
            data = json.loads(mensaje)

            precio = data["p"]
            cantidad = data["q"]

            print(f"{SYMBOL.upper()} | Precio: {precio} | Cantidad: {cantidad}")

async def main():
    while True:
        try:
            await escuchar_precio()
        except websockets.exceptions.ConnectionClosed:
            print("⚠️ Conexión cerrada por Binance. Reintentando en 5 segundos...")
            time.sleep(5)
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            time.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPrograma detenido por el usuario.")
