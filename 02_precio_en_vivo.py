import json
import asyncio
import websockets

# Símbolo a monitorear (puedes cambiarlo luego)
SYMBOL = "btcusdt"

# WebSocket de Binance REAL (mainnet)
WS_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL}@trade"

async def main():
    print(f"Conectando a Binance WebSocket para {SYMBOL.upper()}...")
    
    async with websockets.connect(WS_URL) as ws:
        print("Conexión OK. Mostrando precios en tiempo real (Ctrl + C para salir).")
        
        while True:
            mensaje = await ws.recv()
            data = json.loads(mensaje)

            precio = data["p"]      # precio del trade
            cantidad = data["q"]    # cantidad del trade
            tiempo = data["T"]      # timestamp en milisegundos

            print(f"{SYMBOL.upper()} | Precio: {precio} | Cantidad: {cantidad} | Tiempo: {tiempo}")

if __name__ == "__main__":
    asyncio.run(main())
