import json
import asyncio
import websockets

SYMBOL = "btcusdt"
UMBRAL_PORCENTAJE = 0.1  # 0.1% (puedes cambiarlo luego)

WS_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL}@trade"

precio_inicial = None

async def main():
    global precio_inicial

    print(f"Conectando a Binance para {SYMBOL.upper()}...")
    
    async with websockets.connect(WS_URL) as ws:
        print("Conexión OK. Esperando precio inicial...")

        while True:
            mensaje = await ws.recv()
            data = json.loads(mensaje)

            precio_actual = float(data["p"])

            # Guardar el primer precio como referencia
            if precio_inicial is None:
                precio_inicial = precio_actual
                print(f"Precio inicial fijado en: {precio_inicial}")
                continue

            cambio_pct = ((precio_actual - precio_inicial) / precio_inicial) * 100

            if cambio_pct >= UMBRAL_PORCENTAJE:
                print(f"ALERTA SUBE ⬆️ {cambio_pct:.2f}% | Precio: {precio_actual}")
                precio_inicial = precio_actual  # reset referencia

            elif cambio_pct <= -UMBRAL_PORCENTAJE:
                print(f"ALERTA BAJA ⬇️ {cambio_pct:.2f}% | Precio: {precio_actual}")
                precio_inicial = precio_actual  # reset referencia

if __name__ == "__main__":
    asyncio.run(main())
