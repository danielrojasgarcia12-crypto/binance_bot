import os
from binance.spot import Spot

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

client = Spot(
    api_key=API_KEY,
    api_secret=API_SECRET
)

print("Probando conexión a Binance REAL (Mainnet)...")


account = client.account()

print("Conexión OK. Balances disponibles:")
for b in account["balances"]:
    if float(b["free"]) > 0:
        print(b)
