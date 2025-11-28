from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from bot import TradingBot
import asyncio

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bot = TradingBot()

@app.get("/")
def read_root():
    return {"message": "Bitcoin Trading Bot API"}

@app.post("/start")
async def start_bot():
    await bot.start()
    return {"status": "Bot started"}

@app.post("/stop")
def stop_bot():
    bot.stop()
    return {"status": "Bot stopped"}

@app.get("/status")
def get_status():
    return bot.get_status()

@app.get("/history")
def get_history():
    return bot.get_history()

@app.get("/price")
def get_price():
    return {"price": bot.get_current_price()}

@app.get("/performance")
def get_performance():
    return bot.get_performance()
