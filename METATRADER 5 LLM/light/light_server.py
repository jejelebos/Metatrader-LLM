"""
MT5 LLM Trading Bot - Light Version
Optimized for low-end computers with minimal resource usage
Author: Jeremy
License: MIT
Version: 1.0

This is the light version of the trading bot, specifically optimized for:
- Minimal memory usage
- Single timeframe analysis
- Basic technical indicators
- Efficient resource management
"""

# Standard library imports
from flask import Flask, request, jsonify
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from threading import Thread
import requests
import json
import gc  # For manual garbage collection
import logging
from typing import Dict, List, Optional, Union
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('light_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# ============ CONFIGURATION SECTION ============
# LLM API Configuration
LLM_API_URL = "http://YOUR_LLM_API_URL:YOUR_LLM_PORT/v1/chat/completions"
LLM_MODEL = "YOUR_MODEL_NAME"
LLM_TIMEOUT = 30  # Reduced timeout for light version

# MT5 Configuration
MT5_LOGIN = 0000000  # Replace with your MT5 account number
MT5_PASSWORD = "YOUR_PASSWORD"  # Replace with your MT5 password
MT5_SERVER = "YOUR_BROKER_SERVER"  # Replace with your MT5 server name

# Trading Parameters
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]  # Limited symbol list
TIMEFRAME = mt5.TIMEFRAME_H1  # Single timeframe
ANALYSIS_INTERVAL = 300  # 5 minutes
HISTORY_BARS = 100  # Limited historical data
MAX_POSITIONS = 3  # Maximum positions
MAX_RETRIES = 3  # Maximum retries

# Risk Management
MAX_RISK_PERCENT = 2.0
MAX_SPREAD_POINTS = 20
MIN_POSITION_INTERVAL = 300

# Memory Management
MAX_CACHE_SIZE = 100 * 1024 * 1024  # 100MB
CLEANUP_INTERVAL = 300  # 5 minutes

# ============ GLOBAL VARIABLES ============
mt5_initialized = False
system_status = {
    "mt5_ready": False,
    "llm_ready": False,
    "last_error": None,
    "last_cleanup": datetime.now(),
    "memory_usage": 0
}

# ============ HELPER FUNCTIONS ============
def log_error(error: str, exception: Optional[Exception] = None) -> None:
    """Centralized error logging"""
    logger.error(f"{error} - {str(exception) if exception else ''}")
    system_status["last_error"] = error

def check_memory_usage() -> float:
    """Monitor memory usage"""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        system_status["memory_usage"] = memory_mb
        
        if memory_mb > MAX_CACHE_SIZE / 1024 / 1024:
            cleanup_memory()
            
        return memory_mb
    except Exception as e:
        log_error("Memory monitoring failed", e)
        return 0

def cleanup_memory() -> None:
    """Memory cleanup"""
    try:
        gc.collect()
        system_status["last_cleanup"] = datetime.now()
        logger.info("Memory cleanup performed")
    except Exception as e:
        log_error("Memory cleanup failed", e)

def cleanup() -> None:
    """Cleanup at shutdown"""
    try:
        if mt5_initialized:
            mt5.shutdown()
        logger.info("System shutdown completed")
    except Exception as e:
        log_error("Cleanup failed", e)

atexit.register(cleanup)

# ============ MT5 FUNCTIONS ============
def initialize_mt5() -> bool:
    """Initialize MT5 connection"""
    global mt5_initialized
    try:
        if mt5_initialized:
            return True
            
        if not mt5.initialize(
            login=MT5_LOGIN,
            server=MT5_SERVER,
            password=MT5_PASSWORD,
            timeout=60000
        ):
            log_error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
            
        if not mt5.terminal_info().connected:
            log_error("MT5 terminal not connected")
            return False
            
        logger.info("MT5 connected successfully")
        logger.info(f"Version: {mt5.version()}")
        
        mt5_initialized = True
        system_status["mt5_ready"] = True
        return True
        
    except Exception as e:
        log_error("MT5 initialization error", e)
        return False

def get_market_data() -> Dict:
    """Get essential market data"""
    try:
        if not check_mt5_connection():
            return {}

        market_data = {
            "account": get_account_info(),
            "positions": get_positions(),
            "symbols": {}
        }

        for symbol in SYMBOLS:
            if not mt5.symbol_select(symbol, True):
                continue

            rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, HISTORY_BARS)
            if rates is None:
                continue

            df = pd.DataFrame(rates)
            df = calculate_indicators(df)
            
            tick = mt5.symbol_info_tick(symbol)
            
            market_data["symbols"][symbol] = {
                "current": {
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "spread": tick.ask - tick.bid
                },
                "indicators": get_indicators_data(df)
            }

        return market_data

    except Exception as e:
        log_error("Market data retrieval failed", e)
        return {}

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate basic indicators"""
    try:
        # SMA
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR for position sizing
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        return df
    except Exception as e:
        log_error("Indicator calculation failed", e)
        return df

def get_account_info() -> Dict:
    """Get account information"""
    try:
        account = mt5.account_info()
        return {
            "balance": float(account.balance),
            "equity": float(account.equity),
            "margin": float(account.margin),
            "free_margin": float(account.margin_free),
            "margin_level": float(account.margin_level) if account.margin_level else None
        }
    except Exception as e:
        log_error("Account info retrieval failed", e)
        return {}

def get_positions() -> List[Dict]:
    """Get current positions"""
    try:
        positions = mt5.positions_get()
        if positions is None:
            return []
            
        return [{
            "ticket": pos.ticket,
            "symbol": pos.symbol,
            "type": "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
            "volume": pos.volume,
            "price_open": pos.price_open,
            "sl": pos.sl,
            "tp": pos.tp,
            "profit": pos.profit
        } for pos in positions]
    except Exception as e:
        log_error("Position retrieval failed", e)
        return []

def check_mt5_connection() -> bool:
    """Check MT5 connection status"""
    global mt5_initialized
    try:
        if not mt5_initialized or not mt5.terminal_info().connected:
            return initialize_mt5()
        return True
    except Exception as e:
        log_error("MT5 connection check failed", e)
        return False

def get_indicators_data(df: pd.DataFrame) -> Dict:
    """Extract indicator data"""
    try:
        return {
            "sma": {
                "20": float(df['sma20'].iloc[-1]),
                "50": float(df['sma50'].iloc[-1])
            },
            "rsi": float(df['rsi'].iloc[-1]),
            "atr": float(df['atr'].iloc[-1]) if 'atr' in df else None
        }
    except Exception as e:
        log_error("Indicator data extraction failed", e)
        return {}

# ============ LLM FUNCTIONS ============
def analyze_market() -> Dict:
    """Analyze market with LLM"""
    try:
        market_data = get_market_data()
        if not market_data:
            return {"error": "No market data available"}

        prompt = create_analysis_prompt(market_data)
        response = query_llm(prompt)
        
        if not response:
            return {"error": "LLM analysis failed"}
            
        return process_llm_response(response)
        
    except Exception as e:
        log_error("Market analysis failed", e)
        return {"error": str(e)}

def create_analysis_prompt(market_data: Dict) -> str:
    """Create LLM prompt"""
    return f"""Analyze the following market data and provide trading decisions.
Market conditions: {json.dumps(market_data, indent=2)}

Rules:
1. Maximum {MAX_POSITIONS} simultaneous positions
2. Risk maximum {MAX_RISK_PERCENT}% per trade
3. Only analyze {', '.join(SYMBOLS)} on {TIMEFRAME} timeframe
4. Avoid high spread conditions (>{MAX_SPREAD_POINTS} points)
5. Use basic indicators: SMA20, SMA50, RSI, ATR

Required JSON response format:
{{
    "analysis": {{
        "market_conditions": "description",
        "risk_level": "LOW/MEDIUM/HIGH"
    }},
    "positions": [
        {{
            "ticket": number,
            "action": "HOLD/CLOSE",
            "reason": "explanation"
        }}
    ],
    "new_trades": [
        {{
            "symbol": "symbol",
            "type": "BUY/SELL",
            "volume": number,
            "sl": number,
            "tp": number,
            "reason": "explanation"
        }}
    ]
}}"""

def query_llm(prompt: str) -> Optional[Dict]:
    """Query LLM with retry mechanism"""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                LLM_API_URL,
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                },
                timeout=LLM_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json()
                
            log_error(f"LLM request failed: {response.status_code}")
            
        except Exception as e:
            log_error(f"LLM query attempt {attempt + 1} failed", e)
            
        time.sleep(2 ** attempt)  # Exponential backoff
    
    return None

def process_llm_response(response: Dict) -> Dict:
    """Process and validate LLM response"""
    try:
        content = response["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        log_error("LLM response processing failed", e)
        return {}

# ============ TRADING FUNCTIONS ============
def execute_trades(analysis: Dict) -> None:
    """Execute trading decisions"""
    try:
        # Manage existing positions
        for position in analysis.get("positions", []):
            if position["action"] == "CLOSE":
                close_position(position["ticket"])

        # Execute new trades
        for trade in analysis.get("new_trades", []):
            if len(get_positions()) >= MAX_POSITIONS:
                logger.warning("Maximum positions reached")
                break
                
            execute_trade(trade)
            
    except Exception as e:
        log_error("Trade execution failed", e)

def execute_trade(trade: Dict) -> bool:
    """Execute single trade"""
    try:
        symbol = trade["symbol"]
        if not mt5.symbol_select(symbol, True):
            return False

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(trade["volume"]),
            "type": mt5.ORDER_TYPE_BUY if trade["type"] == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": mt5.symbol_info_tick(symbol).ask if trade["type"] == "BUY" else mt5.symbol_info_tick(symbol).bid,
            "sl": float(trade["sl"]),
            "tp": float(trade["tp"]),
            "deviation": 20,
            "magic": 78968786,
            "comment": "LLM trade",
            "type_time": mt5.ORDER_TIME_GTC
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log_error(f"Trade failed: {result.comment}")
            return False
            
        logger.info(f"Trade executed: {trade['symbol']} {trade['type']}")
        return True
        
    except Exception as e:
        log_error("Trade execution failed", e)
        return False

def close_position(ticket: int) -> bool:
    """Close specific position"""
    try:
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": position[0].symbol,
            "volume": position[0].volume,
            "type": mt5.ORDER_TYPE_SELL if position[0].type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(position[0].symbol).bid if position[0].type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(position[0].symbol).ask,
            "deviation": 20,
            "magic": 78968786,
            "comment": "LLM close",
            "type_time": mt5.ORDER_TIME_GTC
        }

        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE
        
    except Exception as e:
        log_error(f"Position close failed: {ticket}", e)
        return False

# ============ API ROUTES ============
@app.route('/status')
def get_status():
    """Get system status"""
    return jsonify({
        "mt5_connected": system_status["mt5_ready"],
        "llm_ready": system_status["llm_ready"],
        "error": system_status["last_error"],
        "memory_usage": system_status["memory_usage"],
        "last_cleanup": system_status["last_cleanup"].isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_route():
    """Analysis endpoint"""
    try:
        analysis = analyze_market()
        if "error" in analysis:
            return jsonify({"error": analysis["error"]}), 400
            
        execute_trades(analysis)
        return jsonify(analysis)
        
    except Exception as e:
        log_error("Analysis route failed", e)
        return jsonify({"error": str(e)}), 500

# ============ BACKGROUND TASKS ============
def periodic_analysis():
    """Periodic market analysis"""
    while True:
        try:
            check_memory_usage()
            
            with app.app_context():
                analysis = analyze_market()
                if "error" not in analysis:
                    execute_trades(analysis)
                    
            time.sleep(ANALYSIS_INTERVAL)
            
        except Exception as e:
            log_error("Periodic analysis failed", e)
            time.sleep(60)  # Wait before retry

# ============ MAIN ============
if __name__ == '__main__':
    try:
        logger.info("Starting Light Trading Bot...")
        
        if not initialize_mt5():
            raise Exception("MT5 initialization failed")
            
        analysis_thread = Thread(target=periodic_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        log_error("Critical startup error", e)
        cleanup()
        exit(1)
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
        cleanup()
        exit(0) 