"""
MT5 LLM Trading Bot - Medium Version
Optimized for mid-range computers with balanced performance and features
Author: Jeremy
License: MIT
Version: 1.0

This is the medium version of the trading bot, featuring:
- Multi-timeframe analysis (H1, H4)
- Extended technical indicators
- Advanced position management
- Balanced resource usage
- Enhanced pattern detection
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
import gc
import logging
from typing import Dict, List, Optional, Union, Tuple
import atexit
from scipy import stats
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medium_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# ============ CONFIGURATION SECTION ============
# LLM API Configuration
LLM_API_URL = "http://YOUR_LLM_API_URL:YOUR_LLM_PORT/v1/chat/completions"  # Replace with your LLM API URL
LLM_MODEL = "YOUR_MODEL_NAME"  # Replace with your model name
LLM_TIMEOUT = 60  # Extended timeout for more complex analysis

# MT5 Configuration
MT5_LOGIN = 0000000  # Replace with your MT5 account number
MT5_PASSWORD = "YOUR_PASSWORD"  # Replace with your MT5 password
MT5_SERVER = "YOUR_BROKER_SERVER"  # Replace with your MT5 server name

# Trading Parameters
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"]  # Extended symbol list
TIMEFRAMES = {  # Multiple timeframes
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4
}
ANALYSIS_INTERVAL = 300
HISTORY_BARS = 200  # Extended historical data
MAX_POSITIONS = 5  # Increased maximum positions
MAX_RETRIES = 3

# Risk Management
MAX_RISK_PERCENT = 2.0
MAX_SPREAD_POINTS = 20
MIN_POSITION_INTERVAL = 300
CORRELATION_THRESHOLD = 0.7

# Performance Management
MAX_CACHE_SIZE = 250 * 1024 * 1024  # 250MB
CLEANUP_INTERVAL = 600  # 10 minutes
BATCH_SIZE = 1000  # Data processing batch size

# HTTP Session Configuration
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=0.5,
    status_forcelist=[500, 502, 503, 504]
)
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

# ============ GLOBAL VARIABLES ============
mt5_initialized = False
system_status = {
    "mt5_ready": False,
    "llm_ready": False,
    "last_error": None,
    "last_cleanup": datetime.now(),
    "memory_usage": 0,
    "performance_metrics": {
        "analysis_time": 0,
        "success_rate": 0,
        "cpu_usage": 0
    }
}

# Cache for optimization
market_data_cache = {}
indicator_cache = {}

# ============ HELPER FUNCTIONS ============
def log_error(error: str, exception: Optional[Exception] = None) -> None:
    """Enhanced error logging with performance tracking"""
    logger.error(f"{error} - {str(exception) if exception else ''}")
    system_status["last_error"] = error
    
    if exception:
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")

def monitor_performance() -> Dict:
    """Monitor system performance"""
    try:
        import psutil
        process = psutil.Process()
        
        metrics = {
            "cpu_percent": process.cpu_percent(),
            "memory_percent": process.memory_percent(),
            "memory_usage": process.memory_info().rss / 1024 / 1024,
            "threads": process.num_threads()
        }
        
        system_status["performance_metrics"].update(metrics)
        return metrics
    except Exception as e:
        log_error("Performance monitoring failed", e)
        return {}

def cleanup_memory() -> None:
    """Enhanced memory cleanup with cache management"""
    try:
        # Clear caches
        market_data_cache.clear()
        indicator_cache.clear()
        
        # Force garbage collection
        gc.collect()
        
        system_status["last_cleanup"] = datetime.now()
        logger.info("Memory cleanup performed")
        
    except Exception as e:
        log_error("Memory cleanup failed", e)

def cleanup() -> None:
    """Enhanced cleanup at shutdown"""
    try:
        if mt5_initialized:
            # Close all positions if needed
            positions = mt5.positions_get()
            if positions:
                logger.warning(f"Closing {len(positions)} positions at shutdown")
                for pos in positions:
                    close_position(pos.ticket)
            
            mt5.shutdown()
            
        cleanup_memory()
        logger.info("System shutdown completed")
    except Exception as e:
        log_error("Cleanup failed", e)

atexit.register(cleanup)

# ============ TECHNICAL ANALYSIS FUNCTIONS ============
def calculate_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Calculate extended technical indicators
    
    Args:
        df (pd.DataFrame): Price data
        symbol (str): Symbol name for caching
    
    Returns:
        pd.DataFrame: DataFrame with calculated indicators
    """
    try:
        cache_key = f"{symbol}_{len(df)}"
        if cache_key in indicator_cache:
            return indicator_cache[cache_key]

        # Moving Averages
        for period in [9, 20, 50, 100]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

        # RSI with multiple periods
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD
        df['macd_line'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
        df['macd_hist'] = df['macd_line'] - df['macd_signal']

        # Bollinger Bands
        for period in [20, 50]:
            middle = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = middle + (std * 2)
            df[f'bb_lower_{period}'] = middle - (std * 2)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / middle

        # Stochastic
        for period in [14, 21]:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            df[f'stoch_k_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()

        # ATR and Volatility
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        df['atr_percent'] = (df['atr'] / df['close']) * 100

        # Store in cache
        indicator_cache[cache_key] = df
        
        return df
    except Exception as e:
        log_error(f"Indicator calculation failed for {symbol}", e)
        return df

def detect_patterns(df: pd.DataFrame) -> Dict:
    """
    Detect chart patterns
    
    Args:
        df (pd.DataFrame): Price data with indicators
    
    Returns:
        Dict: Detected patterns and their strengths
    """
    try:
        patterns = {}
        
        # Trend Detection
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        close = df['close'].iloc[-1]
        
        patterns['trend'] = {
            'direction': 'bullish' if sma_20 > sma_50 else 'bearish',
            'strength': abs(sma_20 - sma_50) / sma_50 * 100
        }
        
        # Support/Resistance Levels
        levels = find_support_resistance(df)
        patterns['levels'] = levels
        
        # Candlestick Patterns
        patterns['candlestick'] = detect_candlestick_patterns(df)
        
        # Momentum
        patterns['momentum'] = {
            'rsi': df['rsi_14'].iloc[-1],
            'macd': {
                'line': df['macd_line'].iloc[-1],
                'signal': df['macd_signal'].iloc[-1],
                'histogram': df['macd_hist'].iloc[-1]
            }
        }
        
        return patterns
    except Exception as e:
        log_error("Pattern detection failed", e)
        return {}

def find_support_resistance(df: pd.DataFrame) -> Dict:
    """Find support and resistance levels"""
    try:
        levels = {
            'support': [],
            'resistance': []
        }
        
        # Use price action to find levels
        highs = df['high'].rolling(window=20, center=True).max()
        lows = df['low'].rolling(window=20, center=True).min()
        
        # Find clusters of highs and lows
        from sklearn.cluster import KMeans
        
        high_clusters = KMeans(n_clusters=3).fit(highs.values.reshape(-1, 1))
        low_clusters = KMeans(n_clusters=3).fit(lows.values.reshape(-1, 1))
        
        levels['resistance'] = sorted(high_clusters.cluster_centers_.flatten())
        levels['support'] = sorted(low_clusters.cluster_centers_.flatten())
        
        return levels
    except Exception as e:
        log_error("Support/Resistance detection failed", e)
        return {'support': [], 'resistance': []}

def detect_candlestick_patterns(df: pd.DataFrame) -> Dict:
    """Detect candlestick patterns"""
    try:
        patterns = {}
        
        # Doji
        body = abs(df['close'] - df['open'])
        shadow = df['high'] - df['low']
        patterns['doji'] = bool(body.iloc[-1] <= shadow.iloc[-1] * 0.1)
        
        # Hammer
        body_low = np.minimum(df['open'], df['close'])
        lower_shadow = body_low - df['low']
        patterns['hammer'] = bool(
            lower_shadow.iloc[-1] >= 2 * body.iloc[-1] and
            body.iloc[-1] > 0
        )
        
        # Engulfing
        patterns['bullish_engulfing'] = bool(
            df['open'].iloc[-1] < df['close'].iloc[-2] and
            df['close'].iloc[-1] > df['open'].iloc[-2]
        )
        
        patterns['bearish_engulfing'] = bool(
            df['open'].iloc[-1] > df['close'].iloc[-2] and
            df['close'].iloc[-1] < df['open'].iloc[-2]
        )
        
        return patterns
    except Exception as e:
        log_error("Candlestick pattern detection failed", e)
        return {}

# ============ MARKET ANALYSIS FUNCTIONS ============
def analyze_market_conditions() -> Dict:
    """Analyze overall market conditions"""
    try:
        conditions = {
            "volatility": analyze_volatility(),
            "correlations": calculate_correlations(),
            "trends": analyze_trends(),
            "risk_level": "MEDIUM"  # Default
        }
        
        # Adjust risk level based on conditions
        if conditions["volatility"]["level"] == "HIGH":
            conditions["risk_level"] = "HIGH"
        elif all(trend["strength"] == "LOW" for trend in conditions["trends"].values()):
            conditions["risk_level"] = "LOW"
            
        return conditions
    except Exception as e:
        log_error("Market condition analysis failed", e)
        return {}

def analyze_volatility() -> Dict:
    """Analyze market volatility"""
    try:
        volatility_data = {}
        
        for symbol in SYMBOLS:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 24)
            if rates is None:
                continue
                
            df = pd.DataFrame(rates)
            atr = df['high'] - df['low']
            avg_atr = atr.mean()
            
            volatility_data[symbol] = {
                "atr": float(avg_atr),
                "atr_percent": float(avg_atr / df['close'].mean() * 100)
            }
        
        # Determine overall volatility level
        avg_vol = np.mean([v["atr_percent"] for v in volatility_data.values()])
        level = "HIGH" if avg_vol > 1.0 else "MEDIUM" if avg_vol > 0.5 else "LOW"
        
        return {
            "level": level,
            "average": float(avg_vol),
            "details": volatility_data
        }
    except Exception as e:
        log_error("Volatility analysis failed", e)
        return {"level": "UNKNOWN", "average": 0, "details": {}}

def calculate_correlations() -> Dict:
    """Calculate correlations between symbols"""
    try:
        prices = {}
        for symbol in SYMBOLS:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
            if rates is not None:
                prices[symbol] = pd.DataFrame(rates)['close']
        
        df = pd.DataFrame(prices)
        corr_matrix = df.corr()
        
        # Find highly correlated pairs
        correlations = {}
        for i in range(len(SYMBOLS)):
            for j in range(i+1, len(SYMBOLS)):
                sym1, sym2 = SYMBOLS[i], SYMBOLS[j]
                corr = corr_matrix.loc[sym1, sym2]
                if abs(corr) > CORRELATION_THRESHOLD:
                    correlations[f"{sym1}/{sym2}"] = float(corr)
        
        return correlations
    except Exception as e:
        log_error("Correlation calculation failed", e)
        return {}

def analyze_trends() -> Dict:
    """Analyze market trends across timeframes"""
    try:
        trends = {}
        for symbol in SYMBOLS:
            symbol_trends = {}
            for tf_name, tf in TIMEFRAMES.items():
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, 100)
                if rates is None:
                    continue
                    
                df = pd.DataFrame(rates)
                df = calculate_indicators(df, f"{symbol}_{tf_name}")
                
                # Trend strength calculation
                sma_20 = df['sma_20'].iloc[-1]
                sma_50 = df['sma_50'].iloc[-1]
                price = df['close'].iloc[-1]
                
                trend_direction = "UP" if sma_20 > sma_50 else "DOWN"
                trend_strength = abs(sma_20 - sma_50) / sma_50 * 100
                
                symbol_trends[tf_name] = {
                    "direction": trend_direction,
                    "strength": "HIGH" if trend_strength > 1.0 else "MEDIUM" if trend_strength > 0.5 else "LOW",
                    "value": float(trend_strength)
                }
            
            trends[symbol] = symbol_trends
            
        return trends
    except Exception as e:
        log_error("Trend analysis failed", e)
        return {}

# ============ POSITION MANAGEMENT ============
def manage_positions() -> None:
    """Advanced position management"""
    try:
        positions = mt5.positions_get()
        if not positions:
            return
            
        for position in positions:
            # Skip if position was recently modified
            if (datetime.now() - datetime.fromtimestamp(position.time_update)).seconds < MIN_POSITION_INTERVAL:
                continue
                
            symbol = position.symbol
            position_type = "BUY" if position.type == mt5.POSITION_TYPE_BUY else "SELL"
            
            # Get current market data
            market_data = get_symbol_data(symbol)
            if not market_data:
                continue
                
            # Check stop loss and take profit conditions
            should_modify = check_position_modification(position, market_data)
            if should_modify:
                modify_position(position, market_data)
                
            # Check exit conditions
            if check_exit_conditions(position, market_data):
                close_position(position.ticket)
                
    except Exception as e:
        log_error("Position management failed", e)

def check_position_modification(position: mt5.TradePosition, market_data: Dict) -> bool:
    """Check if position needs modification"""
    try:
        # Get current indicators
        current_price = market_data['current']['ask' if position.type == mt5.POSITION_TYPE_BUY else 'bid']
        atr = market_data['indicators']['atr']
        
        # Calculate new stop loss based on ATR
        if position.type == mt5.POSITION_TYPE_BUY:
            new_sl = current_price - (atr * 2)
            if new_sl > position.sl and new_sl < current_price:
                return True
        else:
            new_sl = current_price + (atr * 2)
            if new_sl < position.sl and new_sl > current_price:
                return True
                
        return False
        
    except Exception as e:
        log_error("Position modification check failed", e)
        return False

def modify_position(position: mt5.TradePosition, market_data: Dict) -> bool:
    """Modify position parameters"""
    try:
        current_price = market_data['current']['ask' if position.type == mt5.POSITION_TYPE_BUY else 'bid']
        atr = market_data['indicators']['atr']
        
        # Calculate new levels
        if position.type == mt5.POSITION_TYPE_BUY:
            new_sl = current_price - (atr * 2)
            new_tp = current_price + (atr * 3)
        else:
            new_sl = current_price + (atr * 2)
            new_tp = current_price - (atr * 3)
            
        request = {
            "action": mt5.TRADE_ACTION_MODIFY,
            "position": position.ticket,
            "symbol": position.symbol,
            "sl": new_sl,
            "tp": new_tp,
            "deviation": 20,
            "magic": 234000,
            "type_time": mt5.ORDER_TIME_GTC
        }
        
        result = mt5.order_send(request)
        return result and result.retcode == mt5.TRADE_RETCODE_DONE
        
    except Exception as e:
        log_error("Position modification failed", e)
        return False

def check_exit_conditions(position: mt5.TradePosition, market_data: Dict) -> bool:
    """Check if position should be closed"""
    try:
        indicators = market_data['indicators']
        
        # Exit on trend reversal
        if position.type == mt5.POSITION_TYPE_BUY:
            if indicators['sma']['20'] < indicators['sma']['50']:
                return True
        else:
            if indicators['sma']['20'] > indicators['sma']['50']:
                return True
                
        # Exit on RSI extremes
        rsi = indicators['rsi']
        if position.type == mt5.POSITION_TYPE_BUY and rsi > 70:
            return True
        if position.type == mt5.POSITION_TYPE_SELL and rsi < 30:
            return True
            
        return False
        
    except Exception as e:
        log_error("Exit condition check failed", e)
        return False

# ============ LLM ANALYSIS ============
def create_analysis_prompt(market_data: Dict) -> str:
    """Create enhanced LLM analysis prompt"""
    return f"""Analyze the following market data and provide trading decisions.
Market conditions: {json.dumps(market_data, indent=2)}

Rules and Context:
1. Maximum {MAX_POSITIONS} simultaneous positions
2. Risk maximum {MAX_RISK_PERCENT}% per trade
3. Analyze {', '.join(SYMBOLS)} on timeframes {', '.join(TIMEFRAMES.keys())}
4. Consider correlations above {CORRELATION_THRESHOLD}
5. Use all available indicators:
   - Moving Averages (9, 20, 50, 100)
   - RSI (7, 14, 21)
   - MACD
   - Bollinger Bands
   - Stochastic
   - ATR
6. Consider market patterns:
   - Support/Resistance levels
   - Candlestick patterns
   - Trend strength
   - Volatility levels

Required JSON response format:
{{
    "analysis": {{
        "market_conditions": "description",
        "volatility": "LOW/MEDIUM/HIGH",
        "risk_level": "LOW/MEDIUM/HIGH",
        "key_levels": {{
            "symbol": {{
                "support": [level1, level2],
                "resistance": [level1, level2]
            }}
        }}
    }},
    "positions": [
        {{
            "ticket": number,
            "action": "HOLD/MODIFY/CLOSE",
            "new_sl": number,
            "new_tp": number,
            "reason": "explanation"
        }}
    ],
    "new_trades": [
        {{
            "symbol": "symbol",
            "type": "BUY/SELL",
            "timeframe": "H1/H4",
            "entry": {{
                "type": "MARKET/LIMIT/STOP",
                "price": number
            }},
            "volume": number,
            "sl": number,
            "tp": number,
            "reason": "explanation"
        }}
    ]
}}"""

def process_llm_response(response: Dict) -> Dict:
    """Process and validate enhanced LLM response"""
    try:
        content = response["choices"][0]["message"]["content"]
        analysis = json.loads(content)
        
        # Validate analysis structure
        required_keys = ["analysis", "positions", "new_trades"]
        if not all(key in analysis for key in required_keys):
            raise ValueError("Missing required keys in analysis")
            
        # Validate position actions
        for position in analysis["positions"]:
            if position["action"] not in ["HOLD", "MODIFY", "CLOSE"]:
                raise ValueError(f"Invalid position action: {position['action']}")
                
        # Validate new trades
        for trade in analysis["new_trades"]:
            if trade["timeframe"] not in TIMEFRAMES:
                raise ValueError(f"Invalid timeframe: {trade['timeframe']}")
            if trade["entry"]["type"] not in ["MARKET", "LIMIT", "STOP"]:
                raise ValueError(f"Invalid entry type: {trade['entry']['type']}")
                
        return analysis
        
    except Exception as e:
        log_error("LLM response processing failed", e)
        return {}

# ============ TRADING EXECUTION ============
def execute_trades(analysis: Dict) -> None:
    """Execute trading decisions with advanced order types"""
    try:
        # Manage existing positions
        for position in analysis.get("positions", []):
            if position["action"] == "CLOSE":
                close_position(position["ticket"])
            elif position["action"] == "MODIFY":
                modify_position_from_analysis(position)

        # Execute new trades
        for trade in analysis.get("new_trades", []):
            if len(get_positions()) >= MAX_POSITIONS:
                logger.warning("Maximum positions reached")
                break
                
            execute_trade_with_type(trade)
            
    except Exception as e:
        log_error("Trade execution failed", e)

def execute_trade_with_type(trade: Dict) -> bool:
    """Execute trade with support for different order types"""
    try:
        symbol = trade["symbol"]
        if not mt5.symbol_select(symbol, True):
            return False

        # Calculate position size based on risk
        volume = calculate_position_size(trade)
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL if trade["entry"]["type"] == "MARKET" else mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": get_order_type(trade),
            "price": trade["entry"]["price"],
            "sl": trade["sl"],
            "tp": trade["tp"],
            "deviation": 20,
            "magic": 234000,
            "comment": f"LLM {trade['timeframe']}",
            "type_time": mt5.ORDER_TIME_GTC
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log_error(f"Trade failed: {result.comment}")
            return False
            
        logger.info(f"Trade executed: {trade['symbol']} {trade['type']} on {trade['timeframe']}")
        return True
        
    except Exception as e:
        log_error("Trade execution failed", e)
        return False

def calculate_position_size(trade: Dict) -> float:
    """Calculate position size based on risk management"""
    try:
        account = mt5.account_info()
        risk_amount = account.equity * (MAX_RISK_PERCENT / 100)
        
        symbol_info = mt5.symbol_info(trade["symbol"])
        pip_value = symbol_info.trade_tick_value
        
        # Calculate stop loss in pips
        sl_pips = abs(trade["entry"]["price"] - trade["sl"]) / symbol_info.point
        
        # Calculate position size
        volume = (risk_amount / (sl_pips * pip_value))
        
        # Round to valid lot size
        volume = round(volume / symbol_info.volume_step) * symbol_info.volume_step
        
        # Ensure within limits
        volume = min(max(volume, symbol_info.volume_min), symbol_info.volume_max)
        
        return volume
        
    except Exception as e:
        log_error("Position size calculation failed", e)
        return symbol_info.volume_min

def get_order_type(trade: Dict) -> int:
    """Get MT5 order type from trade specification"""
    if trade["entry"]["type"] == "MARKET":
        return mt5.ORDER_TYPE_BUY if trade["type"] == "BUY" else mt5.ORDER_TYPE_SELL
    elif trade["entry"]["type"] == "LIMIT":
        return mt5.ORDER_TYPE_BUY_LIMIT if trade["type"] == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT
    else:  # STOP
        return mt5.ORDER_TYPE_BUY_STOP if trade["type"] == "BUY" else mt5.ORDER_TYPE_SELL_STOP

# ============ API ROUTES ============
@app.route('/status')
def get_status():
    """Get enhanced system status"""
    return jsonify({
        "mt5_connected": system_status["mt5_ready"],
        "llm_ready": system_status["llm_ready"],
        "error": system_status["last_error"],
        "memory_usage": system_status["memory_usage"],
        "performance": system_status["performance_metrics"],
        "last_cleanup": system_status["last_cleanup"].isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_route():
    """Enhanced analysis endpoint"""
    try:
        start_time = time.time()
        
        analysis = analyze_market()
        if "error" in analysis:
            return jsonify({"error": analysis["error"]}), 400
            
        execute_trades(analysis)
        
        # Update performance metrics
        system_status["performance_metrics"]["analysis_time"] = time.time() - start_time
        
        return jsonify(analysis)
        
    except Exception as e:
        log_error("Analysis route failed", e)
        return jsonify({"error": str(e)}), 500

@app.route('/positions')
def get_positions_route():
    """Get current positions with analysis"""
    try:
        positions = get_positions()
        for position in positions:
            position["analysis"] = analyze_position(position["ticket"])
        return jsonify(positions)
    except Exception as e:
        log_error("Positions route failed", e)
        return jsonify({"error": str(e)}), 500

# ============ BACKGROUND TASKS ============
def periodic_analysis():
    """Enhanced periodic market analysis"""
    while True:
        try:
            # Monitor system performance
            monitor_performance()
            
            # Cleanup if needed
            if (datetime.now() - system_status["last_cleanup"]).seconds > CLEANUP_INTERVAL:
                cleanup_memory()
            
            with app.app_context():
                analysis = analyze_market()
                if "error" not in analysis:
                    execute_trades(analysis)
                    manage_positions()
                    
            time.sleep(ANALYSIS_INTERVAL)
            
        except Exception as e:
            log_error("Periodic analysis failed", e)
            time.sleep(60)

# ============ MAIN ============
if __name__ == '__main__':
    try:
        logger.info("Starting Medium Trading Bot...")
        
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