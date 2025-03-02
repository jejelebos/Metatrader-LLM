from flask import Flask, request, jsonify
import requests
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import atexit  # Ajouter cet import
import time
from threading import Thread
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json

app = Flask(__name__)

# Adresse API du LLM
LLM_API_URL = "http://192.168.0.52:1234/v1/chat/completions"

# Variables pour les notif discord et telegram
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1345088580715937923/fL9Ce8lKIYjKqHxMMqaDSInl087syh-rt-zsDdrH389GFcO0p5GZKBJiSZMhPpKXstaN"
TELEGRAM_BOT_TOKEN = "7348761806:AAHLNMpnRcAy8jA9kz4WouLHj4JAjFgWD3Q"
TELEGRAM_CHAT_ID = "7233793813"

# Parametres supplémentaires
SEND_DISCORD_NOTIFICATIONS = False
SEND_TELEGRAM_NOTIFICATIONS = False

# Variables globales
mt5_initialized = False
mt5_connection = None

# Variables globales pour l'état du système
system_status = {
    "mt5_ready": False,
    "llm_ready": False,
    "last_error": None
}

# Variables globales pour le suivi des trades
daily_trades = {
    "date": datetime.now().date(),
    "count": 0
}

# Augmenter les timeouts
TIMEOUT_CONNECT = 120  # 2 minutes pour la connexion
TIMEOUT_READ = 300    # 5 minutes pour la lecture

# Modifier la configuration de la session HTTP
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=0.5,  # Augmenté pour plus de patience
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["POST", "GET"]  # Explicitement autoriser POST
)
adapter = HTTPAdapter(
    max_retries=retries,
    pool_connections=10,
    pool_maxsize=10,
    pool_block=False
)
session.mount('http://', adapter)
session.timeout = (TIMEOUT_CONNECT, TIMEOUT_READ)

def cleanup():
    """Nettoyage à la fermeture"""
    global mt5_initialized, mt5_connection
    if mt5_initialized:
        print("Fermeture de la connexion MT5...")
        mt5_connection.shutdown()
        mt5_initialized = False
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message("Fermeture de la connexion MT5...")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message("Fermeture de la connexion MT5...")
    session.close()

atexit.register(cleanup)  # Enregistre la fonction de nettoyage

def initialize_mt5():
    """Initialise la connexion MT5 une seule fois au démarrage"""
    global mt5_initialized, mt5_connection
    
    try:
        if mt5_initialized:
            return True
            
        # Initialisation avec les paramètres
        if not mt5.initialize(
            login=40391310,
            server="Deriv-Demo",
            password="Jeje8Vince3+-",
            timeout=60000
        ):
            error = mt5.last_error()
            print(f"Échec de l'initialisation MT5: {error}")
            if SEND_DISCORD_NOTIFICATIONS:
                send_discord_message(f"Échec de l'initialisation MT5: {error}")
            if SEND_TELEGRAM_NOTIFICATIONS:
                send_telegram_message(f"Échec de l'initialisation MT5: {error}")
            return False
            
        # Vérifier la connexion
        if not mt5.terminal_info().connected:
            print("Terminal non connecté")
            if SEND_DISCORD_NOTIFICATIONS:
                send_discord_message("Terminal non connecté")
            if SEND_TELEGRAM_NOTIFICATIONS:
                send_telegram_message("Terminal non connecté")
            return False
            
        # Vérifier l'autorisation de trading
        if not mt5.terminal_info().trade_allowed:
            print("Trading non autorisé")
            return False
            
        print("MT5 connecté avec succès:")
        print(f"- Version: {mt5.version()}")
        account_info = mt5.account_info()
        print(f"- Compte: #{account_info.login}")
        print(f"- Serveur: {account_info.server}")
        print(f"- Balance: {account_info.balance}")
        print(f"- Trading autorisé: {mt5.terminal_info().trade_allowed}")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message("MT5 connecté avec succès:")
            send_telegram_message(f"- Version: {mt5.version()}")
            send_telegram_message(f"- Compte: #{account_info.login}")
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message(f"- Serveur: {account_info.server}")
            send_discord_message(f"- Balance: {account_info.balance}")
            send_discord_message(f"- Trading autorisé: {mt5.terminal_info().trade_allowed}")
        
        mt5_initialized = True
        mt5_connection = mt5
        return True
        
    except Exception as e:
        print(f"Erreur d'initialisation MT5: {str(e)}")
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message(f"Erreur d'initialisation MT5: {str(e)}")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message(f"Erreur d'initialisation MT5: {str(e)}")
        return False

def check_mt5_connection():
    """Vérifie et maintient la connexion MT5"""
    global mt5_initialized, system_status
    
    try:
        if not mt5_initialized or not mt5.terminal_info().connected:
            print("Connexion MT5 perdue, tentative de reconnexion...")
            if SEND_DISCORD_NOTIFICATIONS:
                send_discord_message("Connexion MT5 perdue, tentative de reconnexion...")
            if SEND_TELEGRAM_NOTIFICATIONS:
                send_telegram_message("Connexion MT5 perdue, tentative de reconnexion...")
            return initialize_mt5()
        return True
    except Exception as e:
        print(f"Erreur lors de la vérification de connexion MT5: {str(e)}")
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message(f"Erreur lors de la vérification de connexion MT5: {str(e)}")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message(f"Erreur lors de la vérification de connexion MT5: {str(e)}")
        return False

def get_economic_calendar():
    """Récupère le calendrier économique depuis MQL5"""
    today = datetime.now()
    week_later = today + timedelta(days=7)
    
    # Utilisation de l'API MQL5 Calendar
    calendar = mt5.calendar_get(today, week_later)
    if calendar:
        return [{
            "event_id": event.id,
            "time": event.time,
            "currency": event.currency,
            "importance": event.importance,
            "event": event.event,
            "actual": event.actual_value,
            "forecast": event.forecast_value,
            "previous": event.previous_value,
            "impact": event.importance
        } for event in calendar]
    return []

def calculate_correlations(symbols, timeframe=mt5.TIMEFRAME_H1, periods=100):
    """Calcule la matrice de corrélation entre les symboles"""
    price_data = {}
    
    for symbol in symbols:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, periods)
        if rates is not None:
            price_data[symbol] = pd.DataFrame(rates)['close']
    
    df = pd.DataFrame(price_data)
    correlation_matrix = df.corr()
    
    return correlation_matrix.to_dict()

def get_market_sentiment(symbol):
    """Récupère le sentiment du marché pour un symbole"""
    # Utilisation de la fonction symbols_total
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        return None
        
    # Récupération des positions totales
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return None
        
    long_positions = sum(1 for pos in positions if pos.type == mt5.POSITION_TYPE_BUY)
    short_positions = sum(1 for pos in positions if pos.type == mt5.POSITION_TYPE_SELL)
    
    total_positions = len(positions)
    if total_positions > 0:
        long_percentage = (long_positions / total_positions) * 100
        short_percentage = (short_positions / total_positions) * 100
    else:
        long_percentage = short_percentage = 0
        
    # Récupération du volume total
    ticks = mt5.copy_ticks_from(symbol, datetime.now() - timedelta(days=1), 100000, mt5.COPY_TICKS_ALL)
    
    return {
        "long_positions": long_positions,
        "short_positions": short_positions,
        "long_percentage": long_percentage,
        "short_percentage": short_percentage,
        "volume_24h": sum(tick.volume for tick in ticks) if ticks is not None else 0,
        "positions_total": total_positions
    }

def calculate_volatility(rates_frame, window=14):
    """Calcule différentes mesures de volatilité - version simplifiée"""
    if len(rates_frame) < window:
        return None
        
    df = pd.DataFrame(rates_frame)
    
    # Calcul de l'ATR basique
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=14).mean()
    
    # Classification plus permissive de la volatilité
    vol_level = "NORMAL"  # Par défaut, considérer la volatilité comme normale
    
    return {
        "atr": atr.iloc[-1],
        "volatility_level": vol_level
    }

def detect_chart_patterns(rates_frame):
    """Détecte les patterns chartistes basiques"""
    df = pd.DataFrame(rates_frame)
    patterns = {}
    
    # Calcul des bougies précédentes
    df['prev_open'] = df['open'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    
    # Détection des patterns basiques
    # Doji
    df['body'] = np.abs(df['close'] - df['open'])
    df['shadow'] = df['high'] - df['low']
    patterns['doji'] = bool(df['body'].iloc[-1] <= df['shadow'].iloc[-1] * 0.1)
    
    # Marteau
    body_low = np.minimum(df['open'], df['close'])
    lower_shadow = body_low - df['low']
    patterns['hammer'] = bool(
        lower_shadow.iloc[-1] >= 2 * df['body'].iloc[-1] and
        df['body'].iloc[-1] > 0
    )
    
    # Engulfing
    patterns['bullish_engulfing'] = bool(
        df['open'].iloc[-1] < df['prev_close'].iloc[-1] and
        df['close'].iloc[-1] > df['prev_open'].iloc[-1]
    )
    
    patterns['bearish_engulfing'] = bool(
        df['open'].iloc[-1] > df['prev_close'].iloc[-1] and
        df['close'].iloc[-1] < df['prev_open'].iloc[-1]
    )
    
    # Tendance
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()
    
    patterns['uptrend'] = bool(
        df['sma20'].iloc[-1] > df['sma50'].iloc[-1] and
        df['close'].iloc[-1] > df['sma20'].iloc[-1]
    )
    
    patterns['downtrend'] = bool(
        df['sma20'].iloc[-1] < df['sma50'].iloc[-1] and
        df['close'].iloc[-1] < df['sma20'].iloc[-1]
    )
    
    return patterns

def find_support_resistance(rates_frame, window=20):
    """Trouve les niveaux de support et résistance"""
    highs = rates_frame['high']
    lows = rates_frame['low']
    
    # Utilisation de l'algorithme de clustering pour identifier les niveaux
    all_prices = np.concatenate([highs, lows])
    
    # Identification des zones de prix fréquentes
    hist, bins = np.histogram(all_prices, bins=50)
    threshold = np.percentile(hist, 80)  # Top 20% des zones les plus fréquentes
    
    support_resistance_zones = []
    for i in range(len(hist)):
        if hist[i] > threshold:
            zone = {
                "price_level": (bins[i] + bins[i+1]) / 2,
                "strength": hist[i],
                "type": "support" if i < len(hist)/2 else "resistance"
            }
            support_resistance_zones.append(zone)
    
    return support_resistance_zones

def analyze_open_positions():
    """Analyse détaillée des positions ouvertes"""
    positions = mt5.positions_get()
    if positions is None:
        return []
        
    positions_analysis = []
    for pos in positions:
        symbol_info = mt5.symbol_info(pos.symbol)
        if symbol_info is None:
            continue
            
        current_price = mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask if pos.type == mt5.POSITION_TYPE_SELL else mt5.symbol_info_tick(pos.symbol).ask if pos.type == mt5.POSITION_TYPE_BUYLIMIT else mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.POSITION_TYPE_SELLLIMIT else mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.POSITION_TYPE_BUYSTOP else mt5.symbol_info_tick(pos.symbol).ask if pos.type == mt5.POSITION_TYPE_SELLSTOP else None
        
        # Calcul des points et de la valeur du pip
        pip_value = float(symbol_info.trade_tick_value * (symbol_info.point / symbol_info.trade_tick_size))
        points = float((current_price - pos.price_open) / symbol_info.point)
        if pos.type == mt5.POSITION_TYPE_SELL:
            points = -points
            
        position_analysis = {
            "ticket": int(pos.ticket),
            "symbol": str(pos.symbol),
            "type": "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
            "volume": float(pos.volume),
            "price_open": float(pos.price_open),
            "price_current": float(current_price),
            "profit": float(pos.profit),
            "profit_points": float(points),
            "pip_value": float(pip_value),
            "swap": float(pos.swap),
            "time_open": datetime.fromtimestamp(pos.time).isoformat(),
            "time_update": datetime.fromtimestamp(pos.time_update).isoformat(),
            "sl": float(pos.sl) if pos.sl else None,
            "tp": float(pos.tp) if pos.tp else None,
            "comment": str(pos.comment),
            "risk_reward_ratio": float(abs((current_price - pos.sl) / (pos.tp - current_price))) if pos.sl and pos.tp else None,
            "margin_required": float(pos.margin_required),
            "risk_percentage": float((abs(pos.price_open - pos.sl) * pos.volume * pip_value / mt5.account_info().equity * 100)) if pos.sl else None
        }
        
        # Ajout des analyses techniques pour la position
        rates = mt5.copy_rates_from_pos(pos.symbol, mt5.TIMEFRAME_H1, 0, 100)
        if rates is not None:
            rates_frame = pd.DataFrame(rates)
            position_analysis.update({
                "volatility": calculate_volatility(rates_frame),
                "patterns": detect_chart_patterns(rates_frame),
                "support_resistance": find_support_resistance(rates_frame),
                "market_sentiment": get_market_sentiment(pos.symbol)
            })
            
        positions_analysis.append(position_analysis)
        
    return positions_analysis

def get_market_data():
    """Récupère des données de marché enrichies depuis MT5"""
    if not mt5_initialized:
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message("MT5 non connecté")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message("MT5 non connecté")
        raise Exception("MT5 non connecté")
    
    # Données du compte
    account_info = mt5.account_info()
    market_data = {
        "account": {
            "balance": float(account_info.balance),
            "equity": float(account_info.equity),
            "margin": float(account_info.margin),
            "free_margin": float(account_info.margin_free),
            "margin_level": float(account_info.margin_level) if account_info.margin_level else None,
            "leverage": int(account_info.leverage),
        },
        "positions": [],
        "market": {}
    }

    # Récupérer les positions ouvertes
    positions = mt5.positions_get()
    if positions:
        market_data["positions"] = [
            {
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                "volume": pos.volume,
                "price_open": pos.price_open,
                "sl": pos.sl,
                "tp": pos.tp,
                "profit": pos.profit
            } for pos in positions
        ]

    # Liste des symboles à analyser
    symbols = ["Jump 75 Index","EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "XAGUSD", "XAUUSD"]  # Réduit à 3 symboles principaux
    
    # Récupérer les données pour chaque symbole
    for symbol in symbols:
        if not mt5.symbol_select(symbol, True):
            continue

        # Récupérer les infos du symbole pour les limites SL/TP
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            symbol_data = {"timeframes": {}, "limits": {}}
            symbol_data["limits"] = {
                "stops_level": symbol_info.trade_stops_level * symbol_info.point,
                "tick_size": symbol_info.trade_tick_size,
                "tick_value": symbol_info.trade_tick_value,
                "volume_min": symbol_info.volume_min,
                "volume_max": symbol_info.volume_max,
                "contract_size": symbol_info.trade_contract_size
            }

        # Ajouter les timeframes pour mieux calculer les SL/TP
        timeframes = {
            #"M1": mt5.TIMEFRAME_M1,
            #"M5": mt5.TIMEFRAME_M5,
            #"M15": mt5.TIMEFRAME_M15,
            #"M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            #"H4": mt5.TIMEFRAME_H4,
            #"D1": mt5.TIMEFRAME_D1
        }

        for tf_name, tf in timeframes.items():
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, 130)
            if rates is None:
                continue
                
            df = pd.DataFrame(rates)
            
            # Ajouter ATR pour le calcul des SL/TP
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = abs(df['high'] - df['close'].shift())
            df['low_close'] = abs(df['low'] - df['close'].shift())
            df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            df['atr'] = df['tr'].rolling(window=14).mean()

            # Indicateurs essentiels uniquement
            df['sma20'] = df['close'].rolling(window=20).mean()
            df['sma50'] = df['close'].rolling(window=50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            df['macd_line'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
            df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
            df['macd_hist'] = df['macd_line'] - df['macd_signal']
            
            # Stochastique
            for period in [14, 21]:
                low_min = df['low'].rolling(window=period).min()
                high_max = df['high'].rolling(window=period).max()
                df[f'stoch_k_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
                df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()
            
            # CCI (Commodity Channel Index)
            for period in [14, 20]:
                tp = (df['high'] + df['low'] + df['close']) / 3
                tp_sma = tp.rolling(window=period).mean()
                tp_std = tp.rolling(window=period).std()
                df[f'cci_{period}'] = (tp - tp_sma) / (0.015 * tp_std)
            
            current_data = {
                "price": {
                    "open": float(df['open'].iloc[-1]),
                    "high": float(df['high'].iloc[-1]),
                    "low": float(df['low'].iloc[-1]),
                    "close": float(df['close'].iloc[-1]),
                },
                "indicators": {
                    "sma": {
                        "20": float(df['sma20'].iloc[-1]),
                        "50": float(df['sma50'].iloc[-1])
                    },
                    "rsi": float(df['rsi'].iloc[-1]),
                    "atr": float(df['atr'].iloc[-1]),
                    "macd": {
                        "line": float(df['macd_line'].iloc[-1]) if 'macd_line' in df else None,
                        "signal": float(df['macd_signal'].iloc[-1]) if 'macd_signal' in df else None,
                        "histogram": float(df['macd_hist'].iloc[-1]) if 'macd_hist' in df else None
                    },
                    "stochastic": {
                        "k_14": float(df['stoch_k_14'].iloc[-1]) if 'stoch_k_14' in df else None,
                        "d_14": float(df['stoch_d_14'].iloc[-1]) if 'stoch_d_14' in df else None
                    },
                    "cci": {
                        "14": float(df['cci_14'].iloc[-1]) if 'cci_14' in df else None,
                        "20": float(df['cci_20'].iloc[-1]) if 'cci_20' in df else None
                    },
                    "momentum": {
                        "10": float(df['momentum_10'].iloc[-1]) if 'momentum_10' in df else None,
                        "21": float(df['momentum_21'].iloc[-1]) if 'momentum_21' in df else None
                    },
                    "roc": {
                        "10": float(df['roc_10'].iloc[-1]) if 'roc_10' in df else None,
                        "21": float(df['roc_21'].iloc[-1]) if 'roc_21' in df else None
                    },
                    "ichimoku": {
                        "tenkan_sen": float(df['tenkan_sen'].iloc[-1]) if 'tenkan_sen' in df else None,
                        "kijun_sen": float(df['kijun_sen'].iloc[-1]) if 'kijun_sen' in df else None,
                        "senkou_span_a": float(df['senkou_span_a'].iloc[-1]) if 'senkou_span_a' in df else None,
                        "senkou_span_b": float(df['senkou_span_b'].iloc[-1]) if 'senkou_span_b' in df else None
                    },
                    "pivot_points": {
                        "pivot": float(df['pivot'].iloc[-1]) if 'pivot' in df else None,
                        "r1": float(df['r1'].iloc[-1]) if 'r1' in df else None,
                        "s1": float(df['s1'].iloc[-1]) if 's1' in df else None,
                        "r2": float(df['r2'].iloc[-1]) if 'r2' in df else None,
                        "s2": float(df['s2'].iloc[-1]) if 's2' in df else None
                    }
                },
                "ranges": {
                    "day_high": float(df['high'].max()),
                    "day_low": float(df['low'].min()),
                    "atr_multiple": float(df['atr'].iloc[-1] * 2)
                }
            }
            
            symbol_data["timeframes"][tf_name] = current_data

        # Données actuelles du symbole
        tick = mt5.symbol_info_tick(symbol)
        symbol_data["current"] = {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": tick.ask - tick.bid
        }
        
        market_data["market"][symbol] = symbol_data

    return market_data

def calculate_ma(prices, period):
    """Calcule la moyenne mobile"""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

def calculate_rsi(prices, period):
    """Calcule le RSI"""
    if len(prices) < period + 1:
        return None
    
    deltas = np.diff(prices)
    gain = [delta if delta > 0 else 0 for delta in deltas]
    loss = [-delta if delta < 0 else 0 for delta in deltas]
    
    avg_gain = sum(gain[:period]) / period
    avg_loss = sum(loss[:period]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices, period, num_std=2):
    """Calcule les bandes de Bollinger"""
    if len(prices) < period:
        return None
    
    sma = calculate_ma(prices, period)
    std = np.std(prices[-period:])
    
    return {
        "upper": sma + (std * num_std),
        "middle": sma,
        "lower": sma - (std * num_std)
    }

def calculate_atr(rates, period):
    """Calcule l'ATR (Average True Range)"""
    if len(rates) < period + 1:
        return None
    
    true_ranges = []
    for i in range(1, len(rates)):
        high = rates[i]['high']
        low = rates[i]['low']
        prev_close = rates[i-1]['close']
        
        tr = max([
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        ])
        true_ranges.append(tr)
    
    return sum(true_ranges[-period:]) / period

def test_llm_connection():
    """Teste la connexion au LLM avec une requête complète"""
    try:
        test_payload = {
            "model": "mistral-7b-instruct-v0.2",
            "messages": [
                {
                    "role": "system",
                    "content": "Vous êtes un assistant de trading qui répond uniquement en format JSON valide."
                },
                {
                    "role": "user",
                    "content": "Confirmez que vous êtes prêt à analyser les données de trading en renvoyant exactement ce JSON : {\"status\": \"OK\", \"message\": \"Prêt à analyser les données de trading\"}"
                }
            ],
            "temperature": 0.1,
            "max_tokens": 100,
            "top_p": 1,
            "stream": False,  # Changé en False pour le test initial
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        
        print("Test de connexion au LLM...")
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message("Test de connexion au LLM...")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message("Test de connexion au LLM...")
        response = session.post(
            LLM_API_URL, 
            json=test_payload, 
            timeout=(TIMEOUT_CONNECT, TIMEOUT_READ)
        )
        
        if response.status_code != 200:
            print(f"Erreur de connexion au LLM: {response.status_code}")
            if SEND_DISCORD_NOTIFICATIONS:
                send_discord_message(f"Erreur de connexion au LLM: {response.status_code}")
            if SEND_TELEGRAM_NOTIFICATIONS:
                send_telegram_message(f"Erreur de connexion au LLM: {response.status_code}")
            return False
            
        response_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        print(f"Réponse brute du LLM: {response_text}")
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message(f"Réponse brute du LLM: {response_text}")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message(f"Réponse brute du LLM: {response_text}")
        

        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                response_json = eval(json_str)
                if response_json.get("status") == "OK":
                    print(f"LLM initialisé et prêt: {response_json.get('message')}")
                    if SEND_DISCORD_NOTIFICATIONS:
                        send_discord_message(f"LLM initialisé et prêt: {response_json.get('message')}")
                    if SEND_TELEGRAM_NOTIFICATIONS:
                        send_telegram_message(f"LLM initialisé et prêt: {response_json.get('message')}")
                    return True
            
            print("Format de réponse invalide")
            return False
            
        except Exception as e:
            print(f"Erreur de parsing de la réponse LLM: {str(e)}")
            if SEND_DISCORD_NOTIFICATIONS:
                send_discord_message(f"Erreur de parsing de la réponse LLM: {str(e)}")
            if SEND_TELEGRAM_NOTIFICATIONS:
                send_telegram_message(f"Erreur de parsing de la réponse LLM: {str(e)}")
            print(f"Réponse complète: {response_text}")
            if SEND_DISCORD_NOTIFICATIONS:
                send_discord_message(f"Réponse complète: {response_text}")
            if SEND_TELEGRAM_NOTIFICATIONS:
                send_telegram_message(f"Réponse complète: {response_text}")
            return False
        
    except Exception as e:
        print(f"Erreur lors du test LLM: {str(e)}")
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message(f"Erreur lors du test LLM: {str(e)}")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message(f"Erreur lors du test LLM: {str(e)}")
        return False

def initialize_background():
    """Initialise les connexions MT5 et LLM en arrière-plan"""
    global system_status
    
    max_attempts = 5
    attempt = 1
    
    while attempt <= max_attempts:
        print(f"\nTentative {attempt} de connexion à MT5...")
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message(f"Tentative {attempt} de connexion à MT5...")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message(f"Tentative {attempt} de connexion à MT5...")
        if initialize_mt5():
            print("\nTest de récupération des données MT5...")
            if SEND_DISCORD_NOTIFICATIONS:
                send_discord_message("\nTest de récupération des données MT5...")
            if SEND_TELEGRAM_NOTIFICATIONS:
                send_telegram_message("\nTest de récupération des données MT5...")
            try:
                # Test simple de récupération de données
                symbol = ["Jump 75 Index","EURUSD", "GBPUSD", "USDJPY", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD", "USDCAD", "AUDUSD", "NZDUSD"]
                mt5.symbol_select(symbol, True)
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
                if rates is not None:
                    print("Récupération des données MT5 OK")
                    if SEND_DISCORD_NOTIFICATIONS:
                        send_discord_message("Récupération des données MT5 OK")
                    if SEND_TELEGRAM_NOTIFICATIONS:
                        send_telegram_message("Récupération des données MT5 OK")
                    break
                else:
                    print("Échec de récupération des données")
                    if SEND_DISCORD_NOTIFICATIONS:
                        send_discord_message("Échec de récupération des données")
                    if SEND_TELEGRAM_NOTIFICATIONS:
                        send_telegram_message("Échec de récupération des données")
            except Exception as e:
                print(f"Erreur test MT5: {str(e)}")
                if SEND_DISCORD_NOTIFICATIONS:
                    send_discord_message(f"Erreur test MT5: {str(e)}")
                if SEND_TELEGRAM_NOTIFICATIONS:
                    send_telegram_message(f"Erreur test MT5: {str(e)}")
        attempt += 1
        if attempt <= max_attempts:
            print("Nouvelle tentative dans 5 secondes...")
            if SEND_DISCORD_NOTIFICATIONS:
                send_discord_message("Nouvelle tentative dans 5 secondes...")
            if SEND_TELEGRAM_NOTIFICATIONS:
                send_telegram_message("Nouvelle tentative dans 5 secondes...")
            time.sleep(5)
    
    # Initialiser la connexion au LLM
    attempt = 1
    while attempt <= max_attempts:
        print(f"\nTentative {attempt} de connexion au LLM...")
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message(f"Tentative {attempt} de connexion au LLM...")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message(f"Tentative {attempt} de connexion au LLM...")
        try:
            print("Test de connexion au LLM...")
            if SEND_DISCORD_NOTIFICATIONS:
                send_discord_message("Test de connexion au LLM...")
            if SEND_TELEGRAM_NOTIFICATIONS:
                send_telegram_message("Test de connexion au LLM...")
            response = session.post(
                LLM_API_URL,
                json={
                    "model": "mathstral-7b-v0.1",
                    "messages": [{"role": "user", "content": "Test de connexion"}],
                    "temperature": 0.7
                },
                timeout=(TIMEOUT_CONNECT, TIMEOUT_READ)
            )
            
            if response.status_code == 200:
                print(f"Réponse brute du LLM: {response.text[:200]}...")
                if SEND_DISCORD_NOTIFICATIONS:
                    send_discord_message(f"Réponse brute du LLM: {response.text[:200]}...") 
                if SEND_TELEGRAM_NOTIFICATIONS:
                    send_telegram_message(f"Réponse brute du LLM: {response.text[:200]}...")
                system_status["llm_ready"] = True
                print(f"LLM initialisé et prêt: {response.json().get('message', 'OK')}")
                if SEND_DISCORD_NOTIFICATIONS:
                    send_discord_message(f"LLM initialisé et prêt: {response.json().get('message', 'OK')}")
                if SEND_TELEGRAM_NOTIFICATIONS:
                    send_telegram_message(f"LLM initialisé et prêt: {response.json().get('message', 'OK')}")
                break
            else:
                print(f"Erreur connexion LLM: {response.status_code}")
                if SEND_DISCORD_NOTIFICATIONS:
                    send_discord_message(f"Erreur connexion LLM: {response.status_code}")
                if SEND_TELEGRAM_NOTIFICATIONS:
                    send_telegram_message(f"Erreur connexion LLM: {response.status_code}")
        except Exception as e:
            print(f"Erreur test LLM: {str(e)}")
            if SEND_DISCORD_NOTIFICATIONS:
                send_discord_message(f"Erreur test LLM: {str(e)}")
            if SEND_TELEGRAM_NOTIFICATIONS:
                send_telegram_message(f"Erreur test LLM: {str(e)}")
        attempt += 1
        if attempt <= max_attempts:
            print("Nouvelle tentative dans 5 secondes...")
            if SEND_DISCORD_NOTIFICATIONS:
                send_discord_message("Nouvelle tentative dans 5 secondes...")
            if SEND_TELEGRAM_NOTIFICATIONS:
                send_telegram_message("Nouvelle tentative dans 5 secondes...")
            time.sleep(5)
    
    if system_status["mt5_ready"] and system_status["llm_ready"]:
        print("\nInitialisation complète du système réussie!")
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message("Initialisation complète du système réussie!")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message("Initialisation complète du système réussie!")
    else:
        print("\nÉchec de l'initialisation complète du système")
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message("Échec de l'initialisation complète du système")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message("Échec de l'initialisation complète du système")

@app.route('/status')
def get_status():
    """Endpoint pour vérifier l'état du système"""
    return jsonify({
        "mt5_connected": system_status["mt5_ready"],
        "llm_connected": system_status["llm_ready"],
        "error": system_status["last_error"],
        "ready": system_status["mt5_ready"] and system_status["llm_ready"]
    })

def reset_daily_trades():
    """Réinitialise le compteur de trades quotidien"""
    global daily_trades
    current_date = datetime.now().date()
    if daily_trades["date"] != current_date:
        daily_trades["date"] = current_date
        daily_trades["count"] = 0

def calculate_indicators(rates_frame):
    """Calcule des indicateurs techniques avancés"""
    df = rates_frame.copy()
    
    # Moyennes mobiles et EMA
    for period in [9, 13, 20, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
    
    # Bandes de Bollinger
    for period in [20, 50]:
        middle = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df[f'bb_upper_{period}'] = middle + (std * 2)
        df[f'bb_lower_{period}'] = middle - (std * 2)
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / middle
    
    # RSI
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
    
    # Stochastique
    for period in [14, 21]:
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        df[f'stoch_k_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()
    
    # CCI (Commodity Channel Index)
    for period in [14, 20]:
        tp = (df['high'] + df['low'] + df['close']) / 3
        tp_sma = tp.rolling(window=period).mean()
        tp_std = tp.rolling(window=period).std()
        df[f'cci_{period}'] = (tp - tp_sma) / (0.015 * tp_std)
    
    # ATR et Volatilité
    tr = pd.DataFrame([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ]).max()
    df['atr_14'] = tr.rolling(window=14).mean()
    df['atr_percent'] = (df['atr_14'] / df['close']) * 100
    
    # Momentum et ROC (Rate of Change)
    for period in [10, 21]:
        df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        df[f'roc_{period}'] = (df['close'] / df['close'].shift(period) - 1) * 100
    
    # Ichimoku Cloud
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2
    
    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    df['kijun_sen'] = (high_26 + low_26) / 2
    
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    df['senkou_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
    
    # Pivots Points
    df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
    df['r1'] = 2 * df['pivot'] - df['low'].shift(1)
    df['s1'] = 2 * df['pivot'] - df['high'].shift(1)
    df['r2'] = df['pivot'] + (df['high'].shift(1) - df['low'].shift(1))
    df['s2'] = df['pivot'] - (df['high'].shift(1) - df['low'].shift(1))
    
    return df

def calculate_advanced_patterns(rates_frame):
    """Calcule les patterns avancés et les analyses techniques complexes"""
    df = rates_frame.copy()
    
    # Niveaux de Fibonacci
    high = df['high'].max()
    low = df['low'].min()
    diff = high - low
    
    fib_levels = {
        "0": low,
        "0.236": low + 0.236 * diff,
        "0.382": low + 0.382 * diff,
        "0.5": low + 0.5 * diff,
        "0.618": low + 0.618 * diff,
        "0.786": low + 0.786 * diff,
        "1": high
    }
    
    # Analyse des vagues d'Elliott
    # Simplifié : détection des 5 vagues principales
    elliott_waves = {
        "wave_1": None,
        "wave_2": None,
        "wave_3": None,
        "wave_4": None,
        "wave_5": None,
        "pattern_detected": False
    }
    
    # Détection basique des vagues
    if len(df) > 20:
        pivots = df['close'].rolling(window=5).apply(lambda x: 1 if x.iloc[-1] == max(x) else (-1 if x.iloc[-1] == min(x) else 0)).fillna(0)
        wave_points = df[pivots != 0]
        if len(wave_points) >= 5:
            elliott_waves["pattern_detected"] = True
            elliott_waves["wave_points"] = wave_points[-5:].to_dict()
    
    # Patterns Harmoniques
    harmonic_patterns = {
        "gartley": {
            "detected": False,
            "completion_point": None,
            "risk_level": None
        },
        "butterfly": {
            "detected": False,
            "completion_point": None,
            "risk_level": None
        },
        "bat": {
            "detected": False,
            "completion_point": None,
            "risk_level": None
        },
        "crab": {
            "detected": False,
            "completion_point": None,
            "risk_level": None
        }
    }
    
    # Analyse de performance historique pour ce pattern
    performance_stats = {
        "success_rate": 0.0,
        "avg_profit": 0.0,
        "avg_loss": 0.0,
        "best_timeframe": None,
        "optimal_parameters": {}
    }
    
    return {
        "fibonacci_levels": fib_levels,
        "elliott_waves": elliott_waves,
        "harmonic_patterns": harmonic_patterns,
        "performance_stats": performance_stats
    }

@app.route('/analyze', methods=['POST'])
def analyze_route():
    """Route Flask pour l'analyse"""
    return analyze()

def get_available_symbols():
    """Récupère la liste des symboles disponibles"""
    symbols = []
    try:
        # Récupérer tous les symboles
        all_symbols = mt5.symbols_get()
        if all_symbols is None:
            return ["Jump 75 Index","EURUSD", "GBPUSD", "USDJPY"]  # Symboles par défaut si erreur
            
        # Filtrer les symboles Forex majeurs
        forex_symbols = [sym.name for sym in all_symbols if 
            sym.name in ["Jump 75 Index","EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]]
            
        return forex_symbols if forex_symbols else ["Jump 75 Index","EURUSD"]
        
    except Exception as e:
        print(f"Erreur récupération symboles: {str(e)}")
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message(f"Erreur récupération symboles: {str(e)}")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message(f"Erreur récupération symboles: {str(e)}")
        return ["Jump 75 Index","EURUSD"]  # Symbole par défaut en cas d'erreur

def send_telegram_message(message):
    """Envoie un message à Telegram"""
    try:
        # Vérifier si le bot est configuré
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            print("Erreur: Bot Telegram non configuré")
            return
        
        # Construire le payload
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        # Envoyer la requête à l'API Telegram
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data=payload
        )
        
        if response.status_code != 200:
            print(f"Erreur envoi message Telegram: {response.text}")
    except Exception as e:
        print(f"Erreur envoi message Telegram: {str(e)}")
        
def send_discord_message(message):
    """Envoie un message à Discord"""
    try:
        # Vérifier si le bot est configuré
        if not DISCORD_WEBHOOK_URL:
            print("Erreur: Webhook Discord non configuré")
            return
        
        # Construire le payload
        payload = {
            "content": message
        }
        
        # Envoyer la requête au webhook Discord
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        
        if response.status_code != 204:
            print(f"Erreur envoi message Discord: {response.text}")
    except Exception as e:
        print(f"Erreur envoi message Discord: {str(e)}")

def analyze():
    try:
        print("\n=== Début de l'analyse ===")
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message("Début de l'analyse")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message("Début de l'analyse")
        

        # Vérifier la connexion MT5
        if not mt5_initialized:
            print("Erreur: MT5 non connecté")
            if SEND_DISCORD_NOTIFICATIONS:
                send_discord_message("Erreur: MT5 non connecté")
            if SEND_TELEGRAM_NOTIFICATIONS:
                send_telegram_message("Erreur: MT5 non connecté")
            return jsonify({"error": "MT5 non connecté"})

        # Récupérer les données de marché
        market_data = get_market_data()
        print("\nDonnées de marché récupérées:")
        print(f"- Balance: {market_data['account']['balance']}")
        print(f"- Equity: {market_data['account']['equity']}")

        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message("Données de marché récupérées:")
            send_discord_message(f"- Balance: {market_data['account']['balance']}")
            send_discord_message(f"- Equity: {market_data['account']['equity']}")
        
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message("Données de marché récupérées:")
            send_telegram_message(f"- Balance: {market_data['account']['balance']}")
            send_telegram_message(f"- Equity: {market_data['account']['equity']}")
        
        # Récupérer les positions ouvertes
        positions = mt5.positions_get()
        if positions:
            print("\nPositions ouvertes:")
            if SEND_DISCORD_NOTIFICATIONS:
                send_discord_message("Positions ouvertes:")
            if SEND_TELEGRAM_NOTIFICATIONS:
                send_telegram_message("Positions ouvertes:")
            for pos in positions:
                print(f"- Ticket {pos.ticket}: {pos.symbol}, Type: {'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL' if pos.type == mt5.POSITION_TYPE_SELL else 'BUYLIMIT' if pos.type == mt5.POSITION_TYPE_BUYLIMIT else 'SELLLIMIT' if pos.type == mt5.POSITION_TYPE_SELLLIMIT else 'BUYSTOP' if pos.type == mt5.POSITION_TYPE_BUYSTOP else 'SELLSTOP' if pos.type == mt5.POSITION_TYPE_SELLSTOP else None}, Profit: {pos.profit}")
                if SEND_DISCORD_NOTIFICATIONS:
                    send_discord_message(f"- Ticket {pos.ticket}: {pos.symbol}, Type: {'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL' if pos.type == mt5.POSITION_TYPE_SELL else 'BUYLIMIT' if pos.type == mt5.POSITION_TYPE_BUYLIMIT else 'SELLLIMIT' if pos.type == mt5.POSITION_TYPE_SELLLIMIT else 'BUYSTOP' if pos.type == mt5.POSITION_TYPE_BUYSTOP else 'SELLSTOP' if pos.type == mt5.POSITION_TYPE_SELLSTOP else None}, Profit: {pos.profit}")
                if SEND_TELEGRAM_NOTIFICATIONS:
                    send_telegram_message(f"- Ticket {pos.ticket}: {pos.symbol}, Type: {'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL' if pos.type == mt5.POSITION_TYPE_SELL else 'BUYLIMIT' if pos.type == mt5.POSITION_TYPE_BUYLIMIT else 'SELLLIMIT' if pos.type == mt5.POSITION_TYPE_SELLLIMIT else 'BUYSTOP' if pos.type == mt5.POSITION_TYPE_BUYSTOP else 'SELLSTOP' if pos.type == mt5.POSITION_TYPE_SELLSTOP else None}, Profit: {pos.profit}")
        else:
            print("\nAucune position ouverte")
            if SEND_DISCORD_NOTIFICATIONS:
                send_discord_message("Aucune position ouverte")
            if SEND_TELEGRAM_NOTIFICATIONS:
                send_telegram_message("Aucune position ouverte")

        # Récupérer les infos du symbole pour le prompt
        symbol_info = mt5.symbol_info("Jump 75 Index")  # Symbole de référence
        stops_level = symbol_info.trade_stops_level if symbol_info else 0

        # Préparer le prompt pour le LLM avec le format JSON complet
        prompt = f"""Analyse technique des marchés Forex.

Données de marché actuelles: {market_data}

INSTRUCTIONS:
1. Vérifications préalables:
   - Vérifier que le marché est ouvert (pas de weekend/férié)
   - Vérifier que l'équité est suffisante (min 100$ pour des lots de 0.1 ou si moins a calculer en fonction du pourcentage de risque)
   - Vérifier que la marge libre est suffisante
   - Vérifier que le spread n'est pas trop élevé (max 20 points sauf opportunité exceptionnelle)
   - Vérifier que la volatilité n'est pas excessive (sauf opportunité exceptionnelle)

2. Analyser les données de marché en profondeur:
   - Analyser les moyennes mobiles (SMA 20, 50) pour la tendance
   - Vérifier le RSI pour les conditions de surachat/survente
   - Utiliser le MACD pour confirmer la tendance et les divergences
   - Analyser les stochastiques pour les signaux de retournement
   - Vérifier le CCI pour les conditions extrêmes
   - Utiliser l'Ichimoku pour une vue complète de la tendance
   - Analyser les pivots points pour les supports/résistances
   - Vérifier le momentum et ROC pour la force du mouvement
   - Utiliser l'ATR pour calibrer les SL/TP

3. Analyser le sentiment du marché:
   - Examiner le ratio long/short des positions
   - Évaluer le volume des dernières 24h

4. Gérer les positions existantes:
   - Analyser chaque position avec les indicateurs techniques
   - Vérifier le ratio risque/récompense actuel
   - Ajuster les SL/TP si nécessaire selon les niveaux techniques

5. Pour les nouvelles positions:
   - Confirmer la direction avec plusieurs timeframes
   - Vérifier la convergence des indicateurs
   - Utiliser les pivots et l'ATR pour les niveaux d'entrée/sortie
   - Respecter strictement la gestion du risque (max 2%)
   - Calculer les lots selon l'équité et le risque
   - Placer les SL/TP selon les niveaux techniques et l'ATR

6. Respecter les contraintes techniques:
   - Vérifier les digits du symbole pour les prix
   - Respecter les limites de volume min/max
   - Tenir compte des stops_level du broker
   - Respecter les règles MT5 (https://mql5.com/docs)
   - Bien calculer les tp/sl pour pas que il y ait l'erreur invalides stops (recalculer plusieurs fois si il le faut)

Note: Les SL/TP doivent être calculés en points et respecter la distance minimale de {stops_level} points du prix d'entrée.
Si une des vérifications préalables échoue, ne pas générer de nouveaux trades.

FORMAT DE RÉPONSE REQUIS :
{{
    "analysis": {{
        "market_conditions": "description",
        "volatility_assessment": "LOW/MEDIUM/HIGH",
        "risk_level": "LOW/MEDIUM/HIGH",
        "overall_bias": "BULLISH/BEARISH/NEUTRAL",
        "patterns_detected": {{
            "harmonic": "description",
            "elliott_wave": "description", 
            "fibonacci": "description"
        }},
        "historical_performance": {{
            "pattern_success_rate": 0.0,
            "recommended_size": 0.0
        }}
    }},
    "positions_management": [
        {{
            "ticket": 0,
            "action": "HOLD/CLOSE/MODIFY",
            "new_sl": 0.0,
            "new_tp": 0.0,
            "reason": "description"
        }}
    ],
    "new_positions": [
        {{
            "decision": "BUY/SELL",
            "symbol": "symbol",
            "lot": 0.1,
            "sl": 0.0,
            "tp": 0.0,
            "magic": 123456,
            "deviation": 20,
            "reason": "description"
        }}
    ]
}}"""

        # Envoyer la requête au LLM
        response = session.post(
            LLM_API_URL,
            json={
                "model": "mathstral-7b-v0.1",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
        )

        if response.status_code != 200:
            raise Exception(f"Erreur LLM: {response.text}")

        # Traiter la réponse du LLM
        trades = json.loads(response.json()["choices"][0]["message"]["content"])
        print("\nDécisions du LLM:")
        print(f"- Conditions de marché: {trades['analysis']['market_conditions']}")
        print(f"- Volatilité: {trades['analysis']['volatility_assessment']}")
        print(f"- Risque: {trades['analysis']['risk_level']}")
        print(f"- Biais général: {trades['analysis']['overall_bias']}")
        print(f"- Patterns détectés: {trades['analysis']['patterns_detected']}")
        print(f"- Performance historique: {trades['analysis']['historical_performance']}")
        
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message("\nDécisions du LLM:")
            send_discord_message(f"- Conditions de marché: {trades['analysis']['market_conditions']}")
            send_discord_message(f"- Volatilité: {trades['analysis']['volatility_assessment']}")
            send_discord_message(f"- Risque: {trades['analysis']['risk_level']}")
            send_discord_message(f"- Biais général: {trades['analysis']['overall_bias']}")
            send_discord_message(f"- Patterns détectés: {trades['analysis']['patterns_detected']}")
            send_discord_message(f"- Performance historique: {trades['analysis']['historical_performance']}")

        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message("\nDécisions du LLM:")
            send_telegram_message(f"- Conditions de marché: {trades['analysis']['market_conditions']}")
            send_telegram_message(f"- Volatilité: {trades['analysis']['volatility_assessment']}")
            send_telegram_message(f"- Risque: {trades['analysis']['risk_level']}")
            send_telegram_message(f"- Biais général: {trades['analysis']['overall_bias']}")
            send_telegram_message(f"- Patterns détectés: {trades['analysis']['patterns_detected']}")
            send_telegram_message(f"- Performance historique: {trades['analysis']['historical_performance']}")

        # Gérer les positions existantes
        for position in trades.get("positions_management", []):
            print(f"\nGestion position {position['ticket']}:")
            if SEND_DISCORD_NOTIFICATIONS:
                send_discord_message(f"\nGestion position {position['ticket']}:")
            if SEND_TELEGRAM_NOTIFICATIONS:
                send_telegram_message(f"\nGestion position {position['ticket']}:")
            if position["action"] == "MODIFY":
                existing_position = mt5.positions_get(ticket=position["ticket"])
                if not existing_position:
                    print(f"- Position {position['ticket']} n'existe pas")
                    if SEND_DISCORD_NOTIFICATIONS:
                        send_discord_message(f"- Position {position['ticket']} n'existe pas")
                    if SEND_TELEGRAM_NOTIFICATIONS:
                        send_telegram_message(f"- Position {position['ticket']} n'existe pas")
                    continue
                
                request = {
                    "action": mt5.TRADE_ACTION_MODIFY,
                    "position": position["ticket"],
                    "symbol": existing_position[0].symbol,
                    "sl": float(position["new_sl"]),
                    "tp": float(position["new_tp"]),
                    "type_time": mt5.ORDER_TIME_GTC
                }
                print(f"- Modification SL: {position['new_sl']}, TP: {position['new_tp']}")
                if SEND_DISCORD_NOTIFICATIONS:
                    send_discord_message(f"- Modification SL: {position['new_sl']}, TP: {position['new_tp']}")
                if SEND_TELEGRAM_NOTIFICATIONS:
                    send_telegram_message(f"- Modification SL: {position['new_sl']}, TP: {position['new_tp']}")
                result = mt5.order_send(request)
                print(f"- Résultat: {result.comment if result else 'Erreur'}")
                if SEND_DISCORD_NOTIFICATIONS:
                    send_discord_message(f"- Résultat: {result.comment if result else 'Erreur'}")
                if SEND_TELEGRAM_NOTIFICATIONS:
                    send_telegram_message(f"- Résultat: {result.comment if result else 'Erreur'}")

        # Exécuter les nouveaux trades
        for trade in trades.get("new_positions", []):
            print(f"\nNouveau trade {trade['symbol']}:")
            print(f"- Décision: {trade['decision']}")
            print(f"- Lot: {trade['lot']}")
            print(f"- Raison: {trade['reason']}")

            if SEND_DISCORD_NOTIFICATIONS:
                send_discord_message(f"\nNouveau trade {trade['symbol']}:")
                send_discord_message(f"- Décision: {trade['decision']}")
                send_discord_message(f"- Lot: {trade['lot']}")
                send_discord_message(f"- Raison: {trade['reason']}")

            if SEND_TELEGRAM_NOTIFICATIONS:
                send_telegram_message(f"\nNouveau trade {trade['symbol']}:")
                send_telegram_message(f"- Décision: {trade['decision']}")
                send_telegram_message(f"- Lot: {trade['lot']}")
                send_telegram_message(f"- Raison: {trade['reason']}")
            
            symbol = trade["symbol"]
            if not mt5.symbol_select(symbol, True):
                print(f"- Erreur: Symbole {symbol} non disponible")
                if SEND_DISCORD_NOTIFICATIONS:
                    send_discord_message(f"- Erreur: Symbole {symbol} non disponible")
                if SEND_TELEGRAM_NOTIFICATIONS:
                    send_telegram_message(f"- Erreur: Symbole {symbol} non disponible")
                continue

            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                print(f"- Erreur: Info symbole {symbol} non disponible")
                if SEND_DISCORD_NOTIFICATIONS:
                    send_discord_message(f"- Erreur: Info symbole {symbol} non disponible")
                if SEND_TELEGRAM_NOTIFICATIONS:
                    send_telegram_message(f"- Erreur: Info symbole {symbol} non disponible")
                continue

            point = symbol_info.point
            tick = mt5.symbol_info_tick(symbol)
            
            if trade["decision"] in ["BUY", "SELL"]:
                price = tick.ask if trade["decision"] == "BUY" else tick.bid
                sl = price - (abs(float(trade["sl"])) * point) if trade["decision"] == "BUY" else price + (abs(float(trade["sl"])) * point)
                tp = price + (abs(float(trade["tp"])) * point) if trade["decision"] == "BUY" else price - (abs(float(trade["tp"])) * point)
                
                # Arrondir les valeurs selon les digits du symbole
                digits = symbol_info.digits
                price = round(price, digits)
                sl = round(sl, digits)
                tp = round(tp, digits)
                lot = round(float(trade["lot"]), 2)  # Les lots sont généralement limités à 2 décimales
                
                # Vérifier que le lot respecte les limites du symbole
                if lot < symbol_info.volume_min or lot > symbol_info.volume_max:
                    print(f"- Erreur: Volume {lot} hors limites ({symbol_info.volume_min}-{symbol_info.volume_max})")
                    if SEND_DISCORD_NOTIFICATIONS:
                        send_discord_message(f"- Erreur: Volume {lot} hors limites ({symbol_info.volume_min}-{symbol_info.volume_max})")
                    if SEND_TELEGRAM_NOTIFICATIONS:
                        send_telegram_message(f"- Erreur: Volume {lot} hors limites ({symbol_info.volume_min}-{symbol_info.volume_max})")
                    continue
                
                print(f"- Prix: {price}")
                print(f"- SL: {sl}")
                print(f"- TP: {tp}")
                print(f"- Lot: {lot}")

                if SEND_DISCORD_NOTIFICATIONS:
                    send_discord_message(f"- Prix: {price}")
                    send_discord_message(f"- SL: {sl}")
                    send_discord_message(f"- TP: {tp}")
                    send_discord_message(f"- Lot: {lot}")

                if SEND_TELEGRAM_NOTIFICATIONS:
                    send_telegram_message(f"- Prix: {price}")
                    send_telegram_message(f"- SL: {sl}")
                    send_telegram_message(f"- TP: {tp}")
                    send_telegram_message(f"- Lot: {lot}")
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot,  # Utiliser le lot arrondi
                    "type": mt5.ORDER_TYPE_BUY if trade["decision"] == "BUY" else mt5.ORDER_TYPE_SELL,
                    "price": price,  # Prix arrondi
                    "sl": sl,  # SL arrondi
                    "tp": tp,  # TP arrondi
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "LLM trade",
                    "type_time": mt5.ORDER_TIME_GTC,
                }

                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"- Ordre exécuté: Ticket {result.order}")
                    if SEND_DISCORD_NOTIFICATIONS:
                        send_discord_message(f"- Ordre exécuté: Ticket {result.order}")
                    if SEND_TELEGRAM_NOTIFICATIONS:
                        send_telegram_message(f"- Ordre exécuté: Ticket {result.order}")
                else:
                    print(f"- Erreur ordre: {result.comment if result else 'Unknown'}")
                    if SEND_DISCORD_NOTIFICATIONS:
                        send_discord_message(f"- Erreur ordre: {result.comment if result else 'Unknown'}")
                    if SEND_TELEGRAM_NOTIFICATIONS:
                        send_telegram_message(f"- Erreur ordre: {result.comment if result else 'Unknown'}")

        print("\n=== Fin de l'analyse ===\n")
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message("Fin de l'analyse")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message("Fin de l'analyse")
        return jsonify({"status": "success", "trades": trades})

    except Exception as e:
        print(f"\nErreur dans analyze(): {str(e)}")
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message(f"\nErreur dans analyze(): {str(e)}")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message(f"\nErreur dans analyze(): {str(e)}")
        return jsonify({"error": str(e)})

def periodic_analysis():
    """Fonction d'analyse périodique"""
    while True:
        try:
            with app.app_context():
                analyze()
            time.sleep(0)
        except Exception as e:
            print(f"Erreur dans l'analyse périodique: {str(e)}")
            if SEND_DISCORD_NOTIFICATIONS:
                send_discord_message(f"Erreur dans l'analyse périodique: {str(e)}")
            if SEND_TELEGRAM_NOTIFICATIONS:
                send_telegram_message(f"Erreur dans l'analyse périodique: {str(e)}")
            time.sleep(0)

if __name__ == '__main__':
    try:
        print("Démarrage du système...")
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message("Démarrage du système...")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message("Démarrage du système...")
        

        # Initialiser MT5 une seule fois
        if not initialize_mt5():
            if SEND_DISCORD_NOTIFICATIONS:
                send_discord_message("Échec connexion MT5")
            if SEND_TELEGRAM_NOTIFICATIONS:
                send_telegram_message("Échec connexion MT5")
            raise Exception("Échec connexion MT5")
            
        # Démarrer l'analyse périodique
        analysis_thread = Thread(target=periodic_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        # Démarrer Flask
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        print(f"Erreur critique: {str(e)}")
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message(f"Erreur critique: {str(e)}")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message(f"Erreur critique: {str(e)}")
        cleanup()
        exit(1)
    except KeyboardInterrupt:
        print("\nArrêt demandé...")
        if SEND_DISCORD_NOTIFICATIONS:
            send_discord_message("\nArrêt demandé...")
        if SEND_TELEGRAM_NOTIFICATIONS:
            send_telegram_message("\nArrêt demandé...")
        cleanup()
        exit(0)
