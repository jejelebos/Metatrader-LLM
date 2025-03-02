"""
MT5 LLM Trading Bot - Heavy Version
Optimized for high-performance computers with maximum features and capabilities
Author: Jeremy
License: MIT
Version: 1.0

This is the heavy version of the trading bot, featuring:
- Complete multi-timeframe analysis (M1 to MN)
- Advanced technical indicators suite
- Machine Learning integration
- Neural Networks for pattern recognition
- Advanced portfolio management
- Complete risk management suite
- Market sentiment analysis
- News impact analysis
"""

# Standard library imports
from flask import Flask, request, jsonify
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from threading import Thread, Lock
import requests
import json
import gc
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import atexit
from scipy import stats
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import talib
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import torch
import torch.nn as nn
import joblib
import warnings
import psutil
import os
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import aiohttp
import websockets
import threading
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# Configure logging with advanced formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.FileHandler('heavy_trader.log'),
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            'heavy_trader_rotating.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask application with advanced configuration
app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-size

# ============ CONFIGURATION SECTION ============
# LLM API Configuration
LLM_API_URL = "http://YOUR_LLM_API_URL:PORT/v1/chat/completions"
LLM_MODEL = "YOUR_MODEL_NAME"
LLM_TIMEOUT = 120  # Extended timeout for complex analysis
LLM_MAX_TOKENS = 4096
LLM_TEMPERATURE = 0.2

# MT5 Configuration
MT5_LOGIN = 0000000
MT5_PASSWORD = "YOUR_PASSWORD"
MT5_SERVER = "YOUR_BROKER_SERVER"

# Trading Parameters - Extended
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", 
    "USDCAD", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP",
    "XAUUSD", "XAGUSD", "US30", "US500", "USTEC"
]

TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1
}

# Analysis Parameters
ANALYSIS_INTERVAL = 60  # 1 minute
HISTORY_BARS = {
    "M1": 1000,
    "M5": 1000,
    "M15": 1000,
    "M30": 500,
    "H1": 500,
    "H4": 500,
    "D1": 200,
    "W1": 100,
    "MN1": 50
}

# Trading Limits
MAX_POSITIONS = 10
MAX_POSITIONS_PER_SYMBOL = 3
MAX_DAILY_TRADES = 50
MAX_RETRIES = 5

# Risk Management
MAX_RISK_PERCENT = 2.0
MAX_PORTFOLIO_RISK = 5.0
MAX_CORRELATION_RISK = 0.8
MAX_SPREAD_POINTS = 25
MIN_POSITION_INTERVAL = 60
POSITION_SIZING_MODELS = ["FIXED", "ATR", "KELLY", "OPTIMAL_F"]
VAR_CONFIDENCE_LEVEL = 0.99
STRESS_TEST_SCENARIOS = ["HISTORICAL", "MONTE_CARLO", "CUSTOM"]

# Performance Management
MAX_CACHE_SIZE = 500 * 1024 * 1024  # 500MB
CLEANUP_INTERVAL = 300  # 5 minutes
BATCH_SIZE = 5000
MAX_THREADS = psutil.cpu_count()
MAX_PROCESSES = psutil.cpu_count() - 1

# Machine Learning Configuration
ML_MODELS_PATH = "./ml_models"
ML_DATA_PATH = "./ml_data"
FEATURE_STORE_PATH = "./feature_store"
MODEL_UPDATE_INTERVAL = 3600  # 1 hour
PREDICTION_THRESHOLD = 0.75
RETRAINING_THRESHOLD = 0.65

# Neural Network Configuration
NN_MODELS_PATH = "./nn_models"
NN_LAYERS = [50, 100, 50]
NN_DROPOUT = 0.2
NN_EPOCHS = 100
NN_BATCH_SIZE = 32
NN_VALIDATION_SPLIT = 0.2

# Pattern Recognition
PATTERN_CONFIDENCE_THRESHOLD = 0.8
HARMONIC_PATTERNS = ["GARTLEY", "BUTTERFLY", "BAT", "CRAB", "SHARK"]
ELLIOTT_WAVE_PATTERNS = ["IMPULSE", "CORRECTION"]
CHART_PATTERNS = ["HEAD_SHOULDERS", "DOUBLE_TOP", "TRIANGLE", "CHANNEL"]

# Market Sentiment Analysis
SENTIMENT_SOURCES = ["NEWS", "SOCIAL_MEDIA", "TECHNICAL", "FUNDAMENTAL"]
SENTIMENT_WEIGHTS = {
    "NEWS": 0.3,
    "SOCIAL_MEDIA": 0.2,
    "TECHNICAL": 0.3,
    "FUNDAMENTAL": 0.2
}

# HTTP Session Configuration
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=0.5,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
)
session.mount('http://', HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100))
session.mount('https://', HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100))

# Thread and Process Pools
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
process_pool = ProcessPoolExecutor(max_workers=MAX_PROCESSES)

# Locks and Synchronization
mt5_lock = Lock()
cache_lock = Lock()
model_lock = Lock()

# ============ GLOBAL VARIABLES ============
mt5_initialized = False
system_status = {
    "mt5_ready": False,
    "llm_ready": False,
    "ml_ready": False,
    "nn_ready": False,
    "last_error": None,
    "last_cleanup": datetime.now(),
    "memory_usage": 0,
    "performance_metrics": {
        "analysis_time": 0,
        "prediction_accuracy": 0,
        "cpu_usage": 0,
        "gpu_usage": 0,
        "active_threads": 0,
        "active_processes": 0
    },
    "trading_metrics": {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "total_profit": 0,
        "max_drawdown": 0,
        "sharpe_ratio": 0
    }
}

# Advanced Caching System
cache_system = {
    "market_data": {},
    "indicators": {},
    "patterns": {},
    "predictions": {},
    "sentiment": {},
    "models": {},
    "statistics": {}
}

# Initialize ML models
ml_models = {
    "trend_classifier": None,
    "pattern_recognizer": None,
    "risk_analyzer": None,
    "sentiment_analyzer": None,
    "portfolio_optimizer": None
}

# Initialize NN models
nn_models = {
    "price_predictor": None,
    "pattern_detector": None,
    "market_regime": None,
    "volatility_predictor": None
}

# ============ CORE FUNCTIONS ============
def cleanup():
    """
    Fonction de nettoyage avancée avec gestion des ressources
    """
    logger.info("Démarrage du nettoyage système...")
    try:
        # Nettoyage MT5
        with mt5_lock:
            if mt5.initialize():
                mt5.shutdown()
        
        # Nettoyage des pools
        thread_pool.shutdown(wait=True)
        process_pool.shutdown(wait=True)
        
        # Sauvegarde des modèles ML
        for name, model in ml_models.items():
            if model:
                joblib.dump(model, f"{ML_MODELS_PATH}/{name}.joblib")
        
        # Sauvegarde des modèles NN
        for name, model in nn_models.items():
            if model:
                model.save(f"{NN_MODELS_PATH}/{name}.h5")
        
        # Nettoyage du cache
        with cache_lock:
            cache_system.clear()
            gc.collect()
        
        # Libération GPU si utilisé
        try:
            torch.cuda.empty_cache()
        except:
            pass
        
        logger.info("Nettoyage système terminé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage: {str(e)}")

def initialize_mt5() -> bool:
    """
    Initialisation avancée de MT5 avec vérifications complètes
    """
    logger.info("Initialisation de MT5...")
    try:
        with mt5_lock:
            # Vérification des ressources système
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                logger.warning("Mémoire système critique (>90%)")
                gc.collect()
            
            # Initialisation MT5 avec paramètres avancés
            if not mt5.initialize(
                login=MT5_LOGIN,
                password=MT5_PASSWORD,
                server=MT5_SERVER,
                timeout=60000,  # Timeout étendu
                portable=False
            ):
                error = mt5.last_error()
                logger.error(f"Échec initialisation MT5: {error}")
                return False
            
            # Vérifications étendues
            account_info = mt5.account_info()
            if not account_info:
                logger.error("Impossible d'obtenir les informations du compte")
                return False
            
            # Configuration des symboles
            for symbol in SYMBOLS:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    logger.warning(f"Symbole {symbol} non trouvé")
                    continue
                if not symbol_info.visible:
                    if not mt5.symbol_select(symbol, True):
                        logger.warning(f"Échec sélection symbole {symbol}")
            
            # Vérification des timeframes
            for tf in TIMEFRAMES.values():
                if not mt5.copy_rates_from_pos("EURUSD", tf, 0, 1):
                    logger.warning(f"Timeframe {tf} non disponible")
            
            logger.info("Initialisation MT5 réussie")
            system_status["mt5_ready"] = True
            return True
            
    except Exception as e:
        logger.error(f"Erreur critique lors de l'initialisation MT5: {str(e)}")
        return False

# Enregistrement du nettoyage à la sortie
atexit.register(cleanup)

def get_advanced_technical_analysis(symbol: str, timeframes: List[str] = None) -> Dict[str, Any]:
    """
    Analyse technique avancée multi-timeframe avec indicateurs complexes
    """
    if timeframes is None:
        timeframes = list(TIMEFRAMES.keys())
    
    analysis_results = {}
    try:
        for tf in timeframes:
            # Récupération des données
            rates = mt5.copy_rates_from_pos(symbol, TIMEFRAMES[tf], 0, HISTORY_BARS[tf])
            if rates is None:
                continue
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Analyse avancée par timeframe
            analysis = {
                # Moyennes mobiles avancées
                'ema_fast': talib.EMA(df['close'], timeperiod=8),
                'ema_medium': talib.EMA(df['close'], timeperiod=21),
                'ema_slow': talib.EMA(df['close'], timeperiod=55),
                'tema': talib.TEMA(df['close'], timeperiod=21),
                'kama': talib.KAMA(df['close'], timeperiod=30),
                
                # Oscillateurs avancés
                'rsi': talib.RSI(df['close'], timeperiod=14),
                'stoch_k', 'stoch_d': talib.STOCH(df['high'], df['low'], df['close']),
                'macd', 'macd_signal', 'macd_hist': talib.MACD(df['close']),
                'cci': talib.CCI(df['high'], df['low'], df['close']),
                'mfi': talib.MFI(df['high'], df['low'], df['close'], df['tick_volume']),
                
                # Analyse de volume
                'obv': talib.OBV(df['close'], df['tick_volume']),
                'ad': talib.AD(df['high'], df['low'], df['close'], df['tick_volume']),
                'adosc': talib.ADOSC(df['high'], df['low'], df['close'], df['tick_volume']),
                
                # Indicateurs de volatilité
                'atr': talib.ATR(df['high'], df['low'], df['close']),
                'natr': talib.NATR(df['high'], df['low'], df['close']),
                'trange': talib.TRANGE(df['high'], df['low'], df['close']),
                
                # Bandes et canaux
                'bbands_upper', 'bbands_middle', 'bbands_lower': talib.BBANDS(df['close']),
                'keltner_upper', 'keltner_middle', 'keltner_lower': calculate_keltner_channels(df),
                
                # Patterns avancés
                'patterns': detect_candlestick_patterns(df),
                'harmonic_patterns': detect_harmonic_patterns(df),
                'elliott_waves': analyze_elliott_waves(df),
                
                # Analyses statistiques
                'volatility': calculate_volatility_metrics(df),
                'momentum': calculate_momentum_metrics(df),
                'trend_strength': calculate_trend_strength(df)
            }
            
            # Nettoyage des NaN et mise en forme
            analysis = {k: v.dropna().tolist() if isinstance(v, pd.Series) else v 
                       for k, v in analysis.items()}
            
            analysis_results[tf] = analysis
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse technique: {str(e)}")
        return None

def detect_candlestick_patterns(df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Détection avancée des patterns de chandeliers
    """
    patterns = {}
    try:
        # Patterns de retournement
        patterns['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        patterns['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        patterns['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        patterns['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
        
        # Patterns de continuation
        patterns['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        patterns['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
        patterns['three_black_crows'] = talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
        
        # Nettoyage et formatage
        patterns = {k: v[v != 0].tolist() for k, v in patterns.items()}
        return patterns
        
    except Exception as e:
        logger.error(f"Erreur lors de la détection des patterns: {str(e)}")
        return {}

class MarketPredictor:
    """
    Classe pour la prédiction avancée des marchés utilisant ML et DL
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.initialize_models()
        
    def initialize_models(self):
        """
        Initialisation des modèles ML et DL
        """
        try:
            # Modèle Random Forest pour la classification des tendances
            self.trend_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Modèle LSTM pour la prédiction des prix
            self.price_predictor = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(30, 10)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=1)
            ])
            self.price_predictor.compile(optimizer='adam', loss='mse')
            
            # Modèle d'anomalie pour la détection des événements rares
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Modèle de clustering pour l'analyse des régimes de marché
            self.market_regime_classifier = KMeans(
                n_clusters=4,
                random_state=42
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des modèles: {str(e)}")
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Préparation des features pour les modèles
        """
        features = []
        try:
            # Features techniques
            features.extend([
                df['close'].pct_change(),
                df['volume'].pct_change(),
                talib.RSI(df['close']),
                talib.ATR(df['high'], df['low'], df['close']),
                pd.Series(talib.MACD(df['close'])[0])
            ])
            
            # Features de volatilité
            features.extend([
                df['high'] - df['low'],
                df['close'].rolling(window=20).std(),
                df['volume'].rolling(window=20).std()
            ])
            
            # Features de momentum
            features.extend([
                df['close'].diff(),
                df['close'].diff(5),
                df['close'].diff(20)
            ])
            
            # Nettoyage et mise en forme
            features_df = pd.concat(features, axis=1).dropna()
            return self.scaler.fit_transform(features_df)
            
        except Exception as e:
            logger.error(f"Erreur lors de la préparation des features: {str(e)}")
            return None
    
    def predict_market(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Prédiction complète du marché
        """
        try:
            # Récupération des données
            rates = mt5.copy_rates_from_pos(symbol, TIMEFRAMES[timeframe], 0, HISTORY_BARS[timeframe])
            df = pd.DataFrame(rates)
            
            # Préparation des features
            X = self.prepare_features(df)
            if X is None:
                return None
            
            # Prédictions
            predictions = {
                'trend': self.trend_classifier.predict(X)[-1],
                'price': self.price_predictor.predict(X.reshape((1, 30, 10)))[-1][0],
                'anomaly': self.anomaly_detector.predict(X)[-1],
                'regime': self.market_regime_classifier.predict(X)[-1]
            }
            
            # Calcul des probabilités et métriques de confiance
            predictions['trend_proba'] = self.trend_classifier.predict_proba(X)[-1]
            predictions['confidence'] = calculate_prediction_confidence(predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            return None

def calculate_prediction_confidence(predictions: Dict[str, Any]) -> float:
    """
    Calcul du niveau de confiance des prédictions
    """
    try:
        # Pondération des différentes métriques
        confidence_scores = {
            'trend_confidence': np.max(predictions['trend_proba']),
            'price_stability': 1 - np.std(predictions['price']) / np.mean(predictions['price']),
            'anomaly_score': 1 if predictions['anomaly'] == 1 else 0.5,
            'regime_stability': calculate_regime_stability(predictions['regime'])
        }
        
        # Moyenne pondérée
        weights = {
            'trend_confidence': 0.4,
            'price_stability': 0.3,
            'anomaly_score': 0.2,
            'regime_stability': 0.1
        }
        
        return sum(score * weights[metric] for metric, score in confidence_scores.items())
        
    except Exception as e:
        logger.error(f"Erreur lors du calcul de la confiance: {str(e)}")
        return 0.0

class PortfolioManager:
    """
    Gestionnaire avancé de portfolio avec optimisation et gestion des risques
    """
    def __init__(self):
        self.positions = {}
        self.risk_metrics = {}
        self.performance_metrics = {}
        self.optimization_params = {
            'max_position_size': 0.1,  # 10% max par position
            'max_sector_exposure': 0.3,  # 30% max par secteur
            'min_sharpe_ratio': 1.5,
            'max_drawdown': 0.2,  # 20% drawdown max
            'rebalancing_threshold': 0.05  # 5% déviation pour rebalancement
        }
    
    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """
        Calcul des métriques avancées du portfolio
        """
        try:
            metrics = {}
            
            # Récupération des positions actuelles
            positions = mt5.positions_get()
            if positions is None:
                return None
            
            # Calcul des métriques de base
            total_equity = mt5.account_info().equity
            position_values = [pos.volume * pos.price_current for pos in positions]
            
            # Métriques de performance
            metrics['total_value'] = sum(position_values)
            metrics['pnl'] = sum(pos.profit + pos.swap for pos in positions)
            metrics['roi'] = metrics['pnl'] / total_equity if total_equity > 0 else 0
            
            # Métriques de risque
            metrics['var_95'] = self.calculate_var(position_values, confidence=0.95)
            metrics['cvar_95'] = self.calculate_cvar(position_values, confidence=0.95)
            metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(position_values)
            metrics['max_drawdown'] = self.calculate_max_drawdown(position_values)
            
            # Métriques de diversification
            metrics['correlation_matrix'] = self.calculate_correlation_matrix(positions)
            metrics['concentration_index'] = self.calculate_herfindahl_index(position_values)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des métriques: {str(e)}")
            return None
    
    def optimize_portfolio(self) -> Dict[str, Any]:
        """
        Optimisation du portfolio selon plusieurs critères
        """
        try:
            # Récupération des données actuelles
            positions = mt5.positions_get()
            if positions is None:
                return None
            
            # Préparation des données pour l'optimisation
            position_data = {
                pos.symbol: {
                    'volume': pos.volume,
                    'price': pos.price_current,
                    'profit': pos.profit,
                    'swap': pos.swap,
                    'margin': pos.margin
                } for pos in positions
            }
            
            # Calcul des poids optimaux
            optimal_weights = self.calculate_optimal_weights(position_data)
            
            # Génération des ordres de rebalancement
            rebalancing_orders = self.generate_rebalancing_orders(
                position_data,
                optimal_weights
            )
            
            return {
                'optimal_weights': optimal_weights,
                'rebalancing_orders': rebalancing_orders,
                'expected_metrics': self.simulate_portfolio_metrics(optimal_weights)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation: {str(e)}")
            return None
    
    def manage_risk(self) -> Dict[str, Any]:
        """
        Gestion avancée des risques
        """
        try:
            risk_assessment = {
                'market_risk': self.assess_market_risk(),
                'position_risk': self.assess_position_risk(),
                'correlation_risk': self.assess_correlation_risk(),
                'liquidity_risk': self.assess_liquidity_risk(),
                'volatility_risk': self.assess_volatility_risk()
            }
            
            # Génération des actions de mitigation
            risk_actions = self.generate_risk_actions(risk_assessment)
            
            # Mise à jour des stops et limites
            self.update_position_limits(risk_actions)
            
            return {
                'risk_assessment': risk_assessment,
                'risk_actions': risk_actions,
                'updated_limits': self.get_current_limits()
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la gestion des risques: {str(e)}")
            return None
    
    def execute_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Exécution sécurisée des trades avec vérifications avancées
        """
        try:
            results = []
            for trade in trades:
                # Vérifications pré-trade
                if not self.validate_trade(trade):
                    continue
                
                # Calcul du sizing optimal
                trade_size = self.calculate_position_size(trade)
                
                # Exécution avec retry et gestion d'erreur
                for _ in range(MAX_RETRIES):
                    result = mt5.order_send(self.prepare_trade_request(trade, trade_size))
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        results.append({
                            'trade': trade,
                            'result': result._asdict(),
                            'metrics': self.calculate_trade_metrics(result)
                        })
                        break
                        
            return {
                'success_rate': len(results) / len(trades),
                'results': results,
                'portfolio_impact': self.calculate_portfolio_impact(results)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution des trades: {str(e)}")
            return None

class SentimentAnalyzer:
    """
    Analyseur avancé de sentiment de marché
    """
    def __init__(self):
        self.sentiment_cache = {}
        self.news_sources = []
        self.social_feeds = []
        self.sentiment_history = {}
        self.correlation_metrics = {}
        
    def analyze_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Analyse complète du sentiment de marché
        """
        try:
            sentiment_data = {
                'news': self.analyze_news_sentiment(symbol),
                'social': self.analyze_social_sentiment(symbol),
                'technical': self.analyze_technical_sentiment(symbol),
                'fundamental': self.analyze_fundamental_sentiment(symbol)
            }
            
            # Calcul du sentiment global pondéré
            weighted_sentiment = sum(
                sentiment_data[source] * SENTIMENT_WEIGHTS[source.upper()]
                for source in sentiment_data
            )
            
            # Analyse d'impact
            impact_analysis = self.analyze_sentiment_impact(
                symbol,
                sentiment_data,
                weighted_sentiment
            )
            
            return {
                'sentiment_data': sentiment_data,
                'weighted_sentiment': weighted_sentiment,
                'impact_analysis': impact_analysis,
                'confidence_score': self.calculate_sentiment_confidence(sentiment_data),
                'historical_correlation': self.get_sentiment_price_correlation(symbol)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse du sentiment: {str(e)}")
            return None
    
    def analyze_news_sentiment(self, symbol: str) -> float:
        """
        Analyse du sentiment des actualités
        """
        try:
            news_items = self.fetch_recent_news(symbol)
            if not news_items:
                return 0.0
            
            sentiments = []
            for news in news_items:
                # Analyse NLP du titre et du contenu
                title_sentiment = self.analyze_text_sentiment(news['title'])
                content_sentiment = self.analyze_text_sentiment(news['content'])
                
                # Pondération basée sur la pertinence et la fraîcheur
                weight = self.calculate_news_weight(news)
                sentiments.append((title_sentiment * 0.3 + content_sentiment * 0.7) * weight)
            
            return sum(sentiments) / len(sentiments)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des news: {str(e)}")
            return 0.0

class EventManager:
    """
    Gestionnaire avancé d'événements de marché
    """
    def __init__(self):
        self.active_events = {}
        self.event_history = {}
        self.impact_metrics = {}
        self.alert_thresholds = {
            'high_impact': 0.8,
            'medium_impact': 0.5,
            'low_impact': 0.2
        }
    
    def monitor_market_events(self) -> Dict[str, Any]:
        """
        Surveillance continue des événements de marché
        """
        try:
            current_events = {
                'economic': self.monitor_economic_events(),
                'geopolitical': self.monitor_geopolitical_events(),
                'corporate': self.monitor_corporate_events(),
                'market_specific': self.monitor_market_specific_events()
            }
            
            # Analyse d'impact
            impact_analysis = self.analyze_events_impact(current_events)
            
            # Génération d'alertes
            alerts = self.generate_event_alerts(impact_analysis)
            
            # Mise à jour de l'historique
            self.update_event_history(current_events, impact_analysis)
            
            return {
                'current_events': current_events,
                'impact_analysis': impact_analysis,
                'alerts': alerts,
                'risk_assessment': self.assess_event_risks(current_events),
                'recommended_actions': self.generate_event_actions(impact_analysis)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la surveillance des événements: {str(e)}")
            return None
    
    def analyze_events_impact(self, events: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyse de l'impact des événements sur le marché
        """
        try:
            impact_scores = {}
            for event_type, events_list in events.items():
                for event in events_list:
                    # Calcul du score d'impact
                    impact_score = self.calculate_event_impact(event)
                    
                    # Analyse de la corrélation historique
                    historical_correlation = self.get_historical_correlation(event)
                    
                    # Ajustement du score basé sur le contexte actuel
                    adjusted_score = self.adjust_impact_score(
                        impact_score,
                        historical_correlation,
                        self.get_market_context()
                    )
                    
                    impact_scores[event['id']] = {
                        'raw_score': impact_score,
                        'adjusted_score': adjusted_score,
                        'confidence': self.calculate_impact_confidence(event),
                        'duration_estimate': self.estimate_impact_duration(event)
                    }
            
            return impact_scores
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse d'impact: {str(e)}")
            return {}

class SystemManager:
    """
    Gestionnaire système pour l'initialisation et la maintenance
    """
    def __init__(self):
        self.start_time = datetime.now()
        self.performance_monitor = None
        self.resource_manager = None
        self.background_tasks = []
        
    def initialize_system(self) -> bool:
        """
        Initialisation complète du système
        """
        try:
            logger.info("Démarrage de l'initialisation du système...")
            
            # Création des répertoires nécessaires
            self.create_required_directories()
            
            # Initialisation des composants
            if not self.initialize_components():
                return False
            
            # Démarrage du monitoring
            self.start_performance_monitoring()
            
            # Démarrage des tâches de fond
            self.start_background_tasks()
            
            logger.info("Système initialisé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du système: {str(e)}")
            return False
    
    def create_required_directories(self):
        """
        Création des répertoires nécessaires
        """
        directories = [
            ML_MODELS_PATH,
            ML_DATA_PATH,
            NN_MODELS_PATH,
            FEATURE_STORE_PATH,
            './logs',
            './cache',
            './backups'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def initialize_components(self) -> bool:
        """
        Initialisation des composants principaux
        """
        try:
            # Initialisation MT5
            if not initialize_mt5():
                return False
            
            # Initialisation des modèles ML
            self.load_ml_models()
            
            # Initialisation des modèles NN
            self.load_nn_models()
            
            # Initialisation des analyseurs
            self.initialize_analyzers()
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des composants: {str(e)}")
            return False
    
    def start_performance_monitoring(self):
        """
        Démarrage du monitoring des performances
        """
        def monitor_performance():
            while True:
                try:
                    # Mesures système
                    cpu_usage = psutil.cpu_percent(interval=1)
                    memory_usage = psutil.virtual_memory().percent
                    
                    # Mise à jour des métriques
                    system_status['performance_metrics'].update({
                        'cpu_usage': cpu_usage,
                        'memory_usage': memory_usage,
                        'active_threads': threading.active_count(),
                        'active_processes': len(psutil.Process().children())
                    })
                    
                    # Vérification des seuils
                    if memory_usage > 90:
                        self.handle_high_memory_usage()
                    if cpu_usage > 90:
                        self.handle_high_cpu_usage()
                    
                    time.sleep(60)  # Intervalle de monitoring
                    
                except Exception as e:
                    logger.error(f"Erreur monitoring performance: {str(e)}")
                    time.sleep(300)  # Attente plus longue en cas d'erreur
        
        self.performance_monitor = Thread(
            target=monitor_performance,
            name="PerformanceMonitor",
            daemon=True
        )
        self.performance_monitor.start()
    
    def start_background_tasks(self):
        """
        Démarrage des tâches de fond
        """
        background_tasks = [
            {
                'name': 'ModelUpdater',
                'target': self.update_models_periodically,
                'interval': MODEL_UPDATE_INTERVAL
            },
            {
                'name': 'CacheManager',
                'target': self.manage_cache,
                'interval': CLEANUP_INTERVAL
            },
            {
                'name': 'MetricsCollector',
                'target': self.collect_metrics,
                'interval': 300
            }
        ]
        
        for task in background_tasks:
            thread = Thread(
                target=self.run_periodic_task,
                args=(task['target'], task['interval']),
                name=task['name'],
                daemon=True
            )
            thread.start()
            self.background_tasks.append(thread)
    
    def run_periodic_task(self, task_func, interval):
        """
        Exécution périodique d'une tâche
        """
        while True:
            try:
                task_func()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Erreur tâche périodique {task_func.__name__}: {str(e)}")
                time.sleep(interval * 2)  # Attente plus longue en cas d'erreur

# Initialisation du gestionnaire système
system_manager = SystemManager()

# ============ API ROUTES ============
@app.route('/health')
def health_check():
    """
    Vérification complète de l'état du système
    """
    try:
        health_status = {
            'status': 'healthy' if all(system_status.values()) else 'degraded',
            'components': {
                'mt5': check_mt5_health(),
                'ml_models': check_ml_models_health(),
                'sentiment_analyzer': check_sentiment_analyzer_health(),
                'portfolio_manager': check_portfolio_manager_health()
            },
            'metrics': {
                'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,
                'cpu_usage': psutil.Process().cpu_percent(),
                'uptime': calculate_uptime(),
                'active_threads': threading.active_count()
            },
            'last_errors': get_recent_errors()
        }
        return jsonify(health_status), 200
    except Exception as e:
        logger.error(f"Erreur health check: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_market():
    """
    Analyse complète du marché avec ML et sentiment
    """
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        timeframes = data.get('timeframes', list(TIMEFRAMES.keys()))
        
        if not symbol:
            return jsonify({'error': 'Symbol requis'}), 400
            
        # Analyse technique
        technical_analysis = get_advanced_technical_analysis(symbol, timeframes)
        
        # Prédictions ML
        market_predictor = MarketPredictor()
        predictions = {tf: market_predictor.predict_market(symbol, tf) 
                      for tf in timeframes}
        
        # Analyse de sentiment
        sentiment_analyzer = SentimentAnalyzer()
        sentiment = sentiment_analyzer.analyze_market_sentiment(symbol)
        
        # Gestion du portfolio
        portfolio_manager = PortfolioManager()
        portfolio_analysis = portfolio_manager.calculate_portfolio_metrics()
        
        # Surveillance des événements
        event_manager = EventManager()
        events = event_manager.monitor_market_events()
        
        # Compilation des résultats
        analysis_results = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'technical_analysis': technical_analysis,
            'ml_predictions': predictions,
            'sentiment_analysis': sentiment,
            'portfolio_metrics': portfolio_analysis,
            'market_events': events,
            'risk_assessment': calculate_global_risk(
                technical_analysis,
                predictions,
                sentiment,
                portfolio_analysis,
                events
            ),
            'trading_recommendations': generate_trading_recommendations(
                symbol,
                technical_analysis,
                predictions,
                sentiment,
                portfolio_analysis,
                events
            )
        }
        
        return jsonify(analysis_results), 200
        
    except Exception as e:
        logger.error(f"Erreur analyse marché: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize_portfolio():
    """
    Optimisation complète du portfolio
    """
    try:
        data = request.get_json()
        optimization_params = data.get('params', {})
        
        portfolio_manager = PortfolioManager()
        optimization_results = portfolio_manager.optimize_portfolio()
        
        if optimization_results:
            return jsonify(optimization_results), 200
        else:
            return jsonify({'error': 'Échec optimisation'}), 400
            
    except Exception as e:
        logger.error(f"Erreur optimisation portfolio: {str(e)}")
        return jsonify({'error': str(e)}), 500

def calculate_global_risk(
    technical_analysis: Dict[str, Any],
    predictions: Dict[str, Any],
    sentiment: Dict[str, Any],
    portfolio_metrics: Dict[str, Any],
    market_events: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calcul du risque global basé sur toutes les analyses
    """
    try:
        risk_components = {
            'technical_risk': calculate_technical_risk(technical_analysis),
            'prediction_risk': calculate_prediction_risk(predictions),
            'sentiment_risk': calculate_sentiment_risk(sentiment),
            'portfolio_risk': calculate_portfolio_risk(portfolio_metrics),
            'event_risk': calculate_event_risk(market_events)
        }
        
        # Calcul du risque global pondéré
        weights = {
            'technical_risk': 0.3,
            'prediction_risk': 0.2,
            'sentiment_risk': 0.15,
            'portfolio_risk': 0.25,
            'event_risk': 0.1
        }
        
        global_risk = sum(
            score * weights[component]
            for component, score in risk_components.items()
        )
        
        return {
            'global_risk': global_risk,
            'risk_components': risk_components,
            'risk_level': categorize_risk_level(global_risk),
            'mitigation_suggestions': suggest_risk_mitigation(risk_components)
        }
        
    except Exception as e:
        logger.error(f"Erreur calcul risque global: {str(e)}")
        return None

# ============ TECHNICAL ANALYSIS FUNCTIONS ============
def calculate_keltner_channels(df: pd.DataFrame, period: int = 20, atr_multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcul des bandes de Keltner
    """
    try:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        ema = talib.EMA(typical_price, timeperiod=period)
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
        
        upper = ema + (atr * atr_multiplier)
        lower = ema - (atr * atr_multiplier)
        
        return upper, ema, lower
    except Exception as e:
        logger.error(f"Erreur calcul Keltner: {str(e)}")
        return None, None, None

def detect_harmonic_patterns(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    Détection des patterns harmoniques avancés
    """
    try:
        patterns = {pattern: [] for pattern in HARMONIC_PATTERNS}
        
        for i in range(len(df) - 5):
            window = df.iloc[i:i+5]
            
            # Calcul des ratios de Fibonacci
            moves = calculate_price_moves(window)
            ratios = calculate_fib_ratios(moves)
            
            # Vérification des patterns
            for pattern in HARMONIC_PATTERNS:
                if matches_harmonic_pattern(ratios, pattern):
                    patterns[pattern].append({
                        'start_idx': i,
                        'end_idx': i + 4,
                        'points': extract_pattern_points(window),
                        'confidence': calculate_pattern_confidence(ratios, pattern)
                    })
        
        return patterns
    except Exception as e:
        logger.error(f"Erreur détection patterns harmoniques: {str(e)}")
        return {}

def analyze_elliott_waves(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyse des vagues d'Elliott avancée
    """
    try:
        waves = {
            'impulse': detect_impulse_waves(df),
            'correction': detect_correction_waves(df),
            'wave_count': count_waves(df),
            'current_wave': identify_current_wave(df),
            'projections': calculate_wave_projections(df)
        }
        
        # Validation et qualification
        waves['confidence'] = validate_wave_structure(waves)
        waves['next_targets'] = project_next_targets(waves)
        
        return waves
    except Exception as e:
        logger.error(f"Erreur analyse Elliott: {str(e)}")
        return {}

def calculate_volatility_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcul des métriques de volatilité avancées
    """
    try:
        returns = df['close'].pct_change().dropna()
        
        metrics = {
            'daily_volatility': returns.std(),
            'annualized_volatility': returns.std() * np.sqrt(252),
            'parkinson': calculate_parkinson_volatility(df),
            'garman_klass': calculate_garman_klass_volatility(df),
            'yang_zhang': calculate_yang_zhang_volatility(df),
            'volatility_skew': calculate_volatility_skew(returns),
            'volatility_regime': identify_volatility_regime(returns)
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Erreur calcul volatilité: {str(e)}")
        return {}

def calculate_momentum_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcul des métriques de momentum avancées
    """
    try:
        metrics = {
            'roc': talib.ROC(df['close'], timeperiod=14),
            'mom': talib.MOM(df['close'], timeperiod=14),
            'trix': talib.TRIX(df['close'], timeperiod=30),
            'willr': talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14),
            'ultimate': talib.ULTOSC(df['high'], df['low'], df['close']),
            'fisher_transform': calculate_fisher_transform(df),
            'momentum_quality': calculate_momentum_quality(df)
        }
        
        return {k: v[-1] for k, v in metrics.items() if v is not None}
    except Exception as e:
        logger.error(f"Erreur calcul momentum: {str(e)}")
        return {}

def calculate_trend_strength(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcul de la force de la tendance avec indicateurs avancés
    """
    try:
        metrics = {
            'adx': talib.ADX(df['high'], df['low'], df['close'], timeperiod=14),
            'aroon_osc': talib.AROONOSC(df['high'], df['low'], timeperiod=14),
            'cci': talib.CCI(df['high'], df['low'], df['close'], timeperiod=14),
            'dx': talib.DX(df['high'], df['low'], df['close'], timeperiod=14),
            'trend_intensity': calculate_trend_intensity(df),
            'trend_quality': calculate_trend_quality(df),
            'trend_consistency': calculate_trend_consistency(df)
        }
        
        return {k: float(v[-1]) for k, v in metrics.items() if v is not None}
    except Exception as e:
        logger.error(f"Erreur calcul force tendance: {str(e)}")
        return {}

# ============ RISK MANAGEMENT FUNCTIONS ============
def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calcul de la Value at Risk avec méthodes multiples
    """
    try:
        # VaR historique
        historical_var = np.percentile(returns, (1 - confidence) * 100)
        
        # VaR paramétrique
        mean = np.mean(returns)
        std = np.std(returns)
        parametric_var = stats.norm.ppf(1 - confidence, mean, std)
        
        # VaR conditionnelle
        cvar = calculate_cvar(returns, confidence)
        
        # VaR Monte Carlo
        mc_var = calculate_monte_carlo_var(returns, confidence)
        
        # Pondération des différentes méthodes
        weighted_var = (
            historical_var * 0.3 +
            parametric_var * 0.3 +
            cvar * 0.2 +
            mc_var * 0.2
        )
        
        return abs(weighted_var)
    except Exception as e:
        logger.error(f"Erreur calcul VaR: {str(e)}")
        return None

def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calcul de la Conditional Value at Risk (Expected Shortfall)
    """
    try:
        var = np.percentile(returns, (1 - confidence) * 100)
        return np.mean(returns[returns <= var])
    except Exception as e:
        logger.error(f"Erreur calcul CVaR: {str(e)}")
        return None

def calculate_monte_carlo_var(returns: np.ndarray, confidence: float = 0.95, simulations: int = 10000) -> float:
    """
    Calcul de la VaR par simulation Monte Carlo
    """
    try:
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Génération des simulations
        sim_returns = np.random.normal(mean, std, simulations)
        
        return np.percentile(sim_returns, (1 - confidence) * 100)
    except Exception as e:
        logger.error(f"Erreur calcul Monte Carlo VaR: {str(e)}")
        return None

# ============ MACHINE LEARNING FUNCTIONS ============
def train_ml_models(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Entraînement des modèles ML avec validation croisée
    """
    try:
        X, y = prepare_training_data(data)
        
        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Entraînement des modèles
        models = {
            'trend': train_trend_classifier(X_train, y_train),
            'regime': train_regime_classifier(X_train, y_train),
            'volatility': train_volatility_predictor(X_train, y_train),
            'price': train_price_predictor(X_train, y_train)
        }
        
        # Validation et métriques
        metrics = validate_models(models, X_test, y_test)
        
        return {
            'models': models,
            'metrics': metrics,
            'feature_importance': calculate_feature_importance(models)
        }
    except Exception as e:
        logger.error(f"Erreur entraînement ML: {str(e)}")
        return None

def train_deep_learning_models(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Entraînement des modèles de Deep Learning
    """
    try:
        X, y = prepare_sequence_data(data)
        
        # Configuration des modèles
        models = {
            'lstm': build_lstm_model(X.shape),
            'gru': build_gru_model(X.shape),
            'transformer': build_transformer_model(X.shape)
        }
        
        # Entraînement
        for name, model in models.items():
            history = model.fit(
                X, y,
                epochs=NN_EPOCHS,
                batch_size=NN_BATCH_SIZE,
                validation_split=NN_VALIDATION_SPLIT,
                callbacks=get_training_callbacks()
            )
            
            models[name] = {
                'model': model,
                'history': history.history
            }
        
        return models
    except Exception as e:
        logger.error(f"Erreur entraînement DL: {str(e)}")
        return None

# ============ UTILITY FUNCTIONS ============
def calculate_regime_stability(regime: int) -> float:
    """
    Calcul de la stabilité du régime de marché
    """
    try:
        # Historique des régimes
        regime_history = cache_system.get('regime_history', [])
        if not regime_history:
            return 0.5
        
        # Calcul de la stabilité
        current_regime_duration = sum(1 for r in reversed(regime_history) if r == regime)
        stability = min(current_regime_duration / len(regime_history), 1.0)
        
        return stability
    except Exception as e:
        logger.error(f"Erreur calcul stabilité régime: {str(e)}")
        return 0.5

def calculate_pattern_confidence(ratios: Dict[str, float], pattern: str) -> float:
    """
    Calcul du niveau de confiance d'un pattern
    """
    try:
        # Ratios idéaux pour chaque pattern
        ideal_ratios = PATTERN_IDEAL_RATIOS[pattern]
        
        # Calcul de la déviation
        deviations = [
            abs(ratios[key] - ideal_ratios[key]) / ideal_ratios[key]
            for key in ideal_ratios
        ]
        
        # Score de confiance
        confidence = 1 - (sum(deviations) / len(deviations))
        
        return max(0.0, min(1.0, confidence))
    except Exception as e:
        logger.error(f"Erreur calcul confiance pattern: {str(e)}")
        return 0.0

# ============ ADDITIONAL CONFIGURATIONS ============
PATTERN_IDEAL_RATIOS = {
    'GARTLEY': {'XA': 1.0, 'AB': 0.618, 'BC': 0.386, 'CD': 1.272},
    'BUTTERFLY': {'XA': 1.0, 'AB': 0.786, 'BC': 0.382, 'CD': 1.618},
    'BAT': {'XA': 1.0, 'AB': 0.382, 'BC': 0.886, 'CD': 2.618},
    'CRAB': {'XA': 1.0, 'AB': 0.382, 'BC': 0.886, 'CD': 3.618},
    'SHARK': {'XA': 1.0, 'AB': 1.13, 'BC': 1.618, 'CD': 0.886}
}

# Configurations pour les callbacks d'entraînement
def get_training_callbacks() -> List[Any]:
    """
    Configuration des callbacks pour l'entraînement des modèles
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{NN_MODELS_PATH}/best_model.h5",
            monitor='val_loss',
            save_best_only=True
        )
    ]

# ============ MAIN EXECUTION ============
if __name__ == '__main__':
    try:
        # Initialisation du système
        if not system_manager.initialize_system():
            logger.error("Échec de l'initialisation du système")
            sys.exit(1)
        
        # Configuration du serveur Flask
        app.config.update(
            JSON_SORT_KEYS=False,
            PROPAGATE_EXCEPTIONS=True,
            MAX_CONTENT_LENGTH=16 * 1024 * 1024,
            TEMPLATES_AUTO_RELOAD=True
        )
        
        # Configuration SSL (optionnel)
        ssl_context = None
        if os.path.exists('cert.pem') and os.path.exists('key.pem'):
            ssl_context = ('cert.pem', 'key.pem')
        
        # Démarrage du serveur
        logger.info("Démarrage du serveur...")
        app.run(
            host='0.0.0.0',
            port=5000,
            threaded=True,
            debug=False,
            ssl_context=ssl_context
        )
        
    except Exception as e:
        logger.critical(f"Erreur fatale lors du démarrage: {str(e)}")
        sys.exit(1)
    finally:
        cleanup() 