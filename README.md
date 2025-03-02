# MT5 LLM Trading Bot

An advanced trading bot that combines MetaTrader 5 with Local Large Language Models for automated trading analysis and execution.

## Table of Contents
1. [Overview](#overview)
2. [Detailed Installation Guide](#detailed-installation-guide)
3. [Internal Architecture](#internal-architecture)
4. [Configuration Examples](#configuration-examples)
5. [Technical Indicators Guide](#technical-indicators-guide)
6. [LLM Models Guide](#llm-models-guide)
7. [Version-Specific Optimization](#version-specific-optimization)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Safety Features](#safety-features)
10. [Contributing](#contributing)
11. [License](#license)

## Overview

This project provides three versions of the trading bot:
- **Light**: For low-end laptops and minimal resource usage (~500MB RAM)
- **Medium**: For mid-range computers and balanced performance (~2GB RAM)
- **Heavy**: For high-end computers with maximum analysis capabilities (~4GB+ RAM)

## Detailed Installation Guide

### 1. System Preparation

#### Windows
1. Install Visual Studio Build Tools
```bash
# Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Select: Desktop development with C++
```

2. Install Python 3.8+
```bash
# Download from https://www.python.org/downloads/
# Enable "Add Python to PATH"
```

3. Install Git
```bash
# Download from https://git-scm.com/download/win
```

#### Linux (Ubuntu/Debian)
```bash
# Update system
sudo apt update && sudo apt upgrade

# Install dependencies
sudo apt install python3.8 python3-pip python3-venv git build-essential
```

### 2. MetaTrader 5 Setup

1. Download MT5
```
https://www.metatrader5.com/en/download
```

2. Installation Steps:
- Run installer
- Create demo account (or connect your account)
- Enable AutoTrading
- Enable DLL imports
- Allow algorithmic trading

3. API Setup:
```bash
# Install MT5 Python package
pip install MetaTrader5
```

### 3. LM Studio Configuration

1. Download LM Studio
```
https://lmstudio.ai/
```

2. Model Setup:
- Launch LM Studio
- Go to Models tab
- Download recommended model for your version
- Configure API settings:
  ```json
  {
    "host": "localhost",
    "port": 1234,
    "context_length": 4096
  }
  ```

### 4. Bot Installation

1. Clone Repository
```bash
git clone https://github.com/yourusername/mt5-llm-trading-bot.git
cd mt5-llm-trading-bot
```

2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install Dependencies
```bash
# Light version
cd light
pip install -r requirements.txt

# Medium version
cd ../medium
pip install -r requirements.txt

# Heavy version
cd ../heavy
pip install -r requirements.txt
```

4. Install TA-Lib (Heavy version only)
```bash
# Windows
# Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.24‑cp38‑cp38‑win_amd64.whl

# Linux
sudo apt install ta-lib
pip install TA-Lib
```

## Internal Architecture

### Component Overview
```
MT5 LLM Trading Bot
├── Data Collection Layer
│   ├── MT5 API Interface
│   ├── Market Data Processor
│   └── Historical Data Manager
├── Analysis Layer
│   ├── Technical Indicator Engine
│   ├── Pattern Recognition System
│   └── Risk Management Module
├── LLM Integration Layer
│   ├── Prompt Engineering
│   ├── Response Parser
│   └── Decision Validator
└── Execution Layer
    ├── Order Manager
    ├── Position Tracker
    └── Performance Monitor
```

### Data Flow
1. Market Data Collection
2. Technical Analysis
3. LLM Analysis
4. Decision Making
5. Trade Execution
6. Performance Monitoring

## Configuration Examples

### Light Version
```python
# config.py
TRADING_CONFIG = {
    "symbols": ["EURUSD", "GBPUSD", "USDJPY"],
    "timeframe": "H1",
    "analysis_interval": 300,
    "risk_percent": 2.0,
    "indicators": ["SMA", "RSI"]
}
```

### Medium Version
```python
# config.py
TRADING_CONFIG = {
    "symbols": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"],
    "timeframes": ["H1", "H4"],
    "analysis_interval": 180,
    "risk_percent": 2.0,
    "correlation_threshold": 0.7,
    "indicators": ["SMA", "EMA", "RSI", "MACD", "Stochastic"]
}
```

### Heavy Version
```python
# config.py
TRADING_CONFIG = {
    "symbols": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", 
                "USDCAD", "NZDUSD", "XAUUSD", "XAGUSD"],
    "timeframes": ["M5", "M15", "H1", "H4", "D1"],
    "analysis_interval": 60,
    "risk_percent": 2.0,
    "advanced_features": {
        "pattern_recognition": True,
        "elliott_waves": True,
        "harmonic_patterns": True,
        "portfolio_optimization": True
    }
}
```

## Technical Indicators Guide

### Basic Indicators (Light Version)
- **SMA (Simple Moving Average)**
  - Usage: Trend identification
  - Parameters: 20, 50 periods
  - Implementation: `calculate_sma()`

- **RSI (Relative Strength Index)**
  - Usage: Overbought/Oversold conditions
  - Parameters: 14 periods
  - Implementation: `calculate_rsi()`

### Intermediate Indicators (Medium Version)
[Detailed indicators list...]

### Advanced Indicators (Heavy Version)
[Detailed indicators list...]

## LLM Models Guide

### Light Version Models
[Detailed models list...]

### Medium Version Models
[Detailed models list...]

### Heavy Version Models
[Detailed models list...]

## Version-Specific Optimization

### Light Version Optimization
[Detailed optimization guide...]

### Medium Version Optimization
[Detailed optimization guide...]

### Heavy Version Optimization
[Detailed optimization guide...]

## Troubleshooting Guide

### Common Issues
[Detailed troubleshooting steps...]

### Version-Specific Issues
[Detailed troubleshooting steps...]

## Safety Features
[Detailed safety features...]

## Contributing
[Contributing guidelines...]

## License
MIT License - see LICENSE file for details.

## Disclaimer
This bot is for educational purposes only. Trading carries significant financial risk. 
