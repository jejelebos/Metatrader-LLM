# MT5 LLM Trading Bot - Medium Version 🚀

## Description
Intermediate version of the MT5 trading bot, optimized for mid-range machines. This version offers a balance between advanced features and resource consumption.

## ⚠️ Warning
**CAUTION: High-Risk Trading**
- Forex trading involves significant risk of loss
- This bot is a decision support tool, not an infallible automated trading system
- The author is not responsible for any potential losses related to the use of this bot
- Always test in a demo account before using real money
- Past performance does not guarantee future results
- Code modifications may lead to unexpected behaviors

## Features 🔍
- Multi-timeframe analysis (H1, H4)
- Support for 7 major pairs
- Extended technical indicators
- Optimized Machine Learning
- Advanced position management
- Enhanced risk analysis
- Complete system monitoring

## System Requirements 💻
- **CPU**: 4 cores recommended
- **RAM**: 4 GB minimum
- **Disk Space**: 2 GB minimum
- **OS**: Windows 10/11
- **Python**: 3.8 or higher
- **MetaTrader 5**: Latest version
- **GPU**: Optional, for enhanced ML
- **Internet**: Stable, 10 Mbps minimum

## Installation 🔧

1. **Prepare Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configuration**
- Copy `.env.example` to `.env`
- Configure MT5 credentials
- Configure LLM API
- Adjust trading parameters
- Configure notifications (optional)

## Configuration 🛠️

### Trading Parameters
```python
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
TIMEFRAMES = {
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4
}
MAX_POSITIONS = 5
RISK_PER_TRADE = 0.02  # 2% per trade
```

### ML/DL Parameters
```python
MODEL_CONFIG = {
    "lstm_units": 50,
    "dropout": 0.2,
    "epochs": 100,
    "batch_size": 32
}
```

## Advanced Features 📊

### Technical Analysis
- Multiple moving averages
- RSI, MACD, Stochastic
- Bollinger Bands
- CCI and MFI
- ATR for sizing

### Machine Learning
- LSTM for prediction
- Trend classification
- Anomaly detection
- Portfolio optimization

### Risk Management
- Dynamic VaR
- Trailing stops
- Pair correlation
- Drawdown protection

## Usage 📈

1. **Start the bot**
```bash
python medium_server.py
```

2. **API Endpoints**
- `/health`: System status
- `/analyze`: Market analysis
- `/manage`: Portfolio management

3. **Monitoring**
- Basic web interface
- Detailed logs
- Performance metrics
- System alerts

## Performance Optimization 🚀

### Memory Management
- Smart caching
- Periodic cleanup
- Resource monitoring
- Data optimization

### ML/DL
- Batch processing
- Optimized prediction
- Model updates
- Feature engineering

## Security 🔒
- Data validation
- Rate limiting
- Credential protection
- Secure logging
- Automatic backup

## Maintenance 🔧

### Daily Tasks
- Check logs
- Monitor performance
- Validate positions
- Backup data

### Weekly Tasks
- Update models
- Clean data
- Optimize cache
- Strategy verification

## Troubleshooting 🚨

### Common Issues
1. **ML Errors**
   - Check input data
   - Retrain models
   - Adjust parameters

2. **Performance Issues**
   - Clean cache
   - Optimize queries
   - Reduce timeframes

3. **Trading Errors**
   - Check connection
   - Validate orders
   - Control risks

## Customization 🎨
- Add indicators
- Modify strategies
- Configure alerts
- Optimize parameters

## Support 💬
- Detailed documentation
- GitHub issues
- Regular updates
- Trading community

## License 📝
MIT License - See LICENSE file for details

## Author ✨
Jeremy

## Contributing 🤝
Contributions are welcome! See CONTRIBUTING.md for details.

---

**Note**: This medium version offers a good balance between features and performance. Adjust parameters according to your needs and system capabilities. 