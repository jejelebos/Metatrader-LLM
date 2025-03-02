# MT5 LLM Trading Bot - Light Version 🚀

## Description
Lightweight version of the MT5 trading bot, optimized for resource-limited machines. This version offers essential trading features while minimizing system resource usage.

## ⚠️ Warning
**CAUTION: High-Risk Trading**
- Forex trading involves significant risk of loss
- This bot is a decision support tool, not an infallible automated trading system
- The author is not responsible for any potential losses related to the use of this bot
- Always test in a demo account before using real money
- Past performance does not guarantee future results

## Features 🔍
- Single timeframe analysis (H1)
- Limited currency pairs (EURUSD, GBPUSD, USDJPY)
- Basic optimized technical indicators
- Efficient memory management
- Light system monitoring
- Automatic cache cleanup

## System Requirements 💻
- **CPU**: 2 cores minimum
- **RAM**: 2 GB minimum
- **Disk Space**: 1 GB minimum
- **OS**: Windows 10/11
- **Python**: 3.8 or higher
- **MetaTrader 5**: Latest version
- **Internet**: Stable, 5 Mbps minimum

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
- Fill in MT5 connection information
- Configure LLM API
- Adjust trading parameters as needed

## Configuration 🛠️

### Main Parameters
```python
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAME = mt5.TIMEFRAME_H1
MAX_POSITIONS = 3
RISK_PER_TRADE = 0.01  # 1% per trade
```

### Resource Limits
```python
MAX_CACHE_SIZE = 100 * 1024 * 1024  # 100MB
CLEANUP_INTERVAL = 300  # 5 minutes
```

## Usage 📊

1. **Start the bot**
```bash
python light_server.py
```

2. **API Endpoints**
- `/status`: System status
- `/analyze`: Market analysis and trading decisions

3. **Monitoring**
- Logs in `light_trader.log`
- Automatic memory monitoring
- Periodic cache cleanup

## Security 🔒
- Secure credential management
- Memory leak protection
- Input data validation
- Robust error handling

## Limitations ⚡
- Single timeframe (H1)
- Maximum 3 currency pairs
- Basic technical indicators only
- No multi-timeframe analysis
- Limited ML/DL capabilities

## Maintenance 🔧
- Regular log cleanup
- Monitor memory usage
- Keep MT5 and Python updated
- Backup configurations

## Troubleshooting 🚨

### Common Issues
1. **MT5 Connection Error**
   - Check credentials
   - Confirm MT5 is running
   - Verify internet connection

2. **High Memory Usage**
   - Restart bot
   - Check for memory leaks
   - Adjust MAX_CACHE_SIZE

3. **Analysis Errors**
   - Check logs
   - Confirm LLM availability
   - Validate market data

## Support 💬
- Create GitHub issue
- Consult MT5 documentation
- Check for regular updates

## License 📝
MIT License - See LICENSE file for details

## Author ✨
Jeremy

## Contributing 🤝
Contributions are welcome! See CONTRIBUTING.md for details.

---

**Note**: This bot is provided as-is, without warranty. Use at your own risk and always test with a demo account first. 