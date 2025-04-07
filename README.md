
# 🧠 AI Trading Agent for Solana - Recall.Network Submission

This repository contains an advanced AI-powered trading agent built for the Solana blockchain, created as a submission for the [Recall.Network AI Agent Competition](https://recall.network).

## 🚀 Features

- **LSTM Model Integration**: Predicts market sentiment using a trained neural network.
- **AI-Driven Decision Logic**: Confidence scoring adjusts based on market trends, recent PnL, win rates, and sentiment.
- **Dexscreener + QuickNode**: Real-time data ingestion from on-chain and off-chain sources.
- **SQL Database Integration**: Stores and retrieves historical market data using PostgreSQL + SQLAlchemy.
- **Risk Management**: Dynamically adjusts position size based on current equity.
- **Transaction Execution**: Handles simulated and real trade executions (compatible with Jito and Warp pathways).
- **Modular Architecture**: Designed with clean separation of concerns.

## 📂 Files Included

- `ai_agent.py`: The full agent logic, including AI inference, database models, filters, execution logic, and risk control.
- `lstm_model.keras` *(not included)*: You must train or provide your LSTM model.

## ⚙️ Environment Setup

```bash
git clone https://github.com/yourusername/solana-ai-agent.git
cd solana-ai-agent

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp apikeys.env.example apikeys.env
# Edit your DATABASE_URL, SOLANA_PRIVATE_KEY, RPC URLs etc.
```

## 🧠 Train LSTM Model

If no LSTM model exists, the system will automatically attempt to train and save a new one based on historical price data.

## 📊 SQL Schema

Two tables are used:
- `market_data`: Token OHLCV + indicators
- `market_indicators`: Separate view for SMA, RSI, MACD

## 🛡️ Safety & Strategy

- 1% risk per trade using dynamic equity-based sizing
- Token filters include renounced, burned, mutable, and pool sizing
- Confidence threshold filtering with simulated OpenAI-style adjustment

## ✅ Ready for Submission

This repository is ready to be linked and submitted at [https://recall.network](https://recall.network).

---

Made with 💡 by Favian.
