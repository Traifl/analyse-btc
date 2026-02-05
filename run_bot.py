"""Bot launcher — trains the model on historical data, then starts the trading engine."""

import os
import sys
import logging
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib

from config import load_api_keys, SYMBOL, FORECAST_HORIZON, OUTPUT_DIR
from src.data_collector import create_client, get_historical_klines
from src.preprocessing import clean_data, add_technical_features
from src.regression import prepare_features, train_model, evaluate_model
from bot.state import BotState
from bot.trading_engine import TradingEngine

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "bot.log")),
    ],
)
logger = logging.getLogger(__name__)


def train_and_save_model():
    """Train XGBoost on historical data and save to models/ directory."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Training model on historical data")
    logger.info("=" * 60)

    keys = load_api_keys()
    client = create_client(keys)

    logger.info(f"Fetching 1 year of 1H data for {SYMBOL}...")
    df = get_historical_klines(symbol=SYMBOL, interval="1h", days=365, client=client)
    logger.info(f"  {len(df)} candles fetched")

    df = clean_data(df)
    df = add_technical_features(df)
    df = df.dropna()
    logger.info(f"  {len(df)} rows after preprocessing")

    X, y = prepare_features(df, horizon=FORECAST_HORIZON)
    logger.info(f"  Features: {X.shape[1]}, Samples: {len(X)}")

    model, X_test, y_test, _, _ = train_model(X, y, model_type="xgboost")
    metrics = evaluate_model(model, X_test, y_test)
    logger.info(f"  XGBoost — RMSE: ${metrics['rmse']:,.2f}, MAE: ${metrics['mae']:,.2f}, R²: {metrics['r2']:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"  Model saved to {model_path}")

    return metrics


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Phase 1: Train model
    model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    if not os.path.exists(model_path):
        train_and_save_model()
    else:
        logger.info(f"Using existing model at {model_path}")

    # Phase 2: Start trading engine
    logger.info("=" * 60)
    logger.info("PHASE 2: Starting trading engine (paper mode)")
    logger.info("=" * 60)

    state = BotState()
    state.reset()

    engine = TradingEngine(mode="paper", state=state)

    # Graceful shutdown
    def shutdown(signum, frame):
        logger.info("Shutdown signal received")
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    engine.start()

    logger.info("")
    logger.info("=" * 60)
    logger.info("Bot is running in PAPER mode!")
    logger.info(f"Dashboard: run 'streamlit run dashboard/app.py' in another terminal")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    # Keep main thread alive
    try:
        while engine.is_running:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        engine.stop()


if __name__ == "__main__":
    main()
