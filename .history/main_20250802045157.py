#!/usr/bin/env python3
"""
Indian Stock Price Prediction System - Main Execution File
Author: [Your Name]
Date: 2025

This is the main entry point for the stock prediction system.
Run this file to execute the complete pipeline.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from stock_predictor import IndianStockPredictor
import config

def main():
    """Main execution function"""
    print("🚀 Indian Stock Price Prediction System")
    print("=" * 60)
    
    # Get stock symbol from user or use default
    symbol = input(f"Enter stock symbol (default: {config.DEFAULT_STOCK}): ").strip()
    if not symbol:
        symbol = config.DEFAULT_STOCK
    
    # Initialize predictor
    predictor = IndianStockPredictor(
        symbol=symbol, 
        period=config.DEFAULT_PERIOD
    )
    
    try:
        # Execute the complete pipeline
        print(f"\n📊 Analyzing {symbol}...")
        
        # Fetch and prepare data
        data = predictor.fetch_data()
        if data is None:
            print("❌ Failed to fetch data. Please check your internet connection.")
            return
        
        predictor.prepare_data()
        
        # Display basic stock info
        print(f"\n📈 Stock Information:")
        print(f"   Current Price: ₹{data['Close'].iloc[-1]:.2f}")
        print(f"   52-week High: ₹{data['High'].max():.2f}")
        print(f"   52-week Low: ₹{data['Low'].min():.2f}")
        print(f"   Trading Volume: {data['Volume'].iloc[-1]:,}")
        
        # Train models
        print(f"\n🤖 Training Models...")
        print("-" * 40)
        
        # Linear Regression
        predictor.train_linear_regression()
        
        # LSTM
        predictor.train_lstm(epochs=config.LSTM_EPOCHS)
        
        # Prophet
        predictor.train_prophet()
        
        # Evaluate models
        print(f"\n📊 Model Evaluation:")
        results = predictor.evaluate_models()
        
        # Find best model
        best_model = min(results.keys(), key=lambda x: results[x]['RMSE'])
        print(f"\n🏆 Best Model: {best_model.upper()} (Lowest RMSE: {results[best_model]['RMSE']:.2f})")
        
        # Generate predictions
        print(f"\n🔮 Future Predictions...")
        future_pred, future_dates = predictor.predict_future(days=config.PREDICTION_DAYS)
        
        # Generate trading signals
        signals = predictor.generate_trading_signals()
        recent_signal = signals.iloc[-1]
        
        if recent_signal == 1:
            signal_text = "BUY 📈"
            signal_color = "🟢"
        elif recent_signal == -1:
            signal_text = "SELL 📉"
            signal_color = "🔴"
        else:
            signal_text = "HOLD 📊"
            signal_color = "🟡"
        
        print(f"\n{signal_color} Current Trading Signal: {signal_text}")
        
        # Plot results
        print(f"\n📈 Generating visualizations...")
        predictor.plot_predictions()
        
        # Save results
        if config.SAVE_RESULTS:
            predictor.save_results()
            print(f"💾 Results saved to 'results/' directory")
        
        print(f"\n✅ Analysis Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        print("Please check your data connection and try again.")

if __name__ == "__main__":
    main()