#!/usr/bin/env python3
"""
Test file to verify the stock prediction system setup
Run this file to check if everything is working correctly
"""

import sys
import os

print("🧪 Testing Stock Price Prediction System Setup")
print("=" * 50)

# Test 1: Import basic libraries
print("📦 Testing library imports...")
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("   ✅ pandas, numpy, matplotlib, seaborn")
except ImportError as e:
    print(f"   ❌ Error importing basic libraries: {e}")
    sys.exit(1)

# Test 2: Import machine learning libraries
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    print("   ✅ scikit-learn")
except ImportError as e:
    print(f"   ❌ Error importing scikit-learn: {e}")

# Test 3: Import TensorFlow
try:
    import tensorflow as tf
    print(f"   ✅ TensorFlow {tf.__version__}")
except ImportError as e:
    print(f"   ❌ Error importing TensorFlow: {e}")

# Test 4: Import financial data library
try:
    import yfinance as yf
    print("   ✅ yfinance")
except ImportError as e:
    print(f"   ❌ Error importing yfinance: {e}")

# Test 5: Import Prophet
try:
    from prophet import Prophet
    print("   ✅ Prophet")
except ImportError as e:
    print(f"   ❌ Error importing Prophet: {e}")

# Test 6: Test data fetching
print("\n📡 Testing data fetching...")
try:
    stock = yf.Ticker("RELIANCE.NS")
    data = stock.history(period="5d")  # Get last 5 days
    if not data.empty:
        print(f"   ✅ Successfully fetched RELIANCE.NS data")
        print(f"   📊 Latest price: ₹{data['Close'].iloc[-1]:.2f}")
        print(f"   📅 Data points: {len(data)}")
    else:
        print("   ⚠️  No data returned (check internet connection)")
except Exception as e:
    print(f"   ❌ Error fetching data: {e}")

# Test 7: Check folder structure
print("\n📁 Checking folder structure...")
required_folders = [
    'src',
    'data',
    'data/raw',
    'data/processed',
    'models',
    'models/saved_models',
    'results',
    'results/plots',
    'results/predictions'
]

for folder in required_folders:
    if os.path.exists(folder):
        print(f"   ✅ {folder}")
    else:
        print(f"   ❌ Missing: {folder}")

# Test 8: Check required files
print("\n📄 Checking required files...")
required_files = [
    'requirements.txt',
    'config.py',
    'main.py',
    'src/__init__.py',
    'src/stock_predictor.py'
]

for file in required_files:
    if os.path.exists(file):
        print(f"   ✅ {file}")
    else:
        print(f"   ❌ Missing: {file}")

# Test 9: Test importing the stock predictor
print("\n🤖 Testing stock predictor import...")
try:
    sys.path.append('src')
    from stock_predictor import IndianStockPredictor
    print("   ✅ IndianStockPredictor class imported successfully")
    
    # Try creating an instance
    predictor = IndianStockPredictor(symbol="RELIANCE.NS", period="5d")
    print("   ✅ IndianStockPredictor instance created")
    
except Exception as e:
    print(f"   ❌ Error importing stock predictor: {e}")

print("\n" + "=" * 50)
print("🎉 Setup test complete!")
print("\nIf all tests passed, you can run:")
print("   python main.py")
print("\nOr try a quick test:")
print("   python -c \"from src.stock_predictor import quick_predict; quick_predict('RELIANCE.NS', '5d', ['linear_regression'])\"")