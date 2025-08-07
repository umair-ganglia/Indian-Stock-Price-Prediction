"""
Real-Time Data Feed Module
Author: [Your Name]
Date: 2025

Real-time market data integration with WebSocket support
"""

import threading
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optional WebSocket support
try:
    import websocket
    import json
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

class RealTimeDataFeed:
    """Real-time market data integration"""
    
    def __init__(self, symbols, update_interval=5):
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.update_interval = update_interval
        self.current_prices = {}
        self.price_history = {}
        self.callbacks = []
        self.running = False
        self.thread = None
        
        # Initialize price history
        for symbol in self.symbols:
            self.price_history[symbol] = []
    
    def add_callback(self, callback):
        """Add callback for price updates"""
        self.callbacks.append(callback)
    
    def start_feed(self, method='simulation'):
        """Start real-time data feed"""
        
        if self.running:
            print("Feed already running!")
            return
        
        self.running = True
        
        if method == 'websocket' and WEBSOCKET_AVAILABLE:
            self.thread = threading.Thread(target=self._websocket_feed)
        else:
            self.thread = threading.Thread(target=self._simulation_feed)
        
        self.thread.daemon = True
        self.thread.start()
        print(f"🔴 Started real-time feed for {len(self.symbols)} symbols")
    
    def stop_feed(self):
        """Stop real-time data feed"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("🟢 Stopped real-time feed")
    
    def _simulation_feed(self):
        """Simulate real-time price updates"""
        
        # Get initial prices
        for symbol in self.symbols:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period="1d")
                if not hist.empty:
                    self.current_prices[symbol] = float(hist['Close'].iloc[-1])
            except:
                self.current_prices[symbol] = 100.0  # Default price
        
        while self.running:
            for symbol in self.symbols:
                if symbol in self.current_prices:
                    # Simulate price movement (random walk)
                    change_pct = np.random.normal(0, 0.005)  # 0.5% std deviation
                    new_price = self.current_prices[symbol] * (1 + change_pct)
                    
                    # Ensure price doesn't go negative
                    new_price = max(new_price, 0.01)
                    
                    self.current_prices[symbol] = new_price
                    
                    # Store in history
                    timestamp = datetime.now()
                    self.price_history[symbol].append({
                        'timestamp': timestamp,
                        'price': new_price,
                        'change': change_pct * 100
                    })
                    
                    # Keep only last 100 records
                    if len(self.price_history[symbol]) > 100:
                        self.price_history[symbol] = self.price_history[symbol][-100:]
                    
                    # Notify callbacks
                    for callback in self.callbacks:
                        callback(symbol, new_price, change_pct * 100)
            
            time.sleep(self.update_interval)
    
    def _websocket_feed(self):
        """WebSocket-based real-time feed (placeholder for actual implementation)"""
        
        # This is a placeholder for actual WebSocket implementation
        # You would connect to a real data provider like Alpha Vantage, IEX, etc.
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                # Parse the message and extract price data
                # This depends on your data provider's format
                symbol = data.get('symbol')
                price = float(data.get('price', 0))
                
                if symbol in self.symbols:
                    self.current_prices[symbol] = price
                    
                    # Notify callbacks
                    for callback in self.callbacks:
                        callback(symbol, price, 0)  # Change calculation depends on data format
                        
            except Exception as e:
                print(f"Error processing message: {e}")
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print("WebSocket connection closed")
            self.running = False
        
        # Example WebSocket URL (replace with actual data provider)
        ws_url = "wss://your-data-provider.com/stream"
        
        if WEBSOCKET_AVAILABLE:
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever()
    
    def get_current_prices(self):
        """Get current prices for all symbols"""
        return self.current_prices.copy()
    
    def get_price_history(self, symbol, minutes=30):
        """Get price history for a symbol"""
        if symbol not in self.price_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            record for record in self.price_history[symbol]
            if record['timestamp'] > cutoff_time
        ]
    
    def get_price_changes(self):
        """Get price changes for all symbols"""
        changes = {}
        for symbol in self.symbols:
            history = self.price_history.get(symbol, [])
            if len(history) >= 2:
                current = history[-1]['price']
                previous = history[-2]['price']
                change_pct = ((current - previous) / previous) * 100
                changes[symbol] = {
                    'current': current,
                    'change_pct': change_pct,
                    'direction': '📈' if change_pct > 0 else '📉' if change_pct < 0 else '➡️'
                }
        return changes

class AlertSystem:
    """Automated alert system for trading signals"""
    
    def __init__(self):
        self.alerts = []
        self.triggered_alerts = []
    
    def add_price_alert(self, symbol, price, condition):
        """Add price-based alert"""
        self.alerts.append({
            'type': 'price',
            'symbol': symbol,
            'price': price,
            'condition': condition,  # 'above' or 'below'
            'created': datetime.now()
        })
    
    def add_change_alert(self, symbol, change_pct, condition):
        """Add percentage change alert"""
        self.alerts.append({
            'type': 'change',
            'symbol': symbol,
            'change_pct': change_pct,
            'condition': condition,  # 'above' or 'below'
            'created': datetime.now()
        })
    
    def check_alerts(self, price_data):
        """Check all alerts against current data"""
        triggered = []
        
        for alert in self.alerts:
            if self._evaluate_alert(alert, price_data):
                triggered.append(alert)
                self.triggered_alerts.append({
                    **alert,
                    'triggered_at': datetime.now()
                })
        
        # Remove triggered alerts
        self.alerts = [alert for alert in self.alerts if alert not in triggered]
        
        return triggered
    
    def _evaluate_alert(self, alert, price_data):
        """Evaluate individual alert"""
        symbol = alert['symbol']
        
        if symbol not in price_data:
            return False
        
        current_data = price_data[symbol]
        
        if alert['type'] == 'price':
            current_price = current_data['current']
            target_price = alert['price']
            
            if alert['condition'] == 'above':
                return current_price >= target_price
            elif alert['condition'] == 'below':
                return current_price <= target_price
        
        elif alert['type'] == 'change':
            current_change = current_data['change_pct']
            target_change = alert['change_pct']
            
            if alert['condition'] == 'above':
                return current_change >= target_change
            elif alert['condition'] == 'below':
                return current_change <= target_change
        
        return False
    
    def get_recent_alerts(self, hours=24):
        """Get recently triggered alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.triggered_alerts
            if alert['triggered_at'] > cutoff_time
        ]

