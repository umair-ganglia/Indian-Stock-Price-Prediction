"""
Indian Stock Price Prediction - Interactive Web Application
Author: [Your Name]
Date: 2025

A beautiful Streamlit web interface for the stock prediction system.
Run with: streamlit run web_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import warnings
from datetime import datetime, timedelta
import time

# Suppress warnings
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="Indian Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src to path
sys.path.append('src')

try:
    from stock_predictor import IndianStockPredictor
    import config
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced stock list with more Indian stocks
EXTENDED_INDIAN_STOCKS = {
    # Large Cap Stocks
    'RELIANCE.NS': 'Reliance Industries Limited',
    'TCS.NS': 'Tata Consultancy Services Limited',
    'INFY.NS': 'Infosys Limited',
    'HDFCBANK.NS': 'HDFC Bank Limited',
    'ITC.NS': 'ITC Limited',
    'HINDUNILVR.NS': 'Hindustan Unilever Limited',
    'SBIN.NS': 'State Bank of India',
    'BHARTIARTL.NS': 'Bharti Airtel Limited',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank Limited',
    'LT.NS': 'Larsen & Toubro Limited',
    
    # IT Sector
    'WIPRO.NS': 'Wipro Limited',
    'TECHM.NS': 'Tech Mahindra Limited',
    'HCLTECH.NS': 'HCL Technologies Limited',
    'MINDTREE.NS': 'Mindtree Limited',
    
    # Banking & Finance
    'ICICIBANK.NS': 'ICICI Bank Limited',
    'AXISBANK.NS': 'Axis Bank Limited',
    'BAJFINANCE.NS': 'Bajaj Finance Limited',
    'BAJAJFINSV.NS': 'Bajaj Finserv Limited',
    
    # Pharmaceuticals
    'SUNPHARMA.NS': 'Sun Pharmaceutical Industries',
    'DRREDDY.NS': 'Dr. Reddy\'s Laboratories',
    'CIPLA.NS': 'Cipla Limited',
    'BIOCON.NS': 'Biocon Limited',
    
    # Automobiles
    'MARUTI.NS': 'Maruti Suzuki India Limited',
    'TATAMOTORS.NS': 'Tata Motors Limited',
    'M&M.NS': 'Mahindra & Mahindra Limited',
    'BAJAJ-AUTO.NS': 'Bajaj Auto Limited',
    
    # Consumer Goods
    'ASIANPAINT.NS': 'Asian Paints Limited',
    'TITAN.NS': 'Titan Company Limited',
    'NESTLEIND.NS': 'Nestle India Limited',
    'BRITANNIA.NS': 'Britannia Industries Limited',
    
    # Energy & Utilities
    'POWERGRID.NS': 'Power Grid Corporation',
    'NTPC.NS': 'NTPC Limited',
    'ONGC.NS': 'Oil and Natural Gas Corporation',
    'IOC.NS': 'Indian Oil Corporation',
    
    # Metals & Mining
    'TATASTEEL.NS': 'Tata Steel Limited',
    'JSWSTEEL.NS': 'JSW Steel Limited',
    'HINDALCO.NS': 'Hindalco Industries Limited',
    'COALINDIA.NS': 'Coal India Limited',
    
    # Cement
    'ULTRACEMCO.NS': 'UltraTech Cement Limited',
    'SHREECEM.NS': 'Shree Cement Limited',
    'ACC.NS': 'ACC Limited',
    
    # Telecom
    'IDEA.NS': 'Vodafone Idea Limited',
    'INDUSINDBK.NS': 'IndusInd Bank Limited'
}

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Indian Stock Price Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("🎛️ Configuration")
    
    # Stock selection
    selected_stock = st.sidebar.selectbox(
        "Select Stock Symbol:",
        options=list(EXTENDED_INDIAN_STOCKS.keys()),
        format_func=lambda x: f"{x} - {EXTENDED_INDIAN_STOCKS[x]}",
        index=0
    )
    
    # Time period selection
    period_options = {
        '3mo': '3 Months',
        '6mo': '6 Months',
        '1y': '1 Year',
        '2y': '2 Years',
        '5y': '5 Years'
    }
    
    selected_period = st.sidebar.selectbox(
        "Select Time Period:",
        options=list(period_options.keys()),
        format_func=lambda x: period_options[x],
        index=2  # Default to 1 year
    )
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    use_linear = st.sidebar.checkbox("Linear Regression", value=True)
    use_lstm = st.sidebar.checkbox("LSTM Neural Network", value=True)
    use_prophet = st.sidebar.checkbox("Prophet Time Series", value=True)
    
    # Prediction settings
    st.sidebar.subheader("🔮 Prediction Settings")
    prediction_days = st.sidebar.slider("Days to Predict:", 7, 90, 30)
    lstm_epochs = st.sidebar.slider("LSTM Epochs:", 10, 100, 50)
    
    # Custom stock input
    st.sidebar.subheader("📝 Custom Stock")
    custom_stock = st.sidebar.text_input(
        "Enter custom stock symbol (e.g., GOOGL, AAPL):",
        placeholder="SYMBOL.NS for NSE stocks"
    )
    
    if custom_stock:
        selected_stock = custom_stock.upper()
    
    # Start analysis button
    if st.sidebar.button("🚀 Start Analysis", type="primary"):
        st.session_state.run_analysis = True
        st.session_state.selected_stock = selected_stock
        st.session_state.selected_period = selected_period
        st.session_state.models = {
            'linear_regression': use_linear,
            'lstm': use_lstm,
            'prophet': use_prophet
        }
        st.session_state.prediction_days = prediction_days
        st.session_state.lstm_epochs = lstm_epochs
    
    # Main content area
    if hasattr(st.session_state, 'run_analysis') and st.session_state.run_analysis:
        run_stock_analysis()
    else:
        show_welcome_screen()

def show_welcome_screen():
    """Display welcome screen with information"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="success-box">
            <h3>🎯 Welcome to the Indian Stock Price Prediction System!</h3>
            <p>This advanced system uses multiple machine learning models to predict stock prices:</p>
            <ul>
                <li><strong>Linear Regression:</strong> Traditional statistical approach</li>
                <li><strong>LSTM Neural Networks:</strong> Deep learning for time series</li>
                <li><strong>Prophet:</strong> Facebook's time series forecasting</li>
            </ul>
            <p>Select your parameters in the sidebar and click <strong>"Start Analysis"</strong> to begin!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display available stocks
    st.subheader("📊 Available Indian Stocks")
    
    # Create a nice table of stocks
    stock_df = pd.DataFrame([
        {'Symbol': symbol, 'Company Name': name, 'Sector': get_sector(symbol)}
        for symbol, name in list(EXTENDED_INDIAN_STOCKS.items())[:20]  # Show first 20
    ])
    
    st.dataframe(stock_df, use_container_width=True)
    
    if len(EXTENDED_INDIAN_STOCKS) > 20:
        st.info(f"Showing 20 of {len(EXTENDED_INDIAN_STOCKS)} available stocks. More available in the dropdown!")

def get_sector(symbol):
    """Get sector for a stock symbol"""
    sectors = {
        'RELIANCE.NS': 'Energy', 'TCS.NS': 'IT', 'INFY.NS': 'IT',
        'HDFCBANK.NS': 'Banking', 'ITC.NS': 'FMCG', 'HINDUNILVR.NS': 'FMCG',
        'SBIN.NS': 'Banking', 'BHARTIARTL.NS': 'Telecom', 'KOTAKBANK.NS': 'Banking',
        'LT.NS': 'Construction', 'WIPRO.NS': 'IT', 'TECHM.NS': 'IT',
        'MARUTI.NS': 'Automobile', 'ASIANPAINT.NS': 'Paint', 'TITAN.NS': 'Consumer'
    }
    return sectors.get(symbol, 'Others')

def run_stock_analysis():
    """Run the complete stock analysis"""
    
    stock = st.session_state.selected_stock
    period = st.session_state.selected_period
    models = st.session_state.models
    prediction_days = st.session_state.prediction_days
    lstm_epochs = st.session_state.lstm_epochs
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize predictor
        status_text.text("🚀 Initializing predictor...")
        progress_bar.progress(10)
        
        predictor = IndianStockPredictor(symbol=stock, period=period)
        
        # Fetch data
        status_text.text("📡 Fetching stock data...")
        progress_bar.progress(20)
        
        data = predictor.fetch_data()
        if data is None:
            st.error("❌ Failed to fetch data. Please check the stock symbol and try again.")
            return
        
        # Prepare data
        status_text.text("🔧 Preparing data and engineering features...")
        progress_bar.progress(30)
        
        predictor.prepare_data()
        
        # Display stock info
        display_stock_info(data, stock)
        
        # Train models
        results = {}
        current_progress = 40
        
        if models['linear_regression']:
            status_text.text("🤖 Training Linear Regression...")
            progress_bar.progress(current_progress)
            predictor.train_linear_regression()
            current_progress += 20
        
        if models['lstm']:
            status_text.text("🧠 Training LSTM Neural Network...")
            progress_bar.progress(current_progress)
            predictor.train_lstm(epochs=lstm_epochs)
            current_progress += 20
        
        if models['prophet']:
            status_text.text("📊 Training Prophet Model...")
            progress_bar.progress(current_progress)
            predictor.train_prophet()
            current_progress += 20
        
        # Evaluate models
        status_text.text("📊 Evaluating models...")
        progress_bar.progress(90)
        
        results = predictor.evaluate_models()
        
        # Generate predictions
        status_text.text("🔮 Generating future predictions...")
        progress_bar.progress(95)
        
        future_pred, future_dates = predictor.predict_future(days=prediction_days)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Display results
        display_results(predictor, results, future_pred, future_dates)
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.exception(e)

def display_stock_info(data, stock):
    """Display stock information"""
    
    st.subheader(f"📈 {stock} - Stock Information")
    
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Current Price",
            f"₹{current_price:.2f}",
            delta=f"{price_change:+.2f} ({price_change_pct:+.1f}%)"
        )
    
    with col2:
        st.metric("52W High", f"₹{data['High'].max():.2f}")
    
    with col3:
        st.metric("52W Low", f"₹{data['Low'].min():.2f}")
    
    with col4:
        st.metric("Volume", f"{data['Volume'].iloc[-1]:,}")
    
    with col5:
        volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
        st.metric("Volatility", f"{volatility:.1f}%")

def display_results(predictor, results, future_pred, future_dates):
    """Display analysis results"""
    
    # Model comparison
    st.subheader("🏆 Model Performance Comparison")
    
    if results:
        results_df = pd.DataFrame(results).T
        results_df = results_df.round(4)
        
        # Color-code the best performing model
        styled_df = results_df.style.highlight_min(['RMSE', 'MAE', 'MAPE'], color='lightgreen')\
                                  .highlight_max(['R²', 'Directional_Accuracy'], color='lightgreen')
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Best model
        best_model = results_df['RMSE'].idxmin()
        st.success(f"🏆 Best Performing Model: **{best_model.upper()}** (Lowest RMSE: ₹{results_df.loc[best_model, 'RMSE']:.2f})")
    
    # Predictions visualization
    st.subheader("📊 Model Predictions")
    plot_predictions_interactive(predictor)
    
    # Future predictions
    st.subheader("🔮 Future Price Predictions")
    display_future_predictions(future_pred, future_dates, predictor)
    
    # Technical indicators
    st.subheader("📈 Technical Analysis")
    plot_technical_indicators(predictor)
    
    # Trading signals
    st.subheader("🎯 Trading Signals")
    display_trading_signals(predictor)

def plot_predictions_interactive(predictor):
    """Create interactive prediction plots"""
    
    for model_name, pred_data in predictor.predictions.items():
        fig = go.Figure()
        
        # Actual prices
        fig.add_trace(go.Scatter(
            x=pred_data['test_dates'],
            y=pred_data['test_actual'],
            mode='lines',
            name='Actual',
            line=dict(color='#2E86AB', width=3)
        ))
        
        # Predicted prices
        fig.add_trace(go.Scatter(
            x=pred_data['test_dates'],
            y=pred_data['test_pred'],
            mode='lines',
            name='Predicted',
            line=dict(color='#F24236', width=3, dash='dash')
        ))
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((pred_data['test_actual'] - pred_data['test_pred'])**2))
        r2 = 1 - (np.sum((pred_data['test_actual'] - pred_data['test_pred'])**2) / 
                  np.sum((pred_data['test_actual'] - np.mean(pred_data['test_actual']))**2))
        
        fig.update_layout(
            title=f'{model_name.upper()} - Stock Price Prediction (RMSE: ₹{rmse:.2f}, R²: {r2:.4f})',
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_future_predictions(future_pred, future_dates, predictor):
    """Display future predictions"""
    
    if not future_pred:
        st.warning("No future predictions available.")
        return
    
    # Create predictions dataframe
    pred_df = pd.DataFrame(future_pred, index=future_dates)
    
    # Interactive plot
    fig = go.Figure()
    
    for model_name, predictions in future_pred.items():
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name=model_name,
            line=dict(width=3)
        ))
    
    fig.update_layout(
        title='Future Price Predictions',
        xaxis_title='Date',
        yaxis_title='Predicted Price (₹)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Predictions table
    st.subheader("📋 Detailed Predictions")
    
    # Show first 10 days
    display_df = pred_df.head(10).round(2)
    display_df.index = display_df.index.date
    
    st.dataframe(display_df, use_container_width=True)
    
    # Download predictions
    csv = pred_df.to_csv()
    st.download_button(
        label="📥 Download Full Predictions (CSV)",
        data=csv,
        file_name=f"{predictor.symbol}_predictions.csv",
        mime="text/csv"
    )

def plot_technical_indicators(predictor):
    """Plot technical indicators"""
    
    if predictor.processed_data is None:
        return
    
    df = predictor.processed_data.tail(200)  # Last 200 days
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price & Moving Averages', 'RSI', 'MACD'),
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price and Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(width=3)), row=1, col=1)
    
    if 'MA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA_20'], name='MA 20', opacity=0.7), row=1, col=1)
    if 'MA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA_50'], name='MA 50', opacity=0.7), row=1, col=1)
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
    
    # MACD
    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal', line=dict(color='red')), row=3, col=1)
    
    fig.update_layout(height=800, template='plotly_white', showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def display_trading_signals(predictor):
    """Display trading signals"""
    
    signals = predictor.generate_trading_signals()
    recent_signal = signals.iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if recent_signal == 1:
            st.success("🟢 **Current Signal: BUY** 📈")
        elif recent_signal == -1:
            st.error("🔴 **Current Signal: SELL** 📉")
        else:
            st.info("🟡 **Current Signal: HOLD** 📊")
    
    with col2:
        # Signal distribution
        signal_counts = signals.value_counts()
        signal_labels = {1: 'Buy', -1: 'Sell', 0: 'Hold'}
        
        fig = px.pie(
            values=signal_counts.values,
            names=[signal_labels.get(idx, f'Signal {idx}') for idx in signal_counts.index],
            title="Signal Distribution"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Recent signals
        recent_signals = signals.tail(10)
        signal_df = pd.DataFrame({
            'Date': recent_signals.index.date,
            'Signal': recent_signals.map({1: '🟢 Buy', -1: '🔴 Sell', 0: '🟡 Hold'})
        })
        st.dataframe(signal_df, use_container_width=True)

if __name__ == "__main__":
    main()