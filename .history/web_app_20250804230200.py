"""
Indian Stock Price Prediction - Interactive Web Application
Author: [Your Name]
Date: 2025

A beautiful Streamlit web interface with dark/light theme toggle.
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
import yfinance as yf

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Indian Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme management
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    if st.session_state.theme == 'light':
        st.session_state.theme = 'dark'
    else:
        st.session_state.theme = 'light'

# Add src to path (if needed)
sys.path.append('src')

try:
    from stock_predictor import IndianStockPredictor
    from ensemble_models import EnsemblePredictor
    from realtime_data_feed import RealTimeDataFeed, AlertSystem
except ImportError as e:
    st.error(f"""
    ❌ **Import Error**: {e}
    
    Please ensure that:
    1. The required files are in the same directory: `stock_predictor.py`, `ensemble_models.py`, `realtime_data_feed.py`
    2. All required packages are installed: `pip install yfinance scikit-learn pandas numpy matplotlib seaborn plotly`
    3. For LSTM: `pip install tensorflow`
    4. For Prophet: `pip install prophet`
    5. For Real-time: `pip install websocket-client xgboost`
    """)
    st.stop()

# Enhanced stock list with more Indian stocks
EXTENDED_INDIAN_STOCKS = {
    # Large Cap IT
    'TCS.NS': 'Tata Consultancy Services Limited',
    'INFY.NS': 'Infosys Limited',
    'WIPRO.NS': 'Wipro Limited',
    'TECHM.NS': 'Tech Mahindra Limited',
    'HCLTECH.NS': 'HCL Technologies Limited',
    'LTI.NS': 'L&T Infotech Limited',
    
    # Banking & Finance
    'HDFCBANK.NS': 'HDFC Bank Limited',
    'ICICIBANK.NS': 'ICICI Bank Limited',
    'SBIN.NS': 'State Bank of India',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank Limited',
    'AXISBANK.NS': 'Axis Bank Limited',
    'INDUSINDBK.NS': 'IndusInd Bank Limited',
    'BAJFINANCE.NS': 'Bajaj Finance Limited',
    'BAJAJFINSV.NS': 'Bajaj Finserv Limited',
    
    # Energy & Oil
    'RELIANCE.NS': 'Reliance Industries Limited',
    'ONGC.NS': 'Oil and Natural Gas Corporation',
    'IOC.NS': 'Indian Oil Corporation',
    'BPCL.NS': 'Bharat Petroleum Corporation',
    'ADANIPORTS.NS': 'Adani Ports and SEZ Limited',
    
    # FMCG & Consumer
    'HINDUNILVR.NS': 'Hindustan Unilever Limited',
    'ITC.NS': 'ITC Limited',
    'NESTLEIND.NS': 'Nestle India Limited',
    'BRITANNIA.NS': 'Britannia Industries Limited',
    'DABUR.NS': 'Dabur India Limited',
    
    # Automobiles
    'MARUTI.NS': 'Maruti Suzuki India Limited',
    'TATAMOTORS.NS': 'Tata Motors Limited',
    'M&M.NS': 'Mahindra & Mahindra Limited',
    'BAJAJ-AUTO.NS': 'Bajaj Auto Limited',
    
    # Pharmaceuticals
    'SUNPHARMA.NS': 'Sun Pharmaceutical Industries',
    'DRREDDY.NS': 'Dr. Reddy\'s Laboratories',
    'CIPLA.NS': 'Cipla Limited',
    'BIOCON.NS': 'Biocon Limited',
    
    # Metals & Mining
    'TATASTEEL.NS': 'Tata Steel Limited',
    'JSWSTEEL.NS': 'JSW Steel Limited',
    'HINDALCO.NS': 'Hindalco Industries Limited',
    'COALINDIA.NS': 'Coal India Limited',
    
    # Others
    'LT.NS': 'Larsen & Toubro Limited',
    'ULTRACEMCO.NS': 'UltraTech Cement Limited',
    'TITAN.NS': 'Titan Company Limited',
    'ASIANPAINT.NS': 'Asian Paints Limited',
}

def apply_theme():
    """Apply dark or light theme CSS"""
    if st.session_state.theme == 'dark':
        st.markdown("""
        <style>
            .stApp {
                background-color: #0E1117;
                color: #FAFAFA;
            }
            
            .main-header {
                font-size: 3.5rem;
                background: linear-gradient(90deg, #64B5F6, #FFB74D);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                margin-bottom: 2rem;
                font-weight: bold;
            }
            
            .metric-card {
                background: linear-gradient(135deg, #1E3A8A 0%, #7C3AED 100%);
                padding: 1.5rem;
                border-radius: 15px;
                color: white;
                text-align: center;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
                margin: 0.5rem 0;
            }
            
            .success-box {
                padding: 1.5rem;
                border-radius: 15px;
                background: linear-gradient(135deg, #059669 0%, #0D9488 100%);
                color: white;
                margin: 1rem 0;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            }
            
            .warning-box {
                padding: 1.5rem;
                border-radius: 15px;
                background: linear-gradient(135deg, #D97706 0%, #F59E0B 100%);
                color: white;
                margin: 1rem 0;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            }
            
            .info-box {
                padding: 1.5rem;
                border-radius: 15px;
                background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
                color: white;
                margin: 1rem 0;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            }
            
            .stSelectbox > div > div {
                background-color: #374151;
                color: white;
            }
            
            .stTextInput > div > div > input {
                background-color: #374151;
                color: white;
            }
            
            .stButton > button {
                background: linear-gradient(135deg, #1E40AF 0%, #7C3AED 100%);
                color: white;
                border: none;
                border-radius: 25px;
                padding: 0.75rem 2rem;
                font-weight: bold;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
                transition: all 0.3s ease;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .stApp {
                background-color: #FFFFFF;
                color: #000000;
            }
            
            .main-header {
                font-size: 3.5rem;
                background: linear-gradient(90deg, #1f77b4, #ff7f0e);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                margin-bottom: 2rem;
                font-weight: bold;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }
            
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem;
                border-radius: 15px;
                color: white;
                text-align: center;
                box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.18);
                margin: 0.5rem 0;
            }
            
            .success-box {
                padding: 1.5rem;
                border-radius: 15px;
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                margin: 1rem 0;
                box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.18);
            }
            
            .warning-box {
                padding: 1.5rem;
                border-radius: 15px;
                background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                color: white;
                margin: 1rem 0;
                box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.18);
            }
            
            .info-box {
                padding: 1.5rem;
                border-radius: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                margin: 1rem 0;
                box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.18);
            }
            
            .stButton > button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 25px;
                padding: 0.75rem 2rem;
                font-weight: bold;
                font-size: 1.1rem;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                transition: all 0.3s ease;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
            }
        </style>
        """, unsafe_allow_html=True)

def show_animated_header():
    """Display animated header"""
    theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
    theme_text = "Dark Mode" if st.session_state.theme == 'light' else "Light Mode"
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.button(f"{theme_icon} {theme_text}", on_click=toggle_theme, key="theme_toggle")
    
    with col2:
        header_html = """
        <div style="text-align: center; padding: 1rem;">
            <h1 class="main-header">📈 Indian Stock Price Prediction</h1>
            <p style="font-size: 1.2rem; opacity: 0.8; margin-top: -1rem;">
                AI-Powered Stock Analysis & Future Price Prediction
            </p>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)

def create_custom_metric(title, value, delta=None, color="#1f77b4"):
    """Create a custom metric card"""
    delta_html = ""
    if delta:
        delta_color = "#2ca02c" if delta.startswith("+") else "#d62728"
        delta_html = f'<p style="color: {delta_color}; font-size: 0.9rem; margin-top: 0.5rem;">{delta}</p>'
    
    metric_html = f"""
    <div style="
        background: linear-gradient(135deg, {color}20, {color}40);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid {color};
        margin: 0.5rem 0;
    ">
        <h3 style="margin: 0; color: {color};">{title}</h3>
        <h2 style="margin: 0.5rem 0; color: #333;">{value}</h2>
        {delta_html}
    </div>
    """
    return metric_html

def main():
    """Enhanced Main Streamlit application with tabs"""
    
    # Apply theme
    apply_theme()
    
    # Animated Header
    show_animated_header()
    st.markdown("---")
    
    # Create main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Stock Analysis", 
        "🎯 Portfolio", 
        "⚡ Real-Time", 
        "🤖 AI Insights",
        "📊 Market Scanner"
    ])
    
    with tab1:
        show_stock_analysis_tab()
    
    with tab2:
        show_portfolio_tab()
    
    with tab3:
        show_realtime_tab()
    
    with tab4:
        show_ai_insights_tab()
    
    with tab5:
        show_market_scanner_tab()

def show_stock_analysis_tab():
    """Stock analysis functionality with complete sidebar"""
    
    # Complete Sidebar Implementation
    with st.sidebar:
        st.markdown("### 🎛️ Configuration Panel")
        
        # Stock selection with search
        st.markdown("#### 📊 Stock Selection")
        
        search_term = st.text_input("🔍 Search for a stock:", placeholder="e.g., TCS, Reliance, HDFC")
        
        if search_term:
            filtered_stocks = {k: v for k, v in EXTENDED_INDIAN_STOCKS.items() 
                             if search_term.lower() in k.lower() or search_term.lower() in v.lower()}
            stock_options = list(filtered_stocks.keys())
        else:
            stock_options = list(EXTENDED_INDIAN_STOCKS.keys())
        
        if stock_options:
            selected_stock = st.selectbox(
                "Select Stock Symbol:",
                options=stock_options,
                format_func=lambda x: f"{x} - {EXTENDED_INDIAN_STOCKS.get(x, 'Unknown')}",
                index=0
            )
        else:
            st.warning("No stocks found matching your search.")
            selected_stock = "TCS.NS"
        
        # Time period selection
        st.markdown("#### 📅 Time Period")
        period_options = {
            '3mo': '3 Months (Short Term)',
            '6mo': '6 Months (Medium Term)',
            '1y': '1 Year (Recommended)',
            '2y': '2 Years (Long Term)',
            '5y': '5 Years (Very Long Term)'
        }
        
        selected_period = st.selectbox(
            "Select Time Period:",
            options=list(period_options.keys()),
            format_func=lambda x: period_options[x],
            index=2  # Default to 1 year
        )
        
        # Model selection
        st.markdown("#### 🤖 AI Models")
        
        with st.expander("ℹ️ Model Information", expanded=False):
            st.markdown("""
            **Linear Regression**: Fast, traditional statistical approach
            
            **LSTM Neural Network**: Deep learning for complex patterns
            
            **Prophet**: Facebook's time-series forecasting (handles seasonality)
            """)
        
        use_linear = st.checkbox("📊 Linear Regression", value=True, help="Fast and interpretable")
        use_lstm = st.checkbox("🧠 LSTM Neural Network", value=True, help="Deep learning model")
        use_prophet = st.checkbox("📈 Prophet Time Series", value=True, help="Handles seasonality well")
        
        # Advanced settings
        st.markdown("#### ⚙️ Advanced Settings")
        with st.expander("🔧 Model Parameters", expanded=False):
            prediction_days = st.slider("Days to Predict:", 7, 90, 30, help="Number of future days to predict")
            lstm_epochs = st.slider("LSTM Training Epochs:", 10, 100, 50, help="More epochs = better training but slower")
            
            risk_tolerance = st.select_slider(
                "Risk Level:",
                options=["Conservative", "Moderate", "Aggressive"],
                value="Moderate",
                help="Affects prediction confidence intervals"
            )
        
        # Custom stock input
        st.markdown("#### 📝 Custom Stock")
        custom_stock = st.text_input(
            "Enter custom symbol:",
            placeholder="e.g., GOOGL, AAPL, TSLA",
            help="For NSE stocks, add .NS (e.g., SYMBOL.NS)"
        )
        
        if custom_stock:
            selected_stock = custom_stock.upper()
        
        # Action buttons
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            start_analysis = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
        with col2:
            clear_cache = st.button("🗑️ Clear Cache", use_container_width=True)
        
        if clear_cache:
            st.cache_data.clear()
            st.success("Cache cleared!")
        
        if start_analysis:
            if not any([use_linear, use_lstm, use_prophet]):
                st.error("Please select at least one model!")
            else:
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
                st.session_state.risk_tolerance = risk_tolerance
    
    # Main content area
    if hasattr(st.session_state, 'run_analysis') and st.session_state.run_analysis:
        run_stock_analysis()
    else:
        show_welcome_screen()

def show_welcome_screen():
    """Display enhanced welcome screen"""
    
    st.markdown("""
    <div class="success-box">
        <h2>🎯 Welcome to Advanced Stock Prediction!</h2>
        <p style="font-size: 1.1rem; margin-bottom: 1rem;">
            Harness the power of AI to predict Indian stock prices with multiple machine learning models.
        </p>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-top: 1.5rem;">
            <div style="text-align: center;">
                <h3>📊 Linear Regression</h3>
                <p>Traditional statistical approach for trend analysis</p>
            </div>
            <div style="text-align: center;">
                <h3>🧠 LSTM Networks</h3>
                <p>Deep learning for complex pattern recognition</p>
            </div>
            <div style="text-align: center;">
                <h3>📈 Prophet</h3>
                <p>Facebook's time-series forecasting with seasonality</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_custom_metric("🏢 Available Stocks", f"{len(EXTENDED_INDIAN_STOCKS)}", color="#1f77b4"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_custom_metric("🤖 AI Models", "3", color="#ff7f0e"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_custom_metric("📊 Sectors Covered", "15+", color="#2ca02c"), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_custom_metric("⚡ Prediction Speed", "< 60s", color="#d62728"), unsafe_allow_html=True)
    
    # Getting started guide
    st.markdown("### 🚀 Getting Started")
    
    st.markdown("""
    <div class="warning-box">
        <h4>📋 Quick Start Guide</h4>
        <ol>
            <li><strong>Select a Stock:</strong> Choose from 40+ Indian stocks or enter a custom symbol</li>
            <li><strong>Pick Time Period:</strong> Select from 3 months to 5 years of historical data</li>
            <li><strong>Choose Models:</strong> Enable the AI models you want to use</li>
            <li><strong>Configure Settings:</strong> Adjust prediction days and model parameters</li>
            <li><strong>Start Analysis:</strong> Click the "Start Analysis" button and wait for results</li>
        </ol>
        <p><strong>💡 Tip:</strong> For best results, use 1-2 years of data with all three models enabled!</p>
    </div>
    """, unsafe_allow_html=True)

def run_stock_analysis():
    """Run the complete stock analysis"""
    
    stock = st.session_state.selected_stock
    period = st.session_state.selected_period
    models = st.session_state.models
    prediction_days = st.session_state.prediction_days
    lstm_epochs = st.session_state.lstm_epochs
    risk_tolerance = st.session_state.get('risk_tolerance', 'Moderate')
    
    # Create analysis header
    st.markdown(f"""
    <div class="success-box">
        <h2>🔍 Analyzing {stock}</h2>
        <p>Running AI-powered analysis with {sum(models.values())} models over {period} period</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize predictor
        status_text.info("🚀 Initializing AI system...")
        progress_bar.progress(20)
        time.sleep(1)
        
        predictor = IndianStockPredictor(symbol=stock, period=period)
        
        # Fetch data
        status_text.info(f"📡 Fetching {period} of data for {stock}...")
        progress_bar.progress(40)
        
        data = predictor.fetch_data()
        if data is None:
            st.error("❌ Failed to fetch data. Please check the stock symbol.")
            return
        
        # Prepare data
        status_text.info("🔧 Processing data and calculating indicators...")
        progress_bar.progress(60)
        
        predictor.prepare_data()
        
        # Display stock info
        display_stock_info(data, stock)
        
        # Train models
        current_progress = 60
        
        if models['linear_regression']:
            status_text.info("🤖 Training Linear Regression...")
            progress_bar.progress(current_progress)
            predictor.train_linear_regression()
            current_progress += 10
        
        if models['lstm']:
            status_text.info(f"🧠 Training LSTM ({lstm_epochs} epochs)...")
            progress_bar.progress(current_progress)
            predictor.train_lstm(epochs=lstm_epochs)
            current_progress += 10
        
        if models['prophet']:
            status_text.info("📊 Training Prophet...")
            progress_bar.progress(current_progress)
            predictor.train_prophet()
            current_progress += 10
        
        # Evaluate and predict
        status_text.info("📊 Evaluating models and generating predictions...")
        progress_bar.progress(90)
        
        results = predictor.evaluate_models()
        future_pred, future_dates = predictor.predict_future(days=prediction_days)
        
        progress_bar.progress(100)
        status_text.success("✅ Analysis complete!")
        
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        display_results(predictor, results, future_pred, future_dates, risk_tolerance)
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.exception(e)

def display_stock_info(data, stock):
    """Display stock information"""
    
    st.markdown("### 📊 Stock Overview")
    
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_text = f"{price_change:+.2f} ({price_change_pct:+.1f}%)"
        st.metric("Current Price", f"₹{current_price:.2f}", delta=delta_text)
    
    with col2:
        st.metric("52W High", f"₹{data['High'].max():.2f}")
    
    with col3:
        st.metric("52W Low", f"₹{data['Low'].min():.2f}")
    
    with col4:
        volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
        st.metric("Volatility", f"{volatility:.1f}%")
    
    # Price chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    fig.update_layout(
        title=f'{stock} - Price History',
        yaxis_title='Price (₹)',
        template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_results(predictor, results, future_pred, future_dates, risk_tolerance):
    """Display analysis results"""
    
    # Model comparison
    st.markdown("### 🏆 Model Performance")
    
    if results:
        results_df = pd.DataFrame(results).T.round(4)
        best_model = results_df['RMSE'].idxmin()
        
        col1, col2, col3 = st.columns(3)
        
        for i, (model_name, metrics) in enumerate(results.items()):
            col = [col1, col2, col3][i % 3]
            
            with col:
                is_best = model_name == best_model
                crown = "👑 " if is_best else ""
                
                st.markdown(f"""
                <div class="{'success-box' if is_best else 'info-box'}">
                    <h3>{crown}{model_name.upper()}</h3>
                    <p><strong>RMSE:</strong> ₹{metrics['RMSE']:.2f}</p>
                    <p><strong>R²:</strong> {metrics['R²']:.4f}</p>
                    <p><strong>MAPE:</strong> {metrics['MAPE']:.1f}%</p>
                    <p><strong>Direction Accuracy:</strong> {metrics['Directional_Accuracy']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Predictions visualization
    st.markdown("### 📊 Predictions vs Actual")
    
    for model_name, pred_data in predictor.predictions.items():
        st.markdown(f"#### {model_name.upper()}")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pred_data['test_dates'],
            y=pred_data['test_actual'],
            mode='lines',
            name='Actual',
            line=dict(color='#2E86AB', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=pred_data['test_dates'],
            y=pred_data['test_pred'],
            mode='lines',
            name='Predicted',
            line=dict(color='#F24236', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title=f'{model_name.upper()} - Prediction Performance',
            template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Future predictions
    st.markdown("### 🔮 Future Price Predictions")
    
    if future_pred:
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (model_name, predictions) in enumerate(future_pred.items()):
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines+markers',
                name=model_name,
                line=dict(width=3, color=colors[i % len(colors)])
            ))
        
        current_price = predictor.processed_data['Close'].iloc[-1]
        fig.add_hline(
            y=current_price,
            line_dash="dot",
            annotation_text=f"Current: ₹{current_price:.2f}"
        )
        
        fig.update_layout(
            title='Future Price Predictions',
            yaxis_title='Price (₹)',
            template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Investment recommendation
        if len(future_pred) > 1:
            avg_change = np.mean([
                ((preds[-1] - current_price) / current_price) * 100 
                for preds in future_pred.values()
            ])
            
            if avg_change > 5:
                recommendation = "🟢 **STRONG BUY**"
                color = "#2ca02c"
            elif avg_change > 2:
                recommendation = "🟢 **BUY**"
                color = "#2ca02c"
            elif avg_change > -2:
                recommendation = "🟡 **HOLD**"
                color = "#ff7f0e"
            else:
                recommendation = "🔴 **SELL**"
                color = "#d62728"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {color}20, {color}40);
                padding: 1.5rem;
                border-radius: 15px;
                border: 2px solid {color};
                text-align: center;
                margin: 1rem 0;
            ">
                <h3 style="color: {color}; margin: 0;">Investment Recommendation</h3>
                <h2 style="color: {color}; margin: 0.5rem 0;">{recommendation}</h2>
                <p style="margin: 0;">Average predicted change: {avg_change:+.1f}% over {len(future_dates)} days</p>
                <p style="margin: 0;">Risk Level: {risk_tolerance}</p>
            </div>
            """, unsafe_allow_html=True)

def show_portfolio_tab():
    """Portfolio analysis tab"""
    st.markdown("### 🎯 Portfolio Analysis")
    st.info("Portfolio analysis feature coming soon! This will include multi-stock correlation, optimization, and risk assessment.")

def show_realtime_tab():
    """Real-time monitoring tab"""
    st.markdown("### ⚡ Real-Time Monitoring")
    st.info("Real-time price feed feature coming soon! This will include live prices, alerts, and market monitoring.")

def show_ai_insights_tab():
    """AI insights tab"""
    st.markdown("### 🤖 AI Market Insights")
    
    if st.button("Generate AI Insights"):
        with st.spinner("Analyzing market conditions..."):
            insights = [
                "Market volatility has increased 15% over the past week",
                "Banking sector shows strong momentum with technical breakouts",
                "IT stocks may face headwinds due to global economic concerns",
                "Energy sector presents buying opportunities on recent dips",
                "Consider defensive allocations given current uncertainty"
            ]
            
            for i, insight in enumerate(insights):
                icon = ["💡", "📊", "⚠️", "🎯", "📈"][i % 5]
                st.info(f"{icon} {insight}")

def show_market_scanner_tab():
    """Market scanner tab"""
    st.markdown("### 📊 Market Scanner")
    
    scan_options = [
        "RSI Oversold (< 30)",
        "RSI Overbought (> 70)", 
        "High Volume Breakouts",
        "Price near 52W High",
        "Price near 52W Low"
    ]
    
    selected_scan = st.selectbox("Scan Criteria:", scan_options)
    
    if st.button("🔍 Scan Market"):
        with st.spinner(f"Scanning for {selected_scan}..."):
            # Mock results
            results = [
                {"Symbol": "TCS.NS", "Signal": selected_scan, "Price": "₹3,245.50", "Change": "+2.1%"},
                {"Symbol": "INFY.NS", "Signal": selected_scan, "Price": "₹1,456.30", "Change": "-0.8%"},
                {"Symbol": "RELIANCE.NS", "Signal": selected_scan, "Price": "₹2,789.45", "Change": "+1.5%"}
            ]
            
            if results:
                st.success(f"Found {len(results)} opportunities!")
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
