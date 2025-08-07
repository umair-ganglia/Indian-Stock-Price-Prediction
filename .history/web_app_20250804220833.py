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

# Add src to path (if needed)
sys.path.append('src')

try:
    from stock_predictor import IndianStockPredictor
except ImportError as e:
    # If import fails, show error
    st.error(f"""
    ❌ **Import Error**: {e}
    
    Please ensure that:
    1. The `stock_predictor.py` file is in the same directory or in a `src/` folder
    2. All required packages are installed: `pip install yfinance scikit-learn pandas numpy matplotlib seaborn`
    3. For LSTM: `pip install tensorflow`
    4. For Prophet: `pip install prophet` or `pip install fbprophet`
    """)
    st.stop()

# Enhanced Custom CSS for better styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --danger-color: #d62728;
        --warning-color: #ff7f0e;
        --info-color: #17a2b8;
        --light-gray: #f8f9fa;
        --dark-gray: #343a40;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main header styling */
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
    
    /* Custom card styling */
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
    
    .error-box {
        padding: 1.5rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
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
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
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
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Metric styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    
    /* Enhanced table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

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
    'HDFCLIFE.NS': 'HDFC Life Insurance',
    'SBILIFE.NS': 'SBI Life Insurance',
    
    # Energy & Oil
    'RELIANCE.NS': 'Reliance Industries Limited',
    'ONGC.NS': 'Oil and Natural Gas Corporation',
    'IOC.NS': 'Indian Oil Corporation',
    'BPCL.NS': 'Bharat Petroleum Corporation',
    'HINDPETRO.NS': 'Hindustan Petroleum Corporation',
    'ADANIPORTS.NS': 'Adani Ports and SEZ Limited',
    'ADANIGREEN.NS': 'Adani Green Energy Limited',
    
    # FMCG & Consumer
    'HINDUNILVR.NS': 'Hindustan Unilever Limited',
    'ITC.NS': 'ITC Limited',
    'NESTLEIND.NS': 'Nestle India Limited',
    'BRITANNIA.NS': 'Britannia Industries Limited',
    'DABUR.NS': 'Dabur India Limited',
    'GODREJCP.NS': 'Godrej Consumer Products',
    'MARICO.NS': 'Marico Limited',
    'COLPAL.NS': 'Colgate-Palmolive India',
    
    # Automobiles
    'MARUTI.NS': 'Maruti Suzuki India Limited',
    'TATAMOTORS.NS': 'Tata Motors Limited',
    'M&M.NS': 'Mahindra & Mahindra Limited',
    'BAJAJ-AUTO.NS': 'Bajaj Auto Limited',
    'HEROMOTOCO.NS': 'Hero MotoCorp Limited',
    'EICHERMOT.NS': 'Eicher Motors Limited',
    'ASHOKLEY.NS': 'Ashok Leyland Limited',
    
    # Pharmaceuticals
    'SUNPHARMA.NS': 'Sun Pharmaceutical Industries',
    'DRREDDY.NS': 'Dr. Reddy\'s Laboratories',
    'CIPLA.NS': 'Cipla Limited',
    'BIOCON.NS': 'Biocon Limited',
    'AUROPHARMA.NS': 'Aurobindo Pharma Limited',
    'LUPIN.NS': 'Lupin Limited',
    'DIVISLAB.NS': 'Divi\'s Laboratories Limited',
    
    # Telecom
    'BHARTIARTL.NS': 'Bharti Airtel Limited',
    'IDEA.NS': 'Vodafone Idea Limited',
    
    # Metals & Mining
    'TATASTEEL.NS': 'Tata Steel Limited',
    'JSWSTEEL.NS': 'JSW Steel Limited',
    'HINDALCO.NS': 'Hindalco Industries Limited',
    'COALINDIA.NS': 'Coal India Limited',
    'VEDL.NS': 'Vedanta Limited',
    'NMDC.NS': 'NMDC Limited',
    
    # Construction & Infrastructure
    'LT.NS': 'Larsen & Toubro Limited',
    'ULTRACEMCO.NS': 'UltraTech Cement Limited',
    'SHREECEM.NS': 'Shree Cement Limited',
    'ACC.NS': 'ACC Limited',
    'GRASIM.NS': 'Grasim Industries Limited',
    
    # Power & Utilities
    'POWERGRID.NS': 'Power Grid Corporation',
    'NTPC.NS': 'NTPC Limited',
    'TATAPOWER.NS': 'Tata Power Company',
    
    # Consumer Durables
    'TITAN.NS': 'Titan Company Limited',
    'ASIANPAINT.NS': 'Asian Paints Limited',
    'PIDILITIND.NS': 'Pidilite Industries Limited',
    'VOLTAS.NS': 'Voltas Limited',
    'HAVELLS.NS': 'Havells India Limited',
}

def show_animated_header():
    """Display animated header"""
    header_html = """
    <div style="text-align: center; padding: 2rem;">
        <h1 class="main-header">📈 Indian Stock Price Prediction System</h1>
        <p style="font-size: 1.2rem; color: #666; margin-top: -1rem;">
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
    """Main Streamlit application"""
    
    # Animated Header
    show_animated_header()
    st.markdown("---")
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("### 🎛️ Configuration Panel")
        
        # Stock selection with search
        st.markdown("#### 📊 Stock Selection")
        
        # Search functionality
        search_term = st.text_input("🔍 Search for a stock:", placeholder="e.g., TCS, Reliance, HDFC")
        
        if search_term:
            # Filter stocks based on search
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
        
        # Time period selection with descriptions
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
        
        # Model selection with explanations
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
            
            # Risk tolerance
            st.markdown("**Risk Tolerance:**")
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
    
    # Hero section
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
    
    # Feature highlights
    st.markdown("### ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🎯 Advanced Analytics</h4>
            <ul>
                <li>Multiple ML model comparison</li>
                <li>Technical indicator analysis</li>
                <li>Risk assessment metrics</li>
                <li>Trading signal generation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>📊 Interactive Visualizations</h4>
            <ul>
                <li>Real-time price charts</li>
                <li>Prediction comparisons</li>
                <li>Technical indicators overlay</li>
                <li>Performance metrics dashboard</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Available stocks showcase
    st.markdown("### 📈 Featured Stocks")
    
    # Create tabs for different sectors
    tab1, tab2, tab3, tab4 = st.tabs(["🏦 Banking", "💻 IT", "🏭 Industrial", "🛒 Consumer"])
    
    with tab1:
        banking_stocks = {k: v for k, v in EXTENDED_INDIAN_STOCKS.items() if any(word in v.lower() for word in ['bank', 'finance'])}
        display_stock_grid(banking_stocks)
    
    with tab2:
        it_stocks = {k: v for k, v in EXTENDED_INDIAN_STOCKS.items() if any(word in v.lower() for word in ['tech', 'info', 'software', 'system'])}
        display_stock_grid(it_stocks)
    
    with tab3:
        industrial_stocks = {k: v for k, v in EXTENDED_INDIAN_STOCKS.items() if any(word in v.lower() for word in ['steel', 'cement', 'power', 'construction', 'larsen'])}
        display_stock_grid(industrial_stocks)
    
    with tab4:
        consumer_stocks = {k: v for k, v in EXTENDED_INDIAN_STOCKS.items() if any(word in v.lower() for word in ['consumer', 'unilever', 'itc', 'titan', 'paint'])}
        display_stock_grid(consumer_stocks)
    
    # Getting started guide
    st.markdown("### 🚀 Getting Started")
    
    st.markdown("""
    <div class="warning-box">
        <h4>📋 Quick Start Guide</h4>
        <ol>
            <li><strong>Select a Stock:</strong> Choose from 50+ Indian stocks or enter a custom symbol</li>
            <li><strong>Pick Time Period:</strong> Select from 3 months to 5 years of historical data</li>
            <li><strong>Choose Models:</strong> Enable the AI models you want to use</li>
            <li><strong>Configure Settings:</strong> Adjust prediction days and model parameters</li>
            <li><strong>Start Analysis:</strong> Click the "Start Analysis" button and wait for results</li>
        </ol>
        <p><strong>💡 Tip:</strong> For best results, use 1-2 years of data with all three models enabled!</p>
    </div>
    """, unsafe_allow_html=True)

def display_stock_grid(stocks_dict):
    """Display stocks in a grid format"""
    if not stocks_dict:
        st.info("No stocks available in this category.")
        return
    
    # Create a DataFrame for better display
    stock_data = []
    for symbol, name in list(stocks_dict.items())[:8]:  # Show first 8
        try:
            # Get basic info (cached to avoid repeated API calls)
            stock_info = get_basic_stock_info(symbol)
            stock_data.append({
                'Symbol': symbol,
                'Company': name[:30] + "..." if len(name) > 30 else name,
                'Current Price': stock_info.get('price', 'N/A'),
                'Change %': stock_info.get('change', 'N/A')
            })
        except:
            stock_data.append({
                'Symbol': symbol,
                'Company': name[:30] + "..." if len(name) > 30 else name,
                'Current Price': 'Loading...',
                'Change %': 'Loading...'
            })
    
    if stock_data:
        df = pd.DataFrame(stock_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_basic_stock_info(symbol):
    """Get basic stock information with caching"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="2d")
        if len(hist) >= 2:
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            change_pct = ((current_price - prev_price) / prev_price) * 100
            return {
                'price': f"₹{current_price:.2f}",
                'change': f"{change_pct:+.1f}%"
            }
    except:
        pass
    return {'price': 'N/A', 'change': 'N/A'}

def run_stock_analysis():
    """Run the complete stock analysis with enhanced UI"""
    
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
    
    # Progress tracking with enhanced UI
    progress_container = st.container()
    status_container = st.container()
    
    with progress_container:
        progress_col1, progress_col2 = st.columns([3, 1])
        with progress_col1:
            progress_bar = st.progress(0)
        with progress_col2:
            progress_text = st.empty()
    
    with status_container:
        status_text = st.empty()
        current_step = st.empty()
    
    try:
        # Initialize predictor
        current_step.markdown("**Step 1/7:** 🚀 Initializing AI system...")
        status_text.info("Setting up the prediction engine...")
        progress_bar.progress(10)
        progress_text.markdown("10%")
        time.sleep(1)
        
        predictor = IndianStockPredictor(symbol=stock, period=period)
        
        # Fetch data
        current_step.markdown("**Step 2/7:** 📡 Fetching market data...")
        status_text.info(f"Downloading {period} of historical data for {stock}...")
        progress_bar.progress(20)
        progress_text.markdown("20%")
        
        data = predictor.fetch_data()
        if data is None:
            st.error("❌ Failed to fetch data. Please check the stock symbol and try again.")
            return
        
        # Prepare data
        current_step.markdown("**Step 3/7:** 🔧 Engineering features...")
        status_text.info("Processing data and calculating technical indicators...")
        progress_bar.progress(30)
        progress_text.markdown("30%")
        
        predictor.prepare_data()
        
        # Display stock info with enhanced cards
        display_enhanced_stock_info(data, stock)
        
        # Train models
        results = {}
        current_progress = 40
        model_count = sum(models.values())
        progress_per_model = 40 // model_count if model_count > 0 else 0
        
        if models['linear_regression']:
            current_step.markdown("**Step 4/7:** 🤖 Training Linear Regression...")
            status_text.info("Training traditional statistical model...")
            progress_bar.progress(current_progress)
            progress_text.markdown(f"{current_progress}%")
            
            predictor.train_linear_regression()
            current_progress += progress_per_model
        
        if models['lstm']:
            current_step.markdown("**Step 5/7:** 🧠 Training LSTM Neural Network...")
            status_text.info(f"Training deep learning model ({lstm_epochs} epochs)...")
            progress_bar.progress(current_progress)
            progress_text.markdown(f"{current_progress}%")
            
            predictor.train_lstm(epochs=lstm_epochs)
            current_progress += progress_per_model
        
        if models['prophet']:
            current_step.markdown("**Step 6/7:** 📊 Training Prophet Model...")
            status_text.info("Training time-series forecasting model...")
            progress_bar.progress(current_progress)
            progress_text.markdown(f"{current_progress}%")
            
            predictor.train_prophet()
            current_progress += progress_per_model
        
        # Evaluate models
        current_step.markdown("**Step 7/7:** 📊 Evaluating models & generating predictions...")
        status_text.info("Comparing model performance and generating forecasts...")
        progress_bar.progress(90)
        progress_text.markdown("90%")
        
        results = predictor.evaluate_models()
        
        # Generate predictions
        future_pred, future_dates = predictor.predict_future(days=prediction_days)
        
        progress_bar.progress(100)
        progress_text.markdown("100%")
        current_step.markdown("**✅ Analysis Complete!**")
        status_text.success("🎉 All models trained successfully! Scroll down to see results.")
        
        time.sleep(2)
        
        # Clear progress indicators
        progress_container.empty()
        status_container.empty()
        
        # Display results with enhanced visualizations
        display_enhanced_results(predictor, results, future_pred, future_dates, risk_tolerance)
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        with st.expander("🔍 Error Details", expanded=False):
            st.exception(e)

def display_enhanced_stock_info(data, stock):
    """Display enhanced stock information with beautiful cards"""
    
    st.markdown("### 📊 Stock Overview")
    
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        delta_text = f"{price_change:+.2f} ({price_change_pct:+.1f}%)"
        st.metric(
            "Current Price",
            f"₹{current_price:.2f}",
            delta=delta_text
        )
    
    with col2:
        high_52w = data['High'].max()
        st.metric("52W High", f"₹{high_52w:.2f}")
    
    with col3:
        low_52w = data['Low'].min()
        st.metric("52W Low", f"₹{low_52w:.2f}")
    
    with col4:
        avg_volume = data['Volume'].tail(30).mean()
        current_volume = data['Volume'].iloc[-1]
        volume_change = ((current_volume - avg_volume) / avg_volume) * 100
        st.metric(
            "Volume", 
            f"{current_volume:,.0f}",
            delta=f"{volume_change:+.1f}% vs 30D avg"
        )
    
    with col5:
        volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
        st.metric("Annual Volatility", f"{volatility:.1f}%")
    
    # Price chart
    st.markdown("### 📈 Price Chart")
    
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
        xaxis_title='Date',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_enhanced_results(predictor, results, future_pred, future_dates, risk_tolerance):
    """Display enhanced analysis results"""
    
    # Model comparison with enhanced styling
    st.markdown("### 🏆 Model Performance Comparison")
    
    if results:
        results_df = pd.DataFrame(results).T
        results_df = results_df.round(4)
        
        # Find best model
        best_model = results_df['RMSE'].idxmin()
        
        # Create performance cards
        col1, col2, col3 = st.columns(3)
        
        for i, (model_name, metrics) in enumerate(results.items()):
            col = [col1, col2, col3][i % 3]
            
            with col:
                is_best = model_name == best_model
                card_color = "#2ca02c" if is_best else "#1f77b4"
                crown = "👑 " if is_best else ""
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {card_color}20, {card_color}40);
                    padding: 1.5rem;
                    border-radius: 15px;
                    border: 2px solid {card_color};
                    margin: 0.5rem 0;
                ">
                    <h3 style="color: {card_color}; margin: 0;">{crown}{model_name.upper()}</h3>
                    <hr style="border-color: {card_color};">
                    <p><strong>RMSE:</strong> ₹{metrics['RMSE']:.2f}</p>
                    <p><strong>R²:</strong> {metrics['R²']:.4f}</p>
                    <p><strong>MAPE:</strong> {metrics['MAPE']:.1f}%</p>
                    <p><strong>Direction Accuracy:</strong> {metrics['Directional_Accuracy']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed comparison table
        with st.expander("📊 Detailed Model Comparison", expanded=False):
            st.dataframe(results_df, use_container_width=True)
    
    # Predictions visualization
    st.markdown("### 📊 Model Predictions vs Actual")
    plot_enhanced_predictions(predictor)
    
    # Future predictions
    st.markdown("### 🔮 Future Price Predictions")
    display_enhanced_future_predictions(future_pred, future_dates, predictor, risk_tolerance)
    
    # Technical analysis
    st.markdown("### 📈 Technical Analysis")
    plot_enhanced_technical_indicators(predictor)
    
    # Trading signals
    st.markdown("### 🎯 Trading Signals & Recommendations")
    display_enhanced_trading_signals(predictor, risk_tolerance)
    
    # Download section
    st.markdown("### 💾 Export Results")
    provide_download_options(predictor, future_pred, future_dates)

def plot_enhanced_predictions(predictor):
    """Create enhanced prediction plots"""
    
    tabs = st.tabs([f"📊 {name.upper()}" for name in predictor.predictions.keys()])
    
    for i, (model_name, pred_data) in enumerate(predictor.predictions.items()):
        with tabs[i]:
            fig = go.Figure()
            
            # Actual prices
            fig.add_trace(go.Scatter(
                x=pred_data['test_dates'],
                y=pred_data['test_actual'],
                mode='lines',
                name='Actual Price',
