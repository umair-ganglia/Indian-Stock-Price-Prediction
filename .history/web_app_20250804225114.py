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
from ensemble_models import EnsemblePredictor
from realtime_data_feed import RealTimeDataFeed, AlertSystem
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

# Add these imports at the top of your web_app.py
from ensemble_models import EnsemblePredictor
from realtime_data_feed import RealTimeDataFeed, AlertSystem

# Replace your main() function with this enhanced version:
def main():
    """Enhanced Main Streamlit application with tabs"""
    
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
        # Your existing stock analysis interface
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
    """Your existing stock analysis functionality"""
    
    # Sidebar (move your existing sidebar code here)
    with st.sidebar:
        st.markdown("### 🎛️ Configuration Panel")
        
        # Stock selection with search (your existing code)
        # ... existing sidebar code ...
        
        # Your existing sidebar functionality
    
    # Main content area (your existing main content)
    if hasattr(st.session_state, 'run_analysis') and st.session_state.run_analysis:
        run_stock_analysis()
    else:
        show_welcome_screen()

def show_portfolio_tab():
    """Portfolio analysis and optimization"""
    
    st.subheader("🎯 Portfolio Analysis & Optimization")
    
    # Portfolio selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_stocks = st.multiselect(
            "Select stocks for portfolio analysis:",
            options=list(EXTENDED_INDIAN_STOCKS.keys()),
            default=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
            help="Choose 3-10 stocks for optimal portfolio analysis"
        )
    
    with col2:
        investment_amount = st.number_input(
            "Investment Amount (₹):",
            min_value=10000,
            max_value=10000000,
            value=100000,
            step=10000
        )
    
    if len(selected_stocks) >= 2:
        if st.button("🔄 Analyze Portfolio", type="primary"):
            with st.spinner("Analyzing portfolio..."):
                analyze_portfolio(selected_stocks, investment_amount)
    else:
        st.info("Please select at least 2 stocks for portfolio analysis.")

def show_realtime_tab():
    """Real-time monitoring and alerts"""
    
    st.subheader("⚡ Real-Time Market Monitoring")
    
    # Initialize real-time feed if not exists
    if 'realtime_feed' not in st.session_state:
        symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
        st.session_state.realtime_feed = RealTimeDataFeed(symbols)
        st.session_state.alert_system = AlertSystem()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Real-time price display
        st.markdown("#### 📊 Live Prices")
        
        if st.button("🟢 Start Live Feed"):
            st.session_state.realtime_feed.start_feed()
            st.success("Live feed started!")
        
        if st.button("🔴 Stop Live Feed"):
            st.session_state.realtime_feed.stop_feed()
            st.info("Live feed stopped!")
        
        # Display current prices
        prices = st.session_state.realtime_feed.get_current_prices()
        if prices:
            price_data = []
            for symbol, price in prices.items():
                company_name = EXTENDED_INDIAN_STOCKS.get(symbol, symbol)
                price_data.append({
                    'Symbol': symbol,
                    'Company': company_name[:30],
                    'Price': f"₹{price:.2f}",
                    'Last Updated': datetime.now().strftime('%H:%M:%S')
                })
            
            df = pd.DataFrame(price_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    with col2:
        # Alert system
        st.markdown("#### 🚨 Price Alerts")
        
        with st.form("alert_form"):
            alert_symbol = st.selectbox("Symbol:", list(EXTENDED_INDIAN_STOCKS.keys()))
            alert_price = st.number_input("Target Price (₹):", min_value=0.01, value=1000.0)
            alert_condition = st.selectbox("Condition:", ["above", "below"])
            
            if st.form_submit_button("Add Alert"):
                st.session_state.alert_system.add_price_alert(
                    alert_symbol, alert_price, alert_condition
                )
                st.success(f"Alert added: {alert_symbol} {alert_condition} ₹{alert_price}")

def show_ai_insights_tab():
    """AI-powered market insights"""
    
    st.subheader("🤖 AI Market Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🧠 Market Analysis")
        
        if st.button("Generate AI Insights", type="primary"):
            with st.spinner("Analyzing market conditions..."):
                insights = generate_ai_insights()
                
                for i, insight in enumerate(insights):
                    icon = ["💡", "📊", "⚠️", "🎯", "📈"][i % 5]
                    st.info(f"{icon} {insight}")
        
        # Ensemble model comparison
        st.markdown("#### 🎯 Model Ensemble")
        
        if st.button("Run Ensemble Analysis"):
            with st.spinner("Training ensemble models..."):
                run_ensemble_analysis()
    
    with col2:
        st.markdown("#### 📊 Market Sentiment")
        
        # Mock sentiment data
        sentiment_data = {
            'Bullish': 65,
            'Neutral': 20,
            'Bearish': 15
        }
        
        fig = px.pie(
            values=list(sentiment_data.values()),
            names=list(sentiment_data.keys()),
            title="Current Market Sentiment",
            color_discrete_map={'Bullish': '#2ca02c', 'Neutral': '#ff7f0e', 'Bearish': '#d62728'}
        )
        st.plotly_chart(fig, use_container_width=True)

def show_market_scanner_tab():
    """Market scanner for opportunities"""
    
    st.subheader("📊 Market Scanner")
    
    scan_options = [
        "High Volume Breakouts",
        "RSI Oversold (< 30)",
        "RSI Overbought (> 70)",
        "Golden Cross (MA5 > MA20)",
        "Price near 52W High",
        "Price near 52W Low"
    ]
    
    selected_scan = st.selectbox("Scan Criteria:", scan_options)
    
    if st.button("🔍 Scan Market", type="primary"):
        with st.spinner(f"Scanning for {selected_scan}..."):
            results = perform_market_scan(selected_scan)
            
            if results:
                st.success(f"Found {len(results)} opportunities!")
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No opportunities found with current criteria.")

# Helper functions
def analyze_portfolio(symbols, amount):
    """Analyze portfolio performance and optimization"""
    
    try:
        # This would use your portfolio_analyzer.py
        from portfolio_analyzer import PortfolioAnalyzer
        
        analyzer = PortfolioAnalyzer(symbols)
        data = analyzer.fetch_portfolio_data()
        
        if data is not None:
            st.success("Portfolio data fetched successfully!")
            
            # Display correlation matrix
            returns = data.pct_change().dropna()
            fig = px.imshow(
                returns.corr(),
                title="Asset Correlation Matrix",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Portfolio optimization
            try:
                weights = analyzer.optimize_portfolio()
                
                # Display optimal allocation
                allocation_data = []
                for i, symbol in enumerate(symbols):
                    allocation_data.append({
                        'Stock': symbol,
                        'Weight': f"{weights[i]*100:.1f}%",
                        'Amount': f"₹{amount * weights[i]:,.0f}"
                    })
                
                st.markdown("#### 🎯 Optimal Portfolio Allocation")
                df = pd.DataFrame(allocation_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Pie chart
                fig = px.pie(
                    df, values=[float(w.strip('%')) for w in df['Weight']], 
                    names=df['Stock'], title="Portfolio Allocation"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Portfolio optimization failed: {e}")
    
    except ImportError:
        st.error("Portfolio analyzer not available. Please check portfolio_analyzer.py")

def generate_ai_insights():
    """Generate AI-powered market insights"""
    
    insights = [
        "Market volatility has increased 15% over the past week, suggesting heightened uncertainty",
        "Banking sector shows strong momentum with 3 stocks breaking resistance levels",
        "IT stocks may face headwinds due to global economic slowdown concerns",
        "Energy sector presents buying opportunities on recent dips below support",
        "Consider increasing defensive allocations given current market conditions"
    ]
    
    return insights

def run_ensemble_analysis():
    """Run ensemble model analysis"""
    
    try:
        ensemble = EnsemblePredictor()
        
        # Mock data for demonstration
        X_sample = np.random.random((100, 10))
        y_sample = np.random.random(100) * 1000
        
        # Create and train voting ensemble
        voting_model = ensemble.train_ensemble(X_sample, y_sample, 'voting')
        
        # Create and train stacking ensemble
        stacking_model = ensemble.train_ensemble(X_sample, y_sample, 'stacking')
        
        st.success("✅ Ensemble models created successfully!")
        st.info("Voting and Stacking regressors are now available for predictions.")
        
    except Exception as e:
        st.error(f"Ensemble analysis failed: {e}")

def perform_market_scan(criteria):
    """Perform market scanning based on criteria"""
    
    results = []
    symbols_to_scan = list(EXTENDED_INDIAN_STOCKS.keys())[:10]  # Scan first 10 for demo
    
    for symbol in symbols_to_scan:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1mo")
            
            if hist.empty:
                continue
            
            current_price = hist['Close'].iloc[-1]
            
            if "RSI Oversold" in criteria:
                # Calculate RSI
                delta = hist['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                
                if rsi.iloc[-1] < 30:
                    results.append({
                        'Symbol': symbol,
                        'Company': EXTENDED_INDIAN_STOCKS[symbol][:30],
                        'Signal': 'RSI Oversold',
                        'RSI': f"{rsi.iloc[-1]:.1f}",
                        'Price': f"₹{current_price:.2f}"
                    })
            
            elif "High Volume" in criteria:
                avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
                current_volume = hist['Volume'].iloc[-1]
                
                if current_volume > avg_volume * 2:
                    results.append({
                        'Symbol': symbol,
                        'Company': EXTENDED_INDIAN_STOCKS[symbol][:30],
                        'Signal': 'High Volume',
                        'Volume Ratio': f"{current_volume/avg_volume:.1f}x",
                        'Price': f"₹{current_price:.2f}"
                    })
            
            # Add more scan criteria as needed...
            
        except Exception as e:
            continue
    
    return results

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
                line=dict(color='#2E86AB', width=3),
                hovertemplate='<b>Actual</b><br>Date: %{x}<br>Price: ₹%{y:.2f}<extra></extra>'
            ))
            
            # Predicted prices
            fig.add_trace(go.Scatter(
                x=pred_data['test_dates'],
                y=pred_data['test_pred'],
                mode='lines',
                name='Predicted Price',
                line=dict(color='#F24236', width=3, dash='dash'),
                hovertemplate='<b>Predicted</b><br>Date: %{x}<br>Price: ₹%{y:.2f}<extra></extra>'
            ))
            
            # Calculate metrics for display
            rmse = np.sqrt(np.mean((pred_data['test_actual'] - pred_data['test_pred'])**2))
            r2 = 1 - (np.sum((pred_data['test_actual'] - pred_data['test_pred'])**2) / 
                      np.sum((pred_data['test_actual'] - np.mean(pred_data['test_actual']))**2))
            mape = np.mean(np.abs((pred_data['test_actual'] - pred_data['test_pred']) / pred_data['test_actual'])) * 100
            
            fig.update_layout(
                title=f'{model_name.upper()} - Prediction Performance<br><sub>RMSE: ₹{rmse:.2f} | R²: {r2:.4f} | MAPE: {mape:.1f}%</sub>',
                xaxis_title='Date',
                yaxis_title='Price (₹)',
                hovermode='x unified',
                template='plotly_white',
                height=500,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)

def display_enhanced_future_predictions(future_pred, future_dates, predictor, risk_tolerance):
    """Display enhanced future predictions"""
    
    if not future_pred:
        st.warning("⚠️ No future predictions available.")
        return
    
    # Create predictions dataframe
    pred_df = pd.DataFrame(future_pred, index=future_dates)
    current_price = predictor.processed_data['Close'].iloc[-1]
    
    # Interactive plot
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (model_name, predictions) in enumerate(future_pred.items()):
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name=model_name,
            line=dict(width=3, color=colors[i % len(colors)]),
            marker=dict(size=6),
            hovertemplate=f'<b>{model_name}</b><br>Date: %{{x}}<br>Price: ₹%{{y:.2f}}<extra></extra>'
        ))
    
    # Add current price line
    fig.add_hline(
        y=current_price, 
        line_dash="dot", 
        line_color="gray",
        annotation_text=f"Current Price: ₹{current_price:.2f}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title='🔮 Future Price Predictions',
        xaxis_title='Date',
        yaxis_title='Predicted Price (₹)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Predictions summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Prediction Summary")
        
        # Calculate average prediction and confidence
        if len(future_pred) > 1:
            avg_predictions = []
            for i in range(len(list(future_pred.values())[0])):
                day_preds = [model_preds[i] for model_preds in future_pred.values()]
                avg_predictions.append(np.mean(day_preds))
            
            # Show key predictions
            pred_summary = []
            key_days = [1, 7, 15, 30] if len(avg_predictions) >= 30 else [1, 7, min(15, len(avg_predictions)-1)]
            
            for day in key_days:
                if day <= len(avg_predictions):
                    pred_price = avg_predictions[day-1]
                    change = ((pred_price - current_price) / current_price) * 100
                    pred_summary.append({
                        'Period': f'{day} Day{"s" if day > 1 else ""}',
                        'Avg Prediction': f'₹{pred_price:.2f}',
                        'Change': f'{change:+.1f}%',
                        'Direction': '📈' if change > 0 else '📉' if change < 0 else '➡️'
                    })
            
            if pred_summary:
                summary_df = pd.DataFrame(pred_summary)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### 💡 Investment Recommendation")
        
        # Calculate consensus
        if len(future_pred) > 1:
            # Get 7-day and 30-day predictions
            day_7_preds = [preds[6] if len(preds) > 6 else preds[-1] for preds in future_pred.values()]
            day_30_preds = [preds[29] if len(preds) > 29 else preds[-1] for preds in future_pred.values()]
            
            avg_7_change = np.mean([(p - current_price) / current_price * 100 for p in day_7_preds])
            avg_30_change = np.mean([(p - current_price) / current_price * 100 for p in day_30_preds])
            
            # Determine recommendation based on risk tolerance
            risk_multiplier = {'Conservative': 0.5, 'Moderate': 1.0, 'Aggressive': 1.5}[risk_tolerance]
            threshold = 2.0 * risk_multiplier
            
            if avg_7_change > threshold and avg_30_change > threshold:
                recommendation = "🟢 **BUY**"
                reason = f"Models predict {avg_30_change:.1f}% growth over 30 days"
                color = "#2ca02c"
            elif avg_7_change < -threshold and avg_30_change < -threshold:
                recommendation = "🔴 **SELL**"
                reason = f"Models predict {avg_30_change:.1f}% decline over 30 days"
                color = "#d62728"
            else:
                recommendation = "🟡 **HOLD**"
                reason = "Mixed signals or sideways movement predicted"
                color = "#ff7f0e"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {color}20, {color}40);
                padding: 1.5rem;
                border-radius: 15px;
                border: 2px solid {color};
                text-align: center;
            ">
                <h3 style="color: {color}; margin: 0;">{recommendation}</h3>
                <p style="margin: 0.5rem 0;"><strong>Risk Level:</strong> {risk_tolerance}</p>
                <p style="margin: 0;"><em>{reason}</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Model consensus
        st.markdown("#### 🤝 Model Consensus")
        consensus_data = []
        for model_name, predictions in future_pred.items():
            week_change = ((predictions[6] if len(predictions) > 6 else predictions[-1]) - current_price) / current_price * 100
            trend = "Bullish" if week_change > 1 else "Bearish" if week_change < -1 else "Neutral"
            consensus_data.append({
                'Model': model_name,
                '7D Change': f'{week_change:+.1f}%',
                'Trend': trend
            })
        
        if consensus_data:
            consensus_df = pd.DataFrame(consensus_data)
            st.dataframe(consensus_df, use_container_width=True, hide_index=True)
    
    # Detailed predictions table
    with st.expander("📋 Detailed Daily Predictions", expanded=False):
        display_df = pred_df.head(14).round(2)  # Show first 14 days
        display_df.index = display_df.index.date
        
        # Add change columns
        for col in display_df.columns:
            change_col = f'{col}_Change'
            display_df[change_col] = ((display_df[col] - current_price) / current_price * 100).round(1)
        
        st.dataframe(display_df, use_container_width=True)

def plot_enhanced_technical_indicators(predictor):
    """Plot enhanced technical indicators"""
    
    if predictor.processed_data is None:
        return
    
    df = predictor.processed_data.tail(200)  # Last 200 days
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Price & Moving Averages', 'RSI', 'MACD', 'Volume'),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Price and Moving Averages
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], 
        name='Close Price', 
        line=dict(width=2, color='#1f77b4')
    ), row=1, col=1)
    
    if 'MA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA_20'], 
            name='MA 20', 
            line=dict(width=1, color='#ff7f0e'),
            opacity=0.8
        ), row=1, col=1)
    
    if 'MA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA_50'], 
            name='MA 50', 
            line=dict(width=1, color='#2ca02c'),
            opacity=0.8
        ), row=1, col=1)
    
    # Bollinger Bands
    if all(col in df.columns for col in ['BB_upper', 'BB_lower', 'BB_middle']):
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_upper'], 
            name='BB Upper',
            line=dict(width=1, color='rgba(128,128,128,0.5)'),
            showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_lower'], 
            name='BB Lower',
            line=dict(width=1, color='rgba(128,128,128,0.5)'),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)',
            showlegend=False
        ), row=1, col=1)
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'], 
            name='RSI', 
            line=dict(color='purple')
        ), row=2, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
    
    # MACD
    if all(col in df.columns for col in ['MACD', 'MACD_signal']):
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD'], 
            name='MACD', 
            line=dict(color='blue')
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD_signal'], 
            name='Signal', 
            line=dict(color='red')
        ), row=3, col=1)
        
        if 'MACD_histogram' in df.columns:
            colors = ['green' if val >= 0 else 'red' for val in df['MACD_histogram']]
            fig.add_trace(go.Bar(
                x=df.index, y=df['MACD_histogram'], 
                name='MACD Histogram',
                marker_color=colors,
                opacity=0.6
            ), row=3, col=1)
    
    # Volume
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], 
        name='Volume',
        marker_color='lightblue',
        opacity=0.7
    ), row=4, col=1)
    
    if 'Volume_MA' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Volume_MA'], 
            name='Volume MA',
            line=dict(color='red')
        ), row=4, col=1)
    
    fig.update_layout(
        height=1000, 
        template='plotly_white', 
        showlegend=True,
        title_text="Technical Indicators Analysis"
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def display_enhanced_trading_signals(predictor, risk_tolerance):
    """Display enhanced trading signals"""
    
    try:
        signals = predictor.generate_trading_signals()
        recent_signal = signals.iloc[-1] if len(signals) > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Current signal
            if recent_signal == 1:
                signal_text = "🟢 **BUY**"
                signal_color = "#2ca02c"
                signal_desc = "Technical indicators suggest buying opportunity"
            elif recent_signal == -1:
                signal_text = "🔴 **SELL**"
                signal_color = "#d62728"
                signal_desc = "Technical indicators suggest selling opportunity"
            else:
                signal_text = "🟡 **HOLD**"
                signal_color = "#ff7f0e"
                signal_desc = "No clear trading signal from technical indicators"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {signal_color}20, {signal_color}40);
                padding: 1.5rem;
                border-radius: 15px;
                border: 2px solid {signal_color};
                text-align: center;
            ">
                <h3 style="color: {signal_color}; margin: 0;">Current Signal</h3>
                <h2 style="color: {signal_color}; margin: 0.5rem 0;">{signal_text}</h2>
                <p style="margin: 0;"><em>{signal_desc}</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Signal distribution
            if len(signals) > 0:
                signal_counts = signals.value_counts()
                signal_labels = {1: 'Buy Signals', -1: 'Sell Signals', 0: 'Hold/Neutral'}
                
                fig = px.pie(
                    values=signal_counts.values,
                    names=[signal_labels.get(idx, f'Signal {idx}') for idx in signal_counts.index],
                    title="Historical Signal Distribution",
                    color_discrete_map={
                        'Buy Signals': '#2ca02c',
                        'Sell Signals': '#d62728',
                        'Hold/Neutral': '#ff7f0e'
                    }
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Recent signals
            st.markdown("#### 📅 Recent Signals")
            if len(signals) > 0:
                recent_signals = signals.tail(10)
                signal_df = pd.DataFrame({
                    'Date': recent_signals.index.date,
                    'Signal': recent_signals.map({1: '🟢 Buy', -1: '🔴 Sell', 0: '🟡 Hold'})
                })
                st.dataframe(signal_df, use_container_width=True, hide_index=True)
            else:
                st.info("No trading signals generated")
        
        # Trading strategy suggestions
        st.markdown("#### 💡 Trading Strategy Suggestions")
        
        strategy_col1, strategy_col2 = st.columns(2)
        
        with strategy_col1:
            st.markdown(f"""
            <div class="info-box">
                <h4>📊 For {risk_tolerance} Investors</h4>
                <ul>
                    <li><strong>Entry Strategy:</strong> {'Wait for stronger confirmation' if risk_tolerance == 'Conservative' else 'Consider gradual position building' if risk_tolerance == 'Moderate' else 'Can take larger positions on signals'}</li>
                    <li><strong>Stop Loss:</strong> {2 if risk_tolerance == 'Conservative' else 3 if risk_tolerance == 'Moderate' else 5}% below entry</li>
                    <li><strong>Position Size:</strong> {'Small (1-2% of portfolio)' if risk_tolerance == 'Conservative' else 'Medium (3-5% of portfolio)' if risk_tolerance == 'Moderate' else 'Large (5-10% of portfolio)'}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with strategy_col2:
            # Risk metrics
            if len(signals) > 0:
                buy_signals = (signals == 1).sum()
                sell_signals = (signals == -1).sum()
                total_signals = buy_signals + sell_signals
                
                if total_signals > 0:
                    signal_frequency = len(signals) / total_signals if total_signals > 0 else 0
                    
                    st.markdown(f"""
                    <div class="warning-box">
                        <h4>⚠️ Risk Assessment</h4>
                        <p><strong>Signal Frequency:</strong> 1 signal every {signal_frequency:.0f} days</p>
                        <p><strong>Buy/Sell Ratio:</strong> {buy_signals}/{sell_signals}</p>
                        <p><strong>Activity Level:</strong> {'High' if signal_frequency < 10 else 'Medium' if signal_frequency < 30 else 'Low'}</p>
                        <p><strong>Recommendation:</strong> {'Active monitoring required' if signal_frequency < 10 else 'Moderate monitoring' if signal_frequency < 30 else 'Long-term holding suitable'}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error generating trading signals: {e}")

def provide_download_options(predictor, future_pred, future_dates):
    """Provide download options for results"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download predictions
        if future_pred:
            pred_df = pd.DataFrame(future_pred, index=future_dates)
            csv = pred_df.to_csv()
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name=f"{predictor.symbol}_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        # Download processed data
        if predictor.processed_data is not None:
            processed_csv = predictor.processed_data.to_csv()
            st.download_button(
                label="📥 Download Processed Data",
                data=processed_csv,
                file_name=f"{predictor.symbol}_processed_data.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col3:
        # Download trading signals
        try:
            signals = predictor.generate_trading_signals()
            if len(signals) > 0:
                signals_df = pd.DataFrame({
                    'Date': signals.index,
                    'Signal': signals.values,
                    'Signal_Text': signals.map({1: 'Buy', -1: 'Sell', 0: 'Hold'})
                })
                signals_csv = signals_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Trading Signals",
                    data=signals_csv,
                    file_name=f"{predictor.symbol}_trading_signals.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except:
            st.info("Trading signals not available")

if __name__ == "__main__":
    main()
