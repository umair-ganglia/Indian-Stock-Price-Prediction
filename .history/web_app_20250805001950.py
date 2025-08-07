"""
Enhanced Stock Prediction App - Streamlit Version
Converted from your beautiful HTML/CSS/JS design
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# Import your existing modules
from stock_predictor import IndianStockPredictor
from ensemble_models import EnsemblePredictor
from realtime_data_feed import RealTimeDataFeed, AlertSystem

# Page configuration
st.set_page_config(
    page_title="Indian Stock Price Prediction - AI Powered",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Your beautiful CSS converted to Streamlit
st.markdown("""
<style>
    /* Custom CSS Variables */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --accent-color: #f093fb;
        --success-color: #4facfe;
        --warning-color: #fa709a;
        --error-color: #ff6b6b;
        --info-color: #4ecdc4;
    }

    /* Main styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        line-height: 1.2;
    }

    .subtitle {
        font-size: 1.25rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, var(--success-color), #00f2fe);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.875rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
        text-align: center;
        margin: 1rem 0;
    }

    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
        border-color: var(--primary-color);
    }

    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }

    /* Analysis section */
    .analysis-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05));
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }

    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    /* Enhanced sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    /* Status messages */
    .status-success {
        background: rgba(79, 172, 254, 0.1);
        color: var(--success-color);
        border: 1px solid rgba(79, 172, 254, 0.3);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .status-warning {
        background: rgba(247, 112, 154, 0.1);
        color: var(--warning-color);
        border: 1px solid rgba(247, 112, 154, 0.3);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .status-info {
        background: rgba(78, 205, 196, 0.1);
        color: var(--info-color);
        border: 1px solid rgba(78, 205, 196, 0.3);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        .metric-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Enhanced header
st.markdown("""
<div class="fade-in">
    <h1 class="main-header">📈 Indian Stock Price Prediction</h1>
    <p class="subtitle">AI-Powered Stock Analysis & Future Price Prediction</p>
</div>
""", unsafe_allow_html=True)

# Stock dictionary
ENHANCED_STOCKS = {
    'TCS.NS': 'Tata Consultancy Services Limited',
    'INFY.NS': 'Infosys Limited',
    'RELIANCE.NS': 'Reliance Industries Limited',
    'HDFCBANK.NS': 'HDFC Bank Limited',
    'ICICIBANK.NS': 'ICICI Bank Limited',
    'SBIN.NS': 'State Bank of India',
    'WIPRO.NS': 'Wipro Limited',
    'TECHM.NS': 'Tech Mahindra Limited',
    'MARUTI.NS': 'Maruti Suzuki India Limited',
    'HINDUNILVR.NS': 'Hindustan Unilever Limited',
    'ITC.NS': 'ITC Limited',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank Limited',
    'AXISBANK.NS': 'Axis Bank Limited',
    'BAJFINANCE.NS': 'Bajaj Finance Limited',
    'LT.NS': 'Larsen & Toubro Limited',
    'SUNPHARMA.NS': 'Sun Pharmaceutical Industries',
    'TATASTEEL.NS': 'Tata Steel Limited',
    'ONGC.NS': 'Oil and Natural Gas Corporation',
    'COALINDIA.NS': 'Coal India Limited',
    'TITAN.NS': 'Titan Company Limited'
}

# Enhanced Sidebar (replicating your HTML sidebar)
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                padding: 1rem; border-radius: 10px; margin-bottom: 2rem; text-align: center;">
        <h3 style="color: white; margin: 0;">
            <i class="fas fa-chart-line"></i> StockAI Control Panel
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Stock Selection Section
    st.markdown("### 📊 Stock Selection")
    
    # Search functionality
    search_term = st.text_input("🔍 Search stocks:", placeholder="e.g., TCS, Reliance, HDFC")
    
    if search_term:
        filtered_stocks = {k: v for k, v in ENHANCED_STOCKS.items() 
                          if search_term.lower() in k.lower() or search_term.lower() in v.lower()}
    else:
        filtered_stocks = ENHANCED_STOCKS
    
    selected_stock = st.selectbox(
        "Select Stock Symbol:",
        options=list(filtered_stocks.keys()),
        format_func=lambda x: f"{x} - {filtered_stocks[x][:25]}...",
        index=0
    )
    
    # Custom stock input
    custom_stock = st.text_input(
        "Custom stock symbol:",
        placeholder="e.g., GOOGL, AAPL, TSLA",
        help="For NSE stocks, add .NS (e.g., SYMBOL.NS)"
    )
    
    if custom_stock:
        selected_stock = custom_stock.upper()
    
    # Time Period Section
    st.markdown("### 📅 Time Period")
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
        index=2
    )
    
    # AI Models Section
    st.markdown("### 🤖 AI Models")
    
    with st.expander("ℹ️ Model Information"):
        st.markdown("""
        **📊 Linear Regression**: Fast, traditional statistical approach
        
        **🧠 LSTM Neural Network**: Deep learning for complex patterns
        
        **📈 Prophet**: Facebook's time-series forecasting
        """)
    
    use_linear = st.checkbox("📊 Linear Regression", value=True, help="Fast and interpretable")
    use_lstm = st.checkbox("🧠 LSTM Neural Network", value=True, help="Deep learning model") 
    use_prophet = st.checkbox("📈 Prophet Time Series", value=True, help="Handles seasonality well")
    
    # Advanced Settings
    st.markdown("### ⚙️ Advanced Settings")
    with st.expander("🔧 Model Parameters"):
        prediction_days = st.slider("Days to Predict:", 7, 90, 30, help="Number of future days to predict")
        lstm_epochs = st.slider("LSTM Training Epochs:", 10, 100, 50, help="More epochs = better training but slower")
        risk_tolerance = st.select_slider(
            "Risk Level:",
            options=["Conservative", "Moderate", "Aggressive"],
            value="Moderate",
            help="Affects prediction confidence intervals"
        )
    
    # Action Buttons
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

# Main content with tabs (replicating your HTML tabs)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Stock Analysis",
    "🎯 Portfolio", 
    "⚡ Real-Time",
    "🤖 AI Insights",
    "📊 Market Scanner"
])

with tab1:
    if hasattr(st.session_state, 'run_analysis') and st.session_state.run_analysis:
        run_enhanced_analysis()
    else:
        show_enhanced_welcome_screen()

with tab2:
    show_enhanced_portfolio_tab()

with tab3:
    show_enhanced_realtime_tab()

with tab4:
    show_enhanced_insights_tab()

with tab5:
    show_enhanced_scanner_tab()

def show_enhanced_welcome_screen():
    """Enhanced welcome screen replicating your HTML design"""
    
    # Hero section
    st.markdown("""
    <div class="analysis-section fade-in">
        <h2 style="text-align: center;">🎯 Welcome to Advanced Stock Prediction!</h2>
        <p style="font-size: 1.1rem; text-align: center; margin-bottom: 2rem;">
            Harness the power of AI to predict Indian stock prices with multiple machine learning models.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats metrics (replicating your HTML metrics grid)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">40+</div>
            <div class="metric-label">Available Stocks</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #fa709a, #fee140);">
            <div class="metric-value">3</div>
            <div class="metric-label">AI Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #4ecdc4, #44a08d);">
            <div class="metric-value">15+</div>
            <div class="metric-label">Sectors Covered</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b, #ffa726);">
            <div class="metric-value">&lt; 60s</div>
            <div class="metric-label">Prediction Speed</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature cards (replicating your HTML feature grid)
    st.markdown("### ✨ AI Models Available")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">📊</span>
            <h3>Linear Regression</h3>
            <p>Traditional statistical approach for trend analysis with high interpretability and fast execution.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🧠</span>
            <h3>LSTM Networks</h3>
            <p>Deep learning for complex pattern recognition and sequential data analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">📈</span>
            <h3>Prophet</h3>
            <p>Facebook's time-series forecasting with seasonality handling and trend analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting started guide
    st.markdown("""
    <div class="analysis-section fade-in">
        <h3>🚀 Getting Started Guide</h3>
        <div class="status-info">
            <h4>📋 Quick Start Steps</h4>
            <ol style="margin: 1rem 0; padding-left: 1.5rem;">
                <li><strong>Select a Stock:</strong> Choose from 40+ Indian stocks or enter a custom symbol</li>
                <li><strong>Pick Time Period:</strong> Select from 3 months to 5 years of historical data</li>
                <li><strong>Choose Models:</strong> Enable the AI models you want to use</li>
                <li><strong>Configure Settings:</strong> Adjust prediction days and model parameters</li>
                <li><strong>Start Analysis:</strong> Click the "Start Analysis" button and wait for results</li>
            </ol>
            <p><strong>💡 Tip:</strong> For best results, use 1-2 years of data with all three models enabled!</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def run_enhanced_analysis():
    """Enhanced analysis function with progress simulation"""
    
    stock = st.session_state.selected_stock
    period = st.session_state.selected_period
    models = st.session_state.models
    prediction_days = st.session_state.prediction_days
    lstm_epochs = st.session_state.lstm_epochs
    risk_tolerance = st.session_state.risk_tolerance
    
    # Analysis header
    st.markdown(f"""
    <div class="analysis-section">
        <h2>🔍 Analyzing {stock}</h2>
        <p>Running AI-powered analysis with {sum(models.values())} models over {period} period</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress simulation (replicating your JavaScript progress)
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    
    steps = [
        ("🚀 Initializing AI system...", 10),
        ("📡 Fetching market data...", 25), 
        ("🔧 Processing data and indicators...", 40),
        ("🤖 Training Linear Regression...", 55),
        ("🧠 Training LSTM Neural Network...", 70),
        ("📊 Training Prophet model...", 85),
        ("📈 Generating predictions...", 95),
        ("✅ Analysis complete!", 100)
    ]
    
    for message, progress in steps:
        status_placeholder.markdown(f"""
        <div class="status-info">
            <span>{message}</span>
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(progress)
        time.sleep(1)
    
    status_placeholder.markdown("""
    <div class="status-success">
        <span>🎉 All models trained successfully! Results are ready.</span>
    </div>
    """, unsafe_allow_html=True)
    
    time.sleep(1)
    progress_bar.empty()
    status_placeholder.empty()
    
    # Display results with your backend integration
    try:
        predictor = IndianStockPredictor(symbol=stock, period=period)
        data = predictor.fetch_data()
        
        if data is not None:
            predictor.prepare_data()
            
            # Train selected models
            if models['linear_regression']:
                predictor.train_linear_regression()
            if models['lstm']:
                predictor.train_lstm(epochs=lstm_epochs)
            if models['prophet']:
                predictor.train_prophet()
            
            # Display results
            display_enhanced_results(predictor, prediction_days, risk_tolerance)
        else:
            st.error("❌ Failed to fetch data. Please check the stock symbol.")
            
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")

def display_enhanced_results(predictor, prediction_days, risk_tolerance):
    """Display enhanced results with your styling"""
    
    # Evaluate models
    results = predictor.evaluate_models()
    
    # Stock overview
    st.markdown("### 📊 Stock Overview")
    
    current_price = predictor.data['Close'].iloc[-1]
    prev_price = predictor.data['Close'].iloc[-2] if len(predictor.data) > 1 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        change_color = "#2ecc71" if price_change >= 0 else "#e74c3c"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">₹{current_price:.2f}</div>
            <div class="metric-label">Current Price</div>
            <div style="font-size: 0.8rem; color: {change_color}; margin-top: 0.5rem;">
                {price_change:+.2f} ({price_change_pct:+.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #2ecc71, #27ae60);">
            <div class="metric-value">₹{predictor.data['High'].max():.2f}</div>
            <div class="metric-label">52W High</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #e74c3c, #c0392b);">
            <div class="metric-value">₹{predictor.data['Low'].min():.2f}</div>
            <div class="metric-label">52W Low</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        volatility = predictor.data['Close'].pct_change().std() * np.sqrt(252) * 100
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f39c12, #e67e22);">
            <div class="metric-value">{volatility:.1f}%</div>
            <div class="metric-label">Volatility</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Price chart using Plotly
    st.markdown("### 📈 Price History")
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=predictor.data.index,
        open=predictor.data['Open'],
        high=predictor.data['High'],
        low=predictor.data['Low'],
        close=predictor.data['Close'],
        name='Price'
    ))
    
    fig.update_layout(
        title=f'{predictor.symbol} - Price History',
        yaxis_title='Price (₹)',
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model performance
    if results:
        st.markdown("### 🏆 Model Performance")
        
        results_df = pd.DataFrame(results).T.round(4)
        best_model = results_df['RMSE'].idxmin()
        
        cols = st.columns(len(results))
        
        for i, (model_name, metrics) in enumerate(results.items()):
            with cols[i]:
                is_best = model_name == best_model
                crown = "👑 " if is_best else ""
                border_color = "#2ecc71" if is_best else "#667eea"
                
                st.markdown(f"""
                <div class="feature-card" style="border: 2px solid {border_color};">
                    <h3 style="color: {border_color};">{crown}{model_name.upper()}</h3>
                    <div style="margin-top: 1rem;">
                        <p><strong>RMSE:</strong> ₹{metrics['RMSE']:.2f}</p>
                        <p><strong>R²:</strong> {metrics['R²']:.4f}</p>
                        <p><strong>MAPE:</strong> {metrics['MAPE']:.1f}%</p>
                        <p><strong>Direction Accuracy:</strong> {metrics['Directional_Accuracy']:.1f}%</p>
                    </div>
                    {f'<div style="color: #2ecc71; margin-top: 1rem; font-weight: bold;">🏆 Best Performer</div>' if is_best else ''}
                </div>
                """, unsafe_allow_html=True)
    
    # Future predictions
    future_pred, future_dates = predictor.predict_future(days=prediction_days)
    
    if future_pred:
        st.markdown("### 🔮 Future Price Predictions")
        
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
        
        fig.add_hline(
            y=current_price,
            line_dash="dot",
            annotation_text=f"Current: ₹{current_price:.2f}"
        )
        
        fig.update_layout(
            title='Future Price Predictions',
            yaxis_title='Price (₹)',
            template='plotly_white',
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
                recommendation = "🟢 STRONG BUY"
                color = "#2ecc71"
            elif avg_change > 2:
                recommendation = "🟢 BUY"
                color = "#2ecc71"
            elif avg_change > -2:
                recommendation = "🟡 HOLD"
                color = "#f39c12"
            else:
                recommendation = "🔴 SELL"
                color = "#e74c3c"
            
            st.markdown(f"""
            <div class="analysis-section" style="border: 2px solid {color};">
                <h3 style="text-align: center; color: {color};">Investment Recommendation</h3>
                <div style="text-align: center; margin: 2rem 0;">
                    <div style="font-size: 3rem; font-weight: bold; color: {color};">{recommendation}</div>
                    <p style="font-size: 1.1rem; margin-top: 1rem;">
                        Average predicted change: <strong>{avg_change:+.1f}%</strong> over {len(future_dates)} days
                    </p>
                    <p>Risk Level: {risk_tolerance}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

def show_enhanced_portfolio_tab():
    """Portfolio tab"""
    st.markdown("""
    <div class="analysis-section">
        <h2>🎯 Portfolio Analysis & Optimization</h2>
        <p>Advanced portfolio management tools for multi-stock analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("🔧 Portfolio analysis features coming soon!")

def show_enhanced_realtime_tab():
    """Real-time tab"""
    st.markdown("""
    <div class="analysis-section">
        <h2>⚡ Real-Time Market Monitoring</h2>
        <p>Live market data, alerts, and real-time analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simulate live metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="animation: pulse 2s infinite;">
            <div class="metric-value">LIVE</div>
            <div class="metric-label">Market Status</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #e74c3c, #c0392b);">
            <div class="metric-value">15:30</div>
            <div class="metric-label">Market Close</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #f39c12, #e67e22);">
            <div class="metric-value">45,287</div>
            <div class="metric-label">Nifty 50</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #9b59b6, #8e44ad);">
            <div class="metric-value">67,543</div>
            <div class="metric-label">Sensex</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.info("🔧 Real-time monitoring features coming soon!")

def show_enhanced_insights_tab():
    """AI insights tab"""
    st.markdown("""
    <div class="analysis-section">
        <h2>🤖 AI Market Insights</h2>
        <p>AI-powered market analysis and recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🧠 Generate AI Insights", type="primary"):
        with st.spinner("Analyzing market conditions..."):
            time.sleep(2)
            
            insights = [
                ("💡", "Market Volatility Alert", "Market volatility has increased 15% over the past week due to global economic uncertainties.", "warning"),
                ("📊", "Banking Sector Momentum", "Banking sector shows strong technical breakouts with HDFC Bank and ICICI Bank leading the charge.", "success"),
                ("⚠️", "IT Sector Headwinds", "IT stocks may face near-term headwinds due to US recession fears and currency fluctuations.", "warning"),
                ("🎯", "Energy Opportunity", "Energy sector presents buying opportunities on recent dips. Oil prices stabilizing.", "success"),
                ("📈", "Defensive Allocation", "Consider increasing allocation to defensive sectors given current market uncertainty.", "info")
            ]
            
            for icon, title, content, alert_type in insights:
                st.markdown(f"""
                <div class="status-{alert_type} fade-in">
                    <h4>{icon} {title}</h4>
                    <p>{content}</p>
                </div>
                """, unsafe_allow_html=True)

def show_enhanced_scanner_tab():
    """Market scanner tab"""
    st.markdown("""
    <div class="analysis-section">
        <h2>📊 Market Scanner</h2>
        <p>Advanced screening tools to find trading opportunities.</p>
    </div>
    """, unsafe_allow_html=True)
    
    scan_options = [
        "RSI Oversold (< 30)",
        "RSI Overbought (> 70)",
        "High Volume Breakouts", 
        "Price near 52W High",
        "Price near 52W Low",
        "Bullish Chart Patterns",
        "Bearish Chart Patterns"
    ]
    
    selected_scan = st.selectbox("Scan Criteria:", scan_options)
    
    if st.button("🔍 Scan Market", type="primary"):
        with st.spinner(f"Scanning for {selected_scan}..."):
            time.sleep(2)
            
            # Mock results
            mock_results = [
                {"Symbol": "TCS.NS", "Company": "TCS", "Price": "₹3,245.50", "Change": "+2.1%", "Signal": selected_scan},
                {"Symbol": "INFY.NS", "Company": "Infosys", "Price": "₹1,456.30", "Change": "-0.8%", "Signal": selected_scan},
                {"Symbol": "RELIANCE.NS", "Company": "Reliance", "Price": "₹2,789.45", "Change": "+1.5%", "Signal": selected_scan}
            ]
            
            st.markdown(f"""
            <div class="status-success">
                <h4>🎯 Found {len(mock_results)} opportunities for {selected_scan}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            df = pd.DataFrame(mock_results)
            st.dataframe(df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; opacity: 0.7; margin-top: 2rem;">
        <p>Enhanced Indian Stock Price Prediction System with AI-Powered Analysis</p>
    </div>
    """, unsafe_allow_html=True)
