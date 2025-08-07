"""
Enhanced Stock Prediction App – Streamlit Version
Converted from your HTML/CSS/JS design
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Your own back-end modules
# (make sure these .py files are in the same folder or PYTHONPATH)
from stock_predictor import IndianStockPredictor
from ensemble_models import EnsemblePredictor
from realtime_data_feed import RealTimeDataFeed, AlertSystem


# ──────────────────────────────────────────────────────────────────────────────
# 1.  DEFINE **ALL** FUNCTIONS FIRST
# ──────────────────────────────────────────────────────────────────────────────
# 1-A  Reusable CSS loader
def load_custom_css() -> None:
    """Inject the long CSS block only once."""
    st.markdown(
        """
        <style>
        :root{
          --primary-color:#667eea;
          --secondary-color:#764ba2;
          --accent-color:#f093fb;
          --success-color:#4facfe;
          --warning-color:#fa709a;
          --error-color:#ff6b6b;
          --info-color:#4ecdc4;
        }
        /*  Header  */
        .main-header{
          font-size:3.5rem;font-weight:800;
          background:linear-gradient(135deg,var(--primary-color),var(--secondary-color));
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
          text-align:center;margin-bottom:1rem;line-height:1.2;
        }
        .subtitle{font-size:1.25rem;color:#4a5568;text-align:center;margin-bottom:2rem;}
        /*  Metric cards  */
        .metric-card{
          background:linear-gradient(135deg,var(--success-color),#00f2fe);
          padding:2rem;border-radius:16px;color:white;text-align:center;
          box-shadow:0 10px 25px rgba(0,0,0,.1);margin:1rem 0;
          transition:.3s ease;
        }
        .metric-card:hover{transform:translateY(-4px);box-shadow:0 20px 40px rgba(0,0,0,.2);}
        .metric-value{font-size:2.5rem;font-weight:800;margin-bottom:.5rem}
        .metric-label{font-size:.875rem;opacity:.9;text-transform:uppercase;letter-spacing:.05em}
        /*  Feature cards  */
        .feature-card{
          background:linear-gradient(135deg,rgba(102,126,234,.1),rgba(118,75,162,.1));
          padding:2rem;border-radius:16px;border:1px solid rgba(102,126,234,.2);
          transition:.3s ease;text-align:center;margin:1rem 0;
        }
        .feature-card:hover{
          transform:translateY(-8px);
          box-shadow:0 20px 40px rgba(102,126,234,.2);
          border-color:var(--primary-color);
        }
        .feature-icon{font-size:3rem;margin-bottom:1rem;display:block;}
        /*  Analysis wrapper  */
        .analysis-section{
          background:linear-gradient(135deg,rgba(102,126,234,.05),rgba(118,75,162,.05));
          border-radius:16px;padding:2rem;margin:2rem 0;
          border:1px solid rgba(102,126,234,.1);
        }
        /*  Status badges  */
        .status-success{background:rgba(79,172,254,.1);color:var(--success-color);
                        border:1px solid rgba(79,172,254,.3);padding:1rem;border-radius:8px;margin:1rem 0;}
        .status-warning{background:rgba(247,112,154,.1);color:var(--warning-color);
                        border:1px solid rgba(247,112,154,.3);padding:1rem;border-radius:8px;margin:1rem 0;}
        .status-info{background:rgba(78,205,196,.1);color:var(--info-color);
                     border:1px solid rgba(78,205,196,.3);padding:1rem;border-radius:8px;margin:1rem 0;}
        /*  Buttons  */
        .stButton>button{
          background:linear-gradient(135deg,var(--primary-color),var(--secondary-color));
          color:#fff;border:none;border-radius:25px;padding:.75rem 2rem;font-weight:bold;
          font-size:1rem;box-shadow:0 4px 15px rgba(102,126,234,.3);transition:.3s ease;
        }
        .stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(102,126,234,.4);}
        /*  Simple fade-in  */
        @keyframes fadeIn{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);} }
        .fade-in{animation:fadeIn .6s ease-out;}
        /*  Remove default Streamlit chrome  */
        #MainMenu,header,footer{visibility:hidden;}
        /*  Mobile tweaks  */
        @media(max-width:768px){
          .main-header{font-size:2.5rem;}
          .metric-card{padding:1rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# 1-B  Welcome screen
def show_enhanced_welcome_screen() -> None:
    """Initial landing view."""
    st.markdown(
        """
        <div class="analysis-section fade-in">
          <h2 style="text-align:center;">🎯 Welcome to Advanced Stock Prediction!</h2>
          <p style="font-size:1.1rem;text-align:center;margin-bottom:2rem;">
            Harness the power of AI to predict Indian stock prices with multiple machine-learning models.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """<div class="metric-card"><div class="metric-value">40+</div>
               <div class="metric-label">Available Stocks</div></div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """<div class="metric-card" style="background:linear-gradient(135deg,#fa709a,#fee140)">
               <div class="metric-value">3</div><div class="metric-label">AI Models</div></div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """<div class="metric-card" style="background:linear-gradient(135deg,#4ecdc4,#44a08d)">
               <div class="metric-value">15+</div><div class="metric-label">Sectors</div></div>""",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """<div class="metric-card" style="background:linear-gradient(135deg,#ff6b6b,#ffa726)">
               <div class="metric-value">&lt; 60s</div><div class="metric-label">Prediction Time</div></div>""",
            unsafe_allow_html=True,
        )

    # Feature cards
    st.markdown("### ✨ AI Models Available")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            """<div class="feature-card"><span class="feature-icon">📊</span>
               <h3>Linear Regression</h3><p>Fast & interpretable.</p></div>""",
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            """<div class="feature-card"><span class="feature-icon">🧠</span>
               <h3>LSTM Network</h3><p>Captures complex sequential patterns.</p></div>""",
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            """<div class="feature-card"><span class="feature-icon">📈</span>
               <h3>Prophet</h3><p>Great for seasonality & long trends.</p></div>""",
            unsafe_allow_html=True,
        )

    # Quick-start guide
    st.markdown(
        """
        <div class="analysis-section fade-in">
          <h3>🚀 Getting Started</h3>
          <div class="status-info">
            <ol style="margin-left:1.5rem;">
              <li><strong>Select Stock</strong> – pick from list or enter custom.</li>
              <li><strong>Choose Period</strong> – 3 months – 5 years.</li>
              <li><strong>Select Models</strong> – Linear, LSTM, Prophet.</li>
              <li><strong>Adjust Settings</strong> – prediction days & parameters.</li>
              <li><strong>Run Analysis</strong> – click the rocket & enjoy!</li>
            </ol>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# 1-C  Main analysis routine
def run_enhanced_analysis() -> None:
    """Fetch data, train models, show progress bar, display results."""
    stock = st.session_state.selected_stock
    period = st.session_state.selected_period
    models = st.session_state.models
    days = st.session_state.prediction_days
    epochs = st.session_state.lstm_epochs
    risk = st.session_state.risk_tolerance

    st.markdown(
        f"""
        <div class="analysis-section">
          <h2>🔍 Analyzing {stock}</h2>
          <p>Using {sum(models.values())} model(s) on {period} of history</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Fancy progress
    phases = [
        ("🚀 Initializing AI system…", 10),
        ("📡 Fetching market data…", 25),
        ("🔧 Preparing features…", 40),
        ("🤖 Training Linear Regression…", 55),
        ("🧠 Training LSTM…", 70),
        ("📈 Training Prophet…", 85),
        ("🔮 Generating predictions…", 95),
        ("✅ Complete!", 100),
    ]
    bar = st.progress(0)
    info = st.empty()
    for msg, pct in phases:
        info.markdown(f'<div class="status-info"><span>{msg}</span></div>', unsafe_allow_html=True)
        bar.progress(pct)
        time.sleep(1)
    info.markdown('<div class="status-success"><span>🎉 Analysis finished!</span></div>', unsafe_allow_html=True)
    bar.empty()

    # Back-end
    predictor = IndianStockPredictor(symbol=stock, period=period)
    if predictor.fetch_data() is None:
        st.error("Failed to download price data – check symbol or internet.")
    return
    predictor.prepare_data()       


    if models["linear_regression"]:
        predictor.train_linear_regression()
    if models["lstm"]:
        predictor.train_lstm(epochs=epochs)
    if models["prophet"]:
        predictor.train_prophet()

    display_results(predictor, days, risk)


# 1-D  Results renderer
def display_results(predictor: IndianStockPredictor, days: int, risk: str) -> None:
    """Show metrics, charts, predictions & recommendation."""
    df = predictor.data
    current = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2] if len(df) > 1 else current
    diff = current - prev
    pct = diff / prev * 100 if prev else 0

    # Stock overview
    st.markdown("### 📊 Stock Overview")
    o1, o2, o3, o4 = st.columns(4)
    c_col = "#2ecc71" if diff >= 0 else "#e74c3c"
    with o1:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-value">₹{current:.2f}</div>
                <div class="metric-label">Price</div>
                <div style="color:{c_col};font-size:.9rem;margin-top:.4rem;">
                {diff:+.2f} ({pct:+.2f}%)</div></div>""",
            unsafe_allow_html=True,
        )
    with o2:
        st.markdown(
            f"""<div class="metric-card" style="background:linear-gradient(135deg,#2ecc71,#27ae60)">
                <div class="metric-value">₹{df['High'].max():.2f}</div>
                <div class="metric-label">52W High</div></div>""",
            unsafe_allow_html=True,
        )
    with o3:
        st.markdown(
            f"""<div class="metric-card" style="background:linear-gradient(135deg,#e74c3c,#c0392b)">
                <div class="metric-value">₹{df['Low'].min():.2f}</div>
                <div class="metric-label">52W Low</div></div>""",
            unsafe_allow_html=True,
        )
    with o4:
        vol = df["Close"].pct_change().std() * np.sqrt(252) * 100
        st.markdown(
            f"""<div class="metric-card" style="background:linear-gradient(135deg,#f39c12,#e67e22)">
                <div class="metric-value">{vol:.1f}%</div>
                <div class="metric-label">Volatility</div></div>""",
            unsafe_allow_html=True,
        )

    # Price history chart
    st.markdown("### 📈 Price History")
    fig = go.Figure(
        data=go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )
    fig.update_layout(template="plotly_white", height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Model metrics
    metrics = predictor.evaluate_models()
    if metrics:
        st.markdown("### 🏆 Model Performance")
        cols = st.columns(len(metrics))
        best = min(metrics, key=lambda m: metrics[m]["RMSE"])
        palette = ["#667eea", "#ff7f0e", "#2ca02c"]
        for i, (name, m) in enumerate(metrics.items()):
            crown = "👑 " if name == best else ""
            border = "#2ecc71" if name == best else palette[i % 3]
            with cols[i]:
                st.markdown(
                    f"""
                    <div class="feature-card" style="border:2px solid {border}">
                      <h3 style="color:{border};">{crown}{name.upper()}</h3>
                      <p><strong>RMSE:</strong> ₹{m['RMSE']:.2f}</p>
                      <p><strong>R²:</strong> {m['R²']:.4f}</p>
                      <p><strong>MAPE:</strong> {m['MAPE']:.2f}%</p>
                      <p><strong>Dir Acc:</strong> {m['Directional_Accuracy']:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Future predictions
    preds, fut_dates = predictor.predict_future(days)
    if preds:
        st.markdown("### 🔮 Future Price Predictions")
        f2 = go.Figure()
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for j, (model, arr) in enumerate(preds.items()):
            f2.add_scatter(
                x=fut_dates,
                y=arr,
                mode="lines+markers",
                name=model,
                line=dict(color=colors[j % 3], width=3),
            )
        f2.add_hline(y=current, line_dash="dot", annotation_text=f"Current ₹{current:.2f}")
        f2.update_layout(template="plotly_white", height=450)
        st.plotly_chart(f2, use_container_width=True)

        # Simple recommendation
        avg_chg = np.mean([(p[-1] - current) / current * 100 for p in preds.values()])
        if avg_chg > 5:
            sig, col = "🟢 STRONG BUY", "#2ecc71"
        elif avg_chg > 2:
            sig, col = "🟢 BUY", "#2ecc71"
        elif avg_chg > -2:
            sig, col = "🟡 HOLD", "#f39c12"
        else:
            sig, col = "🔴 SELL", "#e74c3c"
        st.markdown(
            f"""
            <div class="analysis-section" style="border:2px solid {col}">
              <h3 style="text-align:center;color:{col}">Investment Signal</h3>
              <p style="text-align:center;font-size:2rem;margin:.5rem 0;color:{col}">{sig}</p>
              <p style="text-align:center;">Avg predicted change {avg_chg:+.2f}% over {days} days.<br>
                 Risk Level : {risk}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# 1-E  Placeholder tabs
def portfolio_tab():
    st.info("Portfolio analysis coming soon…")


def realtime_tab():
    st.info("Real-time market feed coming soon…")


def insights_tab():
    st.info("AI insights coming soon…")


def scanner_tab():
    st.info("Market scanner coming soon…")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  PAGE CONFIG & CSS
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Stock Price Prediction – AI Powered",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_custom_css()  # inject CSS exactly once


# ──────────────────────────────────────────────────────────────────────────────
# 3.  HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="fade-in"><h1 class="main-header">📈 Indian Stock Price Prediction</h1>'
    '<p class="subtitle">AI-Powered Stock Analysis & Future Price Prediction</p></div>',
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  SIDEBAR (control panel)
# ──────────────────────────────────────────────────────────────────────────────
STOCKS = {
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "RELIANCE.NS": "Reliance Industries",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India",
    "WIPRO.NS": "Wipro",
    "TECHM.NS": "Tech Mahindra",
    "MARUTI.NS": "Maruti Suzuki",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "AXISBANK.NS": "Axis Bank",
    "BAJFINANCE.NS": "Bajaj Finance",
    "LT.NS": "Larsen & Toubro",
    "SUNPHARMA.NS": "Sun Pharma",
    "TATASTEEL.NS": "Tata Steel",
    "ONGC.NS": "ONGC",
    "COALINDIA.NS": "Coal India",
    "TITAN.NS": "Titan Company",
}

with st.sidebar:
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#667eea,#764ba2);
             border-radius:10px;padding:1rem;text-align:center;margin-bottom:1rem;">
        <h3 style="color:#fff;margin:0"><i class="fas fa-chart-line"></i> StockAI Panel</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 📊 Stock Selection")
    keyword = st.text_input("Search", "")
    filtered = {k: v for k, v in STOCKS.items() if keyword.lower() in k.lower() or keyword.lower() in v.lower()}
    symbol = st.selectbox("Symbol", list(filtered.keys()), format_func=lambda x: f"{x} – {filtered[x]}")
    custom = st.text_input("Custom symbol (opt)", "")
    symbol = custom.upper() if custom else symbol

    st.markdown("### 🗓️ History Period")
    period_map = {
        "3mo": "3 Months",
        "6mo": "6 Months",
        "1y": "1 Year",
        "2y": "2 Years",
        "5y": "5 Years",
    }
    period = st.selectbox("Look-back", list(period_map.keys()), format_func=lambda x: period_map[x], index=2)

    st.markdown("### 🤖 Models")
    use_lin = st.checkbox("📊 Linear Regression", True)
    use_lstm = st.checkbox("🧠 LSTM", True)
    use_prop = st.checkbox("📈 Prophet", True)

    st.markdown("### ⚙️ Parameters")
    days = st.slider("Days to predict", 7, 90, 30)
    epochs = st.slider("LSTM epochs", 10, 100, 50)
    risk = st.select_slider("Risk tolerance", ["Conservative", "Moderate", "Aggressive"], value="Moderate")

    st.markdown("---")
    go_btn = st.button("🚀 Start Analysis", type="primary")
    if go_btn:
        if not any([use_lin, use_lstm, use_prop]):
            st.error("Pick at least one model.")
        else:
            st.session_state.run_analysis = True
            st.session_state.selected_stock = symbol
            st.session_state.selected_period = period
            st.session_state.models = {
                "linear_regression": use_lin,
                "lstm": use_lstm,
                "prophet": use_prop,
            }
            st.session_state.prediction_days = days
            st.session_state.lstm_epochs = epochs
            st.session_state.risk_tolerance = risk
    clr = st.button("🗑️ Clear", key="clr")
    if clr:
        st.session_state.clear()


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MAIN TABS
# ──────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📈 Analysis", "🎯 Portfolio", "⚡ Real-Time", "🤖 Insights", "📊 Scanner"])

with tabs[0]:
    if st.session_state.get("run_analysis"):
        run_enhanced_analysis()
    else:
        show_enhanced_welcome_screen()

with tabs[1]:
    portfolio_tab()

with tabs[2]:
    realtime_tab()

with tabs[3]:
    insights_tab()

with tabs[4]:
    scanner_tab()


# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <hr style="margin-top:3rem;">
    <p style="text-align:center;opacity:.6">
      © 2025 Indian Stock Price Prediction System – Powered by Streamlit
    </p>
    """,
    unsafe_allow_html=True,
)
