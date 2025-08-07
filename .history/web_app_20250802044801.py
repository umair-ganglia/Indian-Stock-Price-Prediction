import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our stock predictor module
from src.stock_predictor import load_data, preprocess_data, train_model, evaluate_model, predict_future
from src.utils import plot_predictions, save_predictions

# Set page configuration
st.set_page_config(page_title="Stock Price Prediction System", layout="wide")

# Add custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f5f5;
}
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
h1, h2, h3 {
    color: #1E3A8A;
}
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("Stock Price Prediction System")
st.markdown("""
This application uses machine learning to predict stock prices based on historical data.
Select a stock symbol, time period, and model parameters to generate predictions.
""")

# Sidebar for inputs
st.sidebar.header("Configuration")

# Stock selection
stock_options = ["RELIANCE.NS", "TCS.NS", "MARUTI.NS"]
selected_stock = st.sidebar.selectbox("Select Stock", stock_options)

# Time period selection
time_period_options = ["2y", "1y", "6mo"]
selected_time_period = st.sidebar.selectbox("Select Time Period", time_period_options)

# Model parameters
st.sidebar.subheader("Model Parameters")
feature_columns = st.sidebar.multiselect(
    "Select Features", 
    ["Open", "High", "Low", "Close", "Volume"], 
    default=["Open", "High", "Low", "Close"]
)

target_column = st.sidebar.selectbox(
    "Select Target", 
    ["Close", "Open", "High", "Low"], 
    index=0
)

model_type = st.sidebar.selectbox(
    "Select Model", 
    ["LSTM", "GRU", "SimpleRNN"], 
    index=0
)

epochs = st.sidebar.slider("Epochs", min_value=10, max_value=200, value=50, step=10)
batch_size = st.sidebar.slider("Batch Size", min_value=8, max_value=128, value=32, step=8)

# Prediction horizon
prediction_days = st.sidebar.slider("Prediction Days", min_value=1, max_value=30, value=7, step=1)

# Function to run the prediction pipeline
def run_prediction():
    try:
        # Display loading message
        with st.spinner("Loading and processing data..."):
            # Load data
            file_path = f"data/raw/{selected_stock}_{selected_time_period}_raw.csv"
            if not os.path.exists(file_path):
                st.error(f"Data file not found: {file_path}. Please download the data first.")
                return
            
            df = load_data(file_path)
            
            # Display raw data
            st.subheader("Raw Data")
            st.dataframe(df.tail())
            
            # Preprocess data
            processed_df, scaler = preprocess_data(
                df, 
                feature_columns=feature_columns, 
                target_column=target_column
            )
            
            # Split data and create sequences
            X_train, X_test, y_train, y_test, train_dates, test_dates = train_model(
                processed_df, 
                target_column=target_column,
                split_ratio=0.8,
                sequence_length=10,
                return_sequences_only=False
            )
            
            # Train model
            st.subheader("Model Training")
            with st.spinner("Training model... This may take a few minutes."):
                predictor, predictions = train_model(
                    processed_df, 
                    target_column=target_column,
                    split_ratio=0.8,
                    sequence_length=10,
                    model_type=model_type,
                    epochs=epochs,
                    batch_size=batch_size
                )
            
            # Evaluate model
            st.subheader("Model Evaluation")
            metrics = evaluate_model(predictor, X_test, y_test)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Absolute Error", f"{metrics['mae']:.4f}")
            with col2:
                st.metric("Mean Squared Error", f"{metrics['mse']:.4f}")
            with col3:
                st.metric("Root Mean Squared Error", f"{metrics['rmse']:.4f}")
            
            # Plot predictions vs actual
            st.subheader("Historical Predictions vs Actual")
            fig = plot_predictions(
                test_dates, 
                y_test, 
                predictions, 
                title=f"{selected_stock} - {target_column} Price Prediction"
            )
            st.pyplot(fig)
            
            # Future predictions
            st.subheader(f"Future {prediction_days} Days Prediction")
            with st.spinner("Generating future predictions..."):
                future_dates, future_predictions = predict_future(
                    predictor, 
                    processed_df, 
                    scaler, 
                    target_column=target_column,
                    days=prediction_days,
                    sequence_length=10
                )
                
                # Create a DataFrame for future predictions
                future_df = pd.DataFrame({
                    'Date': future_dates,
                    f'Predicted {target_column}': future_predictions.flatten()
                })
                
                # Display future predictions
                st.dataframe(future_df)
                
                # Plot future predictions
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(future_dates, future_predictions, 'r-', label=f'Predicted {target_column}')
                ax.set_title(f"{selected_stock} - Future {target_column} Price Prediction")
                ax.set_xlabel('Date')
                ax.set_ylabel(f'{target_column} Price')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                
                # Save predictions
                save_path = f"results/predictions/{selected_stock}_{target_column}_prediction.csv"
                save_predictions(future_df, save_path)
                st.success(f"Predictions saved to {save_path}")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

# Run prediction button
if st.sidebar.button("Run Prediction"):
    run_prediction()

# About section
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("""
This Stock Price Prediction System uses deep learning models to predict future stock prices based on historical data.

The system supports multiple model architectures including LSTM, GRU, and SimpleRNN.

Developed as part of a machine learning project for financial time series forecasting.
""")

# Main page instructions when no prediction is running
if 'button_clicked' not in st.session_state:
    st.info("""
    👈 Configure your prediction parameters in the sidebar and click 'Run Prediction' to start.
    
    The system will load historical data, train a model based on your selected parameters, and generate predictions for future stock prices.
    """)
    
    # Sample visualization
    st.subheader("Sample Visualization")
    sample_img = plt.figure(figsize=(10, 6))
    plt.plot(np.arange(100), np.cumsum(np.random.randn(100)), 'b-', label='Sample Stock Trend')
    plt.title("Sample Stock Price Visualization")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    st.pyplot(sample_img)