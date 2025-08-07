import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf

class PortfolioAnalyzer:
    """Advanced portfolio analysis and optimization"""
    
    def __init__(self, symbols, period="2y"):
        self.symbols = symbols
        self.period = period
        self.data = {}
        self.returns = None
        
    def fetch_portfolio_data(self):
        """Fetch data for all symbols"""
        for symbol in self.symbols:
            try:
                stock = yf.Ticker(symbol)
                self.data[symbol] = stock.history(period=self.period)['Close']
            except:
                print(f"Failed to fetch data for {symbol}")
        
        # Create returns matrix
        df = pd.DataFrame(self.data).dropna()
        self.returns = df.pct_change().dropna()
        return df
    
    def optimize_portfolio(self, method='sharpe'):
        """Optimize portfolio using various methods"""
        if self.returns is None:
            self.fetch_portfolio_data()
        
        mean_returns = self.returns.mean() * 252
        cov_matrix = self.returns.cov() * 252
        
        if method == 'sharpe':
            return self._maximize_sharpe(mean_returns, cov_matrix)
        elif method == 'min_variance':
            return self._minimize_variance(cov_matrix)
        elif method == 'risk_parity':
            return self._risk_parity(cov_matrix)
    
    def calculate_var(self, weights, confidence=0.05):
        """Calculate Value at Risk"""
        portfolio_return = np.sum(self.returns.mean() * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        return portfolio_return - 1.96 * portfolio_std  # 95% VaR
    
    def monte_carlo_simulation(self, weights, days=252, simulations=10000):
        """Monte Carlo simulation for portfolio"""
        mean_returns = self.returns.mean()
        cov_matrix = self.returns.cov()
        
        results = []
        for _ in range(simulations):
            random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, days)
            portfolio_returns = np.sum(random_returns * weights, axis=1)
            results.append(np.prod(1 + portfolio_returns))
        
        return np.array(results)
