import pytest
import pandas as pd
import numpy as np
from src.risk_metrics import RiskEngine

def test_calculate_beta():
    # Construct an artificial perfectly correlated market and stock
    market_returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02] * 10)
    stock_returns = market_returns * 1.5  # Beta should be exactly 1.5
    
    beta = RiskEngine.calculate_beta(stock_returns, market_returns)
    assert np.isclose(beta, 1.5, atol=1e-5)

def test_calculate_historical_var():
    # Array from -100 to 99, 95% VaR (5th percentile) should be roughly -90
    returns = pd.Series(np.linspace(-100, 99, 200)) 
    var = RiskEngine.calculate_historical_var(returns, confidence_level=0.95)
    
    assert var < 0
    assert np.isclose(var, -90.05, atol=0.1)

def test_beta_insufficient_data():
    market = pd.Series([0.01, 0.02])
    stock = pd.Series([0.02, 0.04])
    
    # Check if the correct exception is raised
    with pytest.raises(ValueError, match="Not enough data points"):
        RiskEngine.calculate_beta(stock, market)