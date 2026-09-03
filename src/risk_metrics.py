import pandas as pd
import numpy as np
from typing import Dict

class RiskEngine:
    """Calculates portfolio risk metrics and factor exposures."""
    
    @staticmethod
    def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculates asset Beta relative to the market."""
        # Align dates
        aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
        if len(aligned_data) < 30:
            raise ValueError("Not enough data points to calculate Beta (minimum 30 required).")
            
        cov_matrix = np.cov(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        return float(beta)

    @staticmethod
    def calculate_historical_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculates Historical Value at Risk (VaR).
        """
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1.")
            
        percentile = (1 - confidence_level) * 100
        var = np.percentile(returns.dropna(), percentile)
        return float(var)