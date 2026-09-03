import logging
from src.data_fetcher import MarketDataFetcher
from src.processor import DataProcessor
from src.db_client import MarketDBClient
from src.risk_metrics import RiskEngine

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Pipeline")

def run_pipeline():
    # 1. 配置参数
    tickers = ["AAPL", "MSFT", "SPY"] # SPY作为市场基准
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    
    # 2. 拉取原始数据
    logger.info("--- Step 1: Fetching Raw Data ---")
    raw_df = MarketDataFetcher.fetch_daily_prices(tickers, start_date, end_date)
    
    # 3. 清洗与特征工程
    logger.info("--- Step 2: Cleaning Data ---")
    clean_df = DataProcessor.clean_price_data(raw_df)
    
    # 4. 数据入库 (持久化)
    logger.info("--- Step 3: Saving to Database ---")
    db = MarketDBClient("market_data.db")
    db.insert_dataframe(clean_df, "daily_prices")
    
    # 5. 计算风险指标 (以 AAPL 为例)
    logger.info("--- Step 4: Calculating Risk Metrics ---")
    
    # 从清洗后的数据中提取 AAPL 和 SPY (市场) 的收益率
    aapl_returns = clean_df[clean_df['symbol'] == 'AAPL'].set_index('date')['daily_return']
    spy_returns = clean_df[clean_df['symbol'] == 'SPY'].set_index('date')['daily_return']
    
    # 计算 Beta
    beta = RiskEngine.calculate_beta(aapl_returns, spy_returns)
    logger.info(f"AAPL Beta relative to SPY: {beta:.4f}")
    
    # 计算 95% 历史 VaR
    var_95 = RiskEngine.calculate_historical_var(aapl_returns, confidence_level=0.95)
    logger.info(f"AAPL 95% Historical VaR (Daily): {var_95:.2%}")

if __name__ == "__main__":
    run_pipeline()