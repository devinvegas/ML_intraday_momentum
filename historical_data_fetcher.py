import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BinanceHistoricalFetcher:
    """Fetch historical data from Binance with rate limiting and file persistence."""
    
    def __init__(self, rate_limit_delay=0.1):
        """
        Args:
            rate_limit_delay: Delay in seconds between API calls (default 0.5s = 120 requests/min)
        """
        self.rate_limit_delay = rate_limit_delay
        self.spot_base_url = "https://api.binance.com/api/v3"
        self.futures_base_url = "https://fapi.binance.com/fapi/v1"
    
    def _respect_rate_limit(self):
        """Sleep to respect rate limits."""
        time.sleep(self.rate_limit_delay)
    
    def fetch_klines_batch(self, symbol="BTCUSDT", interval="5m", start_time=None, end_time=None, limit=1000):
        """Fetch a single batch of klines."""
        url = f"{self.spot_base_url}/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            raw = response.json()
            
            if not raw:
                return None
            
            cols = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", 
                "taker_buy_base", "taker_buy_quote", "ignore"
            ]
            
            df = pd.DataFrame(raw, columns=cols)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            
            numeric_cols = ["open", "high", "low", "close", "volume", 
                          "quote_volume", "taker_buy_base", "taker_buy_quote"]
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            return df.drop('ignore', axis=1)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching klines: {e}")
            return None
    
    def fetch_funding_rates_batch(self, symbol="BTCUSDT", start_time=None, end_time=None, limit=1000):
        """Fetch a single batch of funding rates."""
        url = f"{self.futures_base_url}/fundingRate"
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            raw = response.json()
            
            if not raw:
                return None
            
            df = pd.DataFrame(raw)
            df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
            df["fundingRate"] = df["fundingRate"].astype(float)
            
            return df[["fundingTime", "fundingRate"]]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching funding rates: {e}")
            return None
    
    def fetch_historical_klines(
        self, 
        symbol="BTCUSDT", 
        interval="5m", 
        start_date=None, 
        end_date=None,
        output_file=None
    ):
        """
        Fetch historical klines from start_date to end_date, saving incrementally.
        
        Args:
            symbol: Trading pair
            interval: Candle interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            start_date: Start date (datetime or string 'YYYY-MM-DD')
            end_date: End date (datetime or string 'YYYY-MM-DD')
            output_file: Path to save CSV file
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        if not output_file:
            output_file = f"data/{symbol}_{interval}_klines.csv"
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists and load existing data
        if Path(output_file).exists():
            existing_df = pd.read_csv(output_file, parse_dates=["open_time", "close_time"])
            logger.info(f"Found existing data: {len(existing_df)} rows, latest: {existing_df['open_time'].max()}")
            
            # Start from where we left off
            start_date = existing_df["open_time"].max() + timedelta(milliseconds=1)
            all_data = [existing_df]
        else:
            all_data = []
            logger.info(f"Starting fresh download from {start_date}")
        
        current_start = start_date
        total_fetched = 0
        
        while current_start < end_date:
            logger.info(f"Fetching from {current_start}")
            
            batch_df = self.fetch_klines_batch(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_date,
                limit=1000
            )
            
            if batch_df is None or len(batch_df) == 0:
                logger.info("No more data available")
                break
            
            all_data.append(batch_df)
            total_fetched += len(batch_df)
            
            # Update current_start to the last timestamp + 1ms
            current_start = batch_df["open_time"].max() + timedelta(milliseconds=1)
            
            logger.info(f"Fetched {len(batch_df)} rows. Total: {total_fetched}. Latest: {batch_df['open_time'].max()}")
            
            # Save incrementally every 10 batches or when we're done
            if len(all_data) % 10 == 0:
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=["open_time"]).sort_values("open_time")
                combined_df.to_csv(output_file, index=False)
                logger.info(f"Saved checkpoint: {len(combined_df)} total rows")
            
            self._respect_rate_limit()
            
            # Break if we got less than limit (reached the end)
            if len(batch_df) < 1000:
                break
        
        # Final save
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["open_time"]).sort_values("open_time")
            combined_df.to_csv(output_file, index=False)
            logger.info(f"✓ Complete! Saved {len(combined_df)} rows to {output_file}")
            return combined_df
        else:
            logger.warning("No data fetched")
            return None
    
    def fetch_historical_funding_rates(
        self,
        symbol="BTCUSDT",
        start_date=None,
        end_date=None,
        output_file=None
    ):
        """
        Fetch historical funding rates from start_date to end_date.
        
        Funding rates occur every 8 hours, so this is less data than klines.
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        if not output_file:
            output_file = f"data/{symbol}_funding_rates.csv"
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists
        if Path(output_file).exists():
            existing_df = pd.read_csv(output_file, parse_dates=["fundingTime"])
            logger.info(f"Found existing data: {len(existing_df)} rows, latest: {existing_df['fundingTime'].max()}")
            start_date = existing_df["fundingTime"].max() + timedelta(milliseconds=1)
            all_data = [existing_df]
        else:
            all_data = []
            logger.info(f"Starting fresh download from {start_date}")
        
        current_start = start_date
        total_fetched = 0
        
        while current_start < end_date:
            logger.info(f"Fetching funding rates from {current_start}")
            
            batch_df = self.fetch_funding_rates_batch(
                symbol=symbol,
                start_time=current_start,
                end_time=end_date,
                limit=1000
            )
            
            if batch_df is None or len(batch_df) == 0:
                logger.info("No more data available")
                break
            
            all_data.append(batch_df)
            total_fetched += len(batch_df)
            
            current_start = batch_df["fundingTime"].max() + timedelta(milliseconds=1)
            
            logger.info(f"Fetched {len(batch_df)} rows. Total: {total_fetched}. Latest: {batch_df['fundingTime'].max()}")
            
            # Save incrementally
            if len(all_data) % 10 == 0:
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=["fundingTime"]).sort_values("fundingTime")
                combined_df.to_csv(output_file, index=False)
                logger.info(f"Saved checkpoint: {len(combined_df)} total rows")
            
            self._respect_rate_limit()
            
            if len(batch_df) < 1000:
                break
        
        # Final save
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["fundingTime"]).sort_values("fundingTime")
            combined_df.to_csv(output_file, index=False)
            logger.info(f"✓ Complete! Saved {len(combined_df)} rows to {output_file}")
            return combined_df
        else:
            logger.warning("No data fetched")
            return None


# Usage example
if __name__ == "__main__":
    fetcher = BinanceHistoricalFetcher(rate_limit_delay=0.5)
    
    # Fetch 3 months of 5-minute klines
    klines_df = fetcher.fetch_historical_klines(
        symbol="BTCUSDT",
        interval="5m",
        start_date="2022-01-01",
        end_date="2025-11-20",
        output_file="data/BTCUSDT_5m_klines.csv"
    )
    
    # Fetch funding rates
    funding_df = fetcher.fetch_historical_funding_rates(
        symbol="BTCUSDT",
        start_date="2022-01-01",
        end_date="2025-11-20",
        output_file="data/BTCUSDT_funding_rates.csv"
    )