#!/usr/bin/env python3
"""
Fetches historical price data for top 20 cryptocurrencies by market cap
and saves them as CSV files for Rust processing.
"""

import requests # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from datetime import datetime, timedelta
import time
import os
import json
import sys

# Add pipeline status tracking
sys.path.append(os.path.dirname(__file__))
try:
    from pipeline_status import PipelineStatus
except ImportError:
    # Fallback if status tracker not available
    class PipelineStatus:
        def update_step(self, *args, **kwargs): pass
        def add_error(self, *args, **kwargs): pass
        def add_warning(self, *args, **kwargs): pass
        def create_summary(self): return "Status tracking not available"

class CryptoDataFetcher:
    def __init__(self, output_dir="../data"):
        self.output_dir = output_dir
        self.api_key = os.environ.get('COINGECKO_API_KEY', '')
        self.status = PipelineStatus(os.path.join(output_dir, "pipeline_status.json"))
        
        # Use demo API if no key provided
        if self.api_key:
            self.base_url = "https://api.coingecko.com/api/v3"
            self.headers = {
                'x-cg-pro-api-key': self.api_key,
                'Accept': 'application/json'
            }
        else:
            # Demo API has limited functionality but works without key
            self.base_url = "https://api.coingecko.com/api/v3"
            self.headers = {
                'Accept': 'application/json'
            }
            print("Warning: No API key provided. Using public API with rate limits.")
            print("To get a free API key, visit: https://www.coingecko.com/en/api/pricing")
            self.status.add_warning("No API key provided, using public API with rate limits")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_top_cryptos(self, limit=20):
        """Get top cryptocurrencies by market cap"""
        endpoint = f"{self.base_url}/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': limit,
            'page': 1,
            'sparkline': 'false'  # Changed from False to 'false'
            # Removed price_change_percentage parameter as it might cause issues
        }
            
        try:
            print(f"Fetching from: {endpoint}")
            print(f"Parameters: {params}")
            response = requests.get(endpoint, params=params, headers=self.headers)
            print(f"Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Response content: {response.text}")
            
            response.raise_for_status()
            
            data = response.json()
            
            # Verify we got the expected data structure
            if not isinstance(data, list) or len(data) == 0:
                raise ValueError("Unexpected response format from CoinGecko")
            
            # Verify required fields exist
            required_fields = ['id', 'symbol', 'name', 'market_cap']
            for coin in data:
                for field in required_fields:
                    if field not in coin:
                        raise ValueError(f"Missing required field '{field}' in response")
            
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print("Rate limit exceeded. Please wait or use an API key.")
            elif e.response.status_code == 400:
                print(f"Bad request. URL: {e.response.url}")
                print(f"Response: {e.response.text}")
            raise
    
    def fetch_historical_data(self, coin_id, days=1825):  # 5 years = 1825 days
        """Fetch historical price data for a specific coin"""
        endpoint = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
            
        try:
            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Verify response structure
            if 'prices' not in data or not isinstance(data['prices'], list):
                raise ValueError(f"Unexpected response format for {coin_id}")
            
            if len(data['prices']) == 0:
                print(f"Warning: No price data returned for {coin_id}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            
            # Verify data types
            if df['timestamp'].dtype not in ['int64', 'float64']:
                raise ValueError(f"Unexpected timestamp format for {coin_id}")
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['date'] = df['timestamp'].dt.date
            
            # Remove duplicates and keep last price of each day
            df = df.drop_duplicates(subset=['date'], keep='last')
            df = df[['date', 'price']]
            
            # Verify we have valid price data
            if df['price'].isna().all():
                print(f"Warning: All prices are NaN for {coin_id}")
                return None
            
            return df
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limit exceeded while fetching {coin_id}. Please wait or use an API key.")
            elif e.response.status_code == 404:
                print(f"Coin {coin_id} not found")
                return None
            raise
    
    def calculate_log_returns(self, df):
        """Calculate daily log returns
        
        Log return from day t-1 to day t is: log(price[t]) - log(price[t-1])
        This is stored at index t, so we need price data from day 0 to calculate
        the log return for day 1.
        """
        df = df.copy()
        df['log_return'] = pd.Series(df['price']).apply(lambda x: np.log(x) if x > 0 else np.nan).diff()
        return df
    
    def save_data(self, coin_data, metadata):
        """Save cryptocurrency data and metadata"""
        # Save individual coin data
        for coin_id, df in coin_data.items():
            filepath = os.path.join(self.output_dir, f"{coin_id}_data.csv")
            df.to_csv(filepath, index=False)
            print(f"Saved {coin_id} data to {filepath}")
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")
        
        # Create a combined dataset for easier processing
        combined_df = pd.DataFrame()
        
        # Get the date range from Bitcoin (reference coin)
        ref_coin_id = metadata['reference_coin']
        if ref_coin_id in coin_data:
            base_dates = coin_data[ref_coin_id][['date']].copy()
            combined_df = base_dates
        else:
            print(f"Warning: Reference coin {ref_coin_id} not found in data")
            return
        
        # Add each coin's data
        for coin_id, df in coin_data.items():
            coin_df = df[['date', 'price', 'log_return']].copy()
            coin_df = coin_df.rename(columns={
                'price': f'{coin_id}_price',
                'log_return': f'{coin_id}_log_return'
            })
            
            combined_df = pd.merge(combined_df, coin_df, on='date', how='left')
        
        # Sort by date - DO NOT forward fill or backward fill
        # Keep NaN values to indicate missing data
        combined_df = combined_df.sort_values('date')
        
        combined_path = os.path.join(self.output_dir, "combined_data.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"Saved combined data to {combined_path}")
    
    def run(self):
        """Main execution function"""
        print("Starting cryptocurrency data fetch pipeline...")
        self.status.update_step("initialization", "success", "Pipeline initialized")
        
        try:
            # Step 1: Fetch top cryptocurrencies
            print("Fetching top 20 cryptocurrencies by market cap...")
            self.status.update_step("fetch_top_coins", "running")
            top_cryptos = self.get_top_cryptos(20)
            self.status.update_step("fetch_top_coins", "success", 
                                  f"Retrieved {len(top_cryptos)} coins")
            
            # Extract metadata
            metadata = {
                'fetch_date': datetime.now().isoformat(),
                'coins': [],
                'pipeline_version': '1.0.0'
            }
            
            coin_data = {}
            failed_coins = []
            
            for i, crypto in enumerate(top_cryptos):
                coin_id = crypto['id']
                symbol = crypto['symbol'].upper()
                name = crypto['name']
                market_cap = crypto['market_cap']
                
                print(f"\nFetching data for {name} ({symbol})... [{i+1}/20]")
                self.status.update_step(f"fetch_{coin_id}", "running")
                
                metadata['coins'].append({
                    'id': coin_id,
                    'symbol': symbol,
                    'name': name,
                    'market_cap': market_cap,
                    'rank': i + 1
                })
                
                try:
                    # Fetch historical data
                    df = self.fetch_historical_data(coin_id)
                    
                    if df is None:
                        self.status.update_step(f"fetch_{coin_id}", "failed", 
                                              "No data returned")
                        failed_coins.append(coin_id)
                        continue
                    
                    # Calculate log returns
                    df = self.calculate_log_returns(df)
                    
                    coin_data[coin_id] = df
                    self.status.update_step(f"fetch_{coin_id}", "success", 
                                          f"{len(df)} days of data")
                    
                    # Rate limiting - increase wait time for public API
                    if self.api_key:
                        time.sleep(0.5)
                    else:
                        time.sleep(3.0)  # Increased from 2.5 to 3.0 seconds
                    
                except Exception as e:
                    error_msg = f"Error fetching data for {coin_id}: {str(e)}"
                    print(error_msg)
                    self.status.update_step(f"fetch_{coin_id}", "failed", 
                                          error=str(e))
                    self.status.add_error(error_msg, f"fetch_{coin_id}")
                    failed_coins.append(coin_id)
                    continue
            
            # Find the coin with highest market cap (should be Bitcoin)
            metadata['reference_coin'] = metadata['coins'][0]['id']
            metadata['failed_coins'] = failed_coins
            metadata['successful_coins'] = len(coin_data)
            
            # Step 2: Save data
            print("\nSaving all data...")
            self.status.update_step("save_data", "running")
            self.save_data(coin_data, metadata)
            self.status.update_step("save_data", "success", 
                                  f"Saved data for {len(coin_data)} coins")
            
            # Final summary
            if failed_coins:
                self.status.add_warning(f"{len(failed_coins)} coins failed to fetch: {', '.join(failed_coins)}")
            
            print("\nData fetching complete!")
            print(f"Successfully fetched: {len(coin_data)} coins")
            print(f"Failed: {len(failed_coins)} coins")
            
            # Save final status
            print("\n" + "="*50)
            print(self.status.create_summary())
            print("="*50)
            
            return metadata
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.status.update_step("pipeline", "failed", error=str(e))
            self.status.add_error(error_msg, "main")
            print(f"\nERROR: {error_msg}")
            print("\n" + "="*50)
            print(self.status.create_summary())
            print("="*50)
            raise

if __name__ == "__main__":
    fetcher = CryptoDataFetcher()
    fetcher.run()