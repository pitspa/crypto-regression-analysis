#!/usr/bin/env python3
"""
Fetches historical price data for top 20 cryptocurrencies by market cap
and saves them as CSV files for Rust processing.
ONLY uses real data from CoinGecko API - no synthetic data.
"""

import requests
import pandas as pd
import numpy as np
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
        
        # Always use the public API endpoint
        self.base_url = "https://api.coingecko.com/api/v3"
        
        # Setup headers
        self.headers = {'Accept': 'application/json'}
        if self.api_key:
            # For demo/free tier API keys
            self.headers['x-cg-demo-api-key'] = self.api_key
            print(f"Using API key: {self.api_key[:8]}...")
            print("API key detected - will fetch 365 days of data")
        else:
            print("WARNING: No API key provided - data will be limited to 90 days")
            self.status.add_warning("No API key provided - historical data will be limited")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def test_api(self):
        """Test API connectivity"""
        try:
            url = f"{self.base_url}/ping"
            response = requests.get(url, headers=self.headers)
            print(f"API ping status: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            print(f"API test failed: {e}")
            return False
    
    def get_top_cryptos(self, limit=20):
        """Get top cryptocurrencies by market cap from API"""
        try:
            url = f"{self.base_url}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': str(limit),
                'page': '1',
                'sparkline': 'false',
                'locale': 'en'
            }
            
            print(f"Fetching top {limit} cryptocurrencies...")
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    print(f"Successfully fetched {len(data)} coins from API")
                    return data
                else:
                    print("Unexpected response format from API")
                    return None
            else:
                print(f"Failed to fetch coin list: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error fetching top coins: {e}")
            return None
    
    def get_fallback_coin_list(self):
        """Fallback list of major cryptocurrencies if API fails"""
        # Only use this if the API completely fails
        return [
            {'id': 'bitcoin', 'symbol': 'btc', 'name': 'Bitcoin', 'market_cap': 1000000000000},
            {'id': 'ethereum', 'symbol': 'eth', 'name': 'Ethereum', 'market_cap': 500000000000},
            {'id': 'tether', 'symbol': 'usdt', 'name': 'Tether', 'market_cap': 100000000000},
            {'id': 'binancecoin', 'symbol': 'bnb', 'name': 'BNB', 'market_cap': 90000000000},
            {'id': 'solana', 'symbol': 'sol', 'name': 'Solana', 'market_cap': 80000000000},
        ]
    
    def fetch_historical_data(self, coin_id, days=365):
        """Fetch REAL historical data using market_chart endpoint"""
        try:
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            
            # Always try to get 365 days with API key, 90 without
            if self.api_key:
                params = {
                    'vs_currency': 'usd',
                    'days': '365',  # Free tier supports up to 365 days
                    'interval': 'daily'
                }
            else:
                params = {
                    'vs_currency': 'usd',
                    'days': '90',  # Limited without API key
                    'interval': 'daily'
                }
            
            print(f"  Fetching {coin_id} data (days={params['days']})...", end='', flush=True)
            
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'prices' in data and len(data['prices']) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['date'] = df['timestamp'].dt.date
                    
                    # Remove duplicates, keeping last price of each day
                    df = df.drop_duplicates(subset=['date'], keep='last')
                    df = df[['date', 'price']].sort_values('date')
                    
                    # Remove any invalid prices
                    df = df[df['price'] > 0]
                    
                    days_fetched = len(df)
                    date_range = f"{df['date'].min()} to {df['date'].max()}"
                    print(f" ✓ {days_fetched} days ({date_range})")
                    
                    return df
                else:
                    print(f" ✗ No price data returned")
                    return None
                    
            elif response.status_code == 429:
                print(f" ✗ Rate limit exceeded")
                self.status.add_error(f"Rate limit exceeded for {coin_id}", f"fetch_{coin_id}")
                return None
            else:
                print(f" ✗ Error {response.status_code}")
                return None
                
        except Exception as e:
            print(f" ✗ Exception: {str(e)}")
            self.status.add_error(f"Failed to fetch {coin_id}: {str(e)}", f"fetch_{coin_id}")
            return None
    
    def calculate_log_returns(self, df):
        """Calculate daily log returns"""
        df = df.copy()
        # Calculate log returns
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        # First value will be NaN, which is correct
        return df
    
    def save_data(self, coin_data, metadata):
        """Save cryptocurrency data and metadata"""
        if not coin_data:
            raise Exception("No coin data to save!")
        
        # Save individual coin data
        for coin_id, df in coin_data.items():
            filepath = os.path.join(self.output_dir, f"{coin_id}_data.csv")
            df.to_csv(filepath, index=False)
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\nSaved metadata to {metadata_path}")
        
        # Find common date range for combined data
        all_dates = set()
        min_dates = []
        max_dates = []
        
        for coin_id, df in coin_data.items():
            all_dates.update(df['date'].tolist())
            min_dates.append(df['date'].min())
            max_dates.append(df['date'].max())
        
        # Report data coverage
        earliest_date = min(min_dates)
        latest_date = max(max_dates)
        total_days = (latest_date - earliest_date).days + 1
        print(f"\nData coverage: {earliest_date} to {latest_date}")
        print(f"Total date range: {total_days} days")
        print(f"Total unique dates across all coins: {len(all_dates)}")
        
        # Create combined DataFrame
        all_dates = sorted(list(all_dates))
        combined_df = pd.DataFrame({'date': all_dates})
        
        # Add each coin's data
        for coin_id, df in coin_data.items():
            coin_df = df[['date', 'price', 'log_return']].copy()
            coin_df = coin_df.rename(columns={
                'price': f'{coin_id}_price',
                'log_return': f'{coin_id}_log_return'
            })
            combined_df = pd.merge(combined_df, coin_df, on='date', how='left')
        
        # Sort by date
        combined_df = combined_df.sort_values('date')
        
        # Save combined data
        combined_path = os.path.join(self.output_dir, "combined_data.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"Saved combined data: {combined_df.shape}")
        
        # Calculate actual data availability
        metadata['data_coverage'] = {
            'earliest_date': str(earliest_date),
            'latest_date': str(latest_date),
            'total_days': len(all_dates),
            'date_range_days': total_days,
            'api_tier': 'free_tier_365_days' if self.api_key else 'public_90_days'
        }
        
        # Re-save metadata with coverage info
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def run(self):
        """Main execution function"""
        print("="*60)
        print("Cryptocurrency Data Fetcher - REAL DATA ONLY")
        print("="*60)
        print(f"Output directory: {os.path.abspath(self.output_dir)}")
        self.status.update_step("initialization", "success", "Pipeline initialized")
        
        # Test API
        if not self.test_api():
            self.status.add_warning("API connectivity test failed")
        
        try:
            # Step 1: Get top cryptocurrencies
            print("\nStep 1: Fetching top cryptocurrencies by market cap...")
            self.status.update_step("fetch_top_coins", "running")
            
            top_cryptos = self.get_top_cryptos(20)
            
            if not top_cryptos:
                print("Failed to fetch from API, using minimal fallback list...")
                top_cryptos = self.get_fallback_coin_list()
                self.status.add_warning("Using fallback coin list due to API failure")
            
            self.status.update_step("fetch_top_coins", "success", 
                                  f"Retrieved {len(top_cryptos)} coins")
            
            # Extract metadata
            metadata = {
                'fetch_date': datetime.now().isoformat(),
                'coins': [],
                'pipeline_version': '2.0.0',  # Version 2: Real data only
                'data_source': 'coingecko_api',
                'api_key_used': bool(self.api_key)
            }
            
            coin_data = {}
            failed_coins = []
            successful_coins = []
            
            # Step 2: Fetch historical data
            print("\nStep 2: Fetching REAL historical data for each coin...")
            print("Note: Data range depends on API tier")
            print("-" * 60)
            
            for i, crypto in enumerate(top_cryptos):
                coin_id = crypto['id']
                symbol = crypto.get('symbol', '').upper()
                name = crypto.get('name', coin_id)
                market_cap = crypto.get('market_cap', 0)
                
                self.status.update_step(f"fetch_{coin_id}", "running")
                
                metadata['coins'].append({
                    'id': coin_id,
                    'symbol': symbol,
                    'name': name,
                    'market_cap': market_cap,
                    'rank': i + 1
                })
                
                # Fetch real historical data
                df = self.fetch_historical_data(coin_id)
                
                if df is not None and len(df) > 10:  # Need at least 10 days
                    # Calculate log returns
                    df = self.calculate_log_returns(df)
                    coin_data[coin_id] = df
                    successful_coins.append(coin_id)
                    self.status.update_step(f"fetch_{coin_id}", "success", 
                                          f"{len(df)} days of real data")
                else:
                    failed_coins.append(coin_id)
                    self.status.update_step(f"fetch_{coin_id}", "failed", 
                                          "Insufficient data")
                    self.status.add_error(f"Failed to fetch sufficient data for {coin_id}", 
                                        f"fetch_{coin_id}")
                
                # Rate limiting
                if i < len(top_cryptos) - 1:  # Don't sleep after last coin
                    sleep_time = 1.0 if self.api_key else 2.5
                    time.sleep(sleep_time)
            
            print("-" * 60)
            
            # Check if we have enough data
            if len(coin_data) < 2:
                raise Exception(f"Insufficient data: only {len(coin_data)} coins fetched successfully. "
                              "Need at least 2 coins including Bitcoin.")
            
            # Ensure Bitcoin is included (it's the reference coin)
            if 'bitcoin' not in coin_data:
                raise Exception("Bitcoin data is required as the reference coin but was not fetched successfully.")
            
            # Update metadata
            metadata['reference_coin'] = 'bitcoin'
            metadata['successful_coins'] = len(successful_coins)
            metadata['failed_coins'] = len(failed_coins)
            metadata['failed_coin_list'] = failed_coins
            
            # Step 3: Save data
            print(f"\nStep 3: Saving data for {len(coin_data)} coins...")
            self.status.update_step("save_data", "running")
            
            self.save_data(coin_data, metadata)
            
            self.status.update_step("save_data", "success", 
                                  f"Saved data for {len(coin_data)} coins")
            
            # Final summary
            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"Successfully fetched: {len(successful_coins)} coins")
            if failed_coins:
                print(f"Failed: {len(failed_coins)} coins ({', '.join(failed_coins)})")
                self.status.add_warning(f"Failed to fetch {len(failed_coins)} coins: {', '.join(failed_coins)}")
            
            print(f"\nData source: CoinGecko API (REAL DATA ONLY)")
            print(f"API key used: {'Yes' if self.api_key else 'No (limited to ~90 days)'}")
            
            # Save final status
            self.status.update_step("pipeline", "success", "Pipeline completed successfully")
            print("\n" + self.status.create_summary())
            print("="*60)
            
            return metadata
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.status.update_step("pipeline", "failed", error=str(e))
            self.status.add_error(error_msg, "main")
            print(f"\nERROR: {error_msg}")
            print("\n" + self.status.create_summary())
            raise

if __name__ == "__main__":
    fetcher = CryptoDataFetcher()
    fetcher.run()