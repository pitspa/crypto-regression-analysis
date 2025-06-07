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
        
        # Use pro API if key provided
        if self.api_key:
            self.base_url = "https://pro-api.coingecko.com/api/v3"
            self.headers = {
                'x-cg-pro-api-key': self.api_key,
                'Accept': 'application/json'
            }
            print(f"Using CoinGecko Pro API with key: {self.api_key[:8]}...")
        else:
            # Public API
            self.base_url = "https://api.coingecko.com/api/v3"
            self.headers = {
                'Accept': 'application/json'
            }
            print("Warning: No API key provided. Using public API with rate limits.")
            self.status.add_warning("No API key provided, using public API with rate limits")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_top_cryptos(self, limit=20):
        """Get top cryptocurrencies by market cap"""
        endpoint = f"{self.base_url}/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': str(limit),  # Convert to string
            'page': '1',  # Convert to string
            'sparkline': 'false',  # Use string 'false' not Python False
            'locale': 'en'
        }
            
        try:
            print(f"Fetching top {limit} cryptocurrencies...")
            print(f"URL: {endpoint}")
            print(f"Params: {params}")
            
            response = requests.get(endpoint, params=params, headers=self.headers)
            print(f"Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error response: {response.text}")
                
            response.raise_for_status()
            
            data = response.json()
            
            # Verify we got the expected data structure
            if not isinstance(data, list) or len(data) == 0:
                raise ValueError("Unexpected response format from CoinGecko")
            
            print(f"Successfully fetched {len(data)} coins")
            
            # Print first coin as example
            if data:
                print(f"First coin: {data[0].get('name', 'Unknown')} ({data[0].get('symbol', '').upper()})")
            
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print("Rate limit exceeded. Please wait or use an API key.")
            elif e.response.status_code == 400:
                print(f"Bad request error. This might be due to invalid parameters.")
                print(f"Full URL: {e.response.url}")
            raise
    
    def fetch_historical_data(self, coin_id, days=365):  # Reduced from 1825 to 365 days
        """Fetch historical price data for a specific coin"""
        endpoint = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': str(days),  # Convert to string
            'interval': 'daily'
        }
            
        try:
            response = requests.get(endpoint, params=params, headers=self.headers)
            
            if response.status_code != 200:
                print(f"Error fetching {coin_id}: {response.status_code} - {response.text[:200]}")
                
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
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['date'] = df['timestamp'].dt.date
            
            # Remove duplicates and keep last price of each day
            df = df.drop_duplicates(subset=['date'], keep='last')
            df = df[['date', 'price']]
            
            # Verify we have valid price data
            if df['price'].isna().all():
                print(f"Warning: All prices are NaN for {coin_id}")
                return None
            
            print(f"Fetched {len(df)} days of data for {coin_id}")
            
            return df
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limit exceeded while fetching {coin_id}.")
            elif e.response.status_code == 404:
                print(f"Coin {coin_id} not found")
                return None
            raise
        except Exception as e:
            print(f"Error fetching {coin_id}: {str(e)}")
            raise
    
    def calculate_log_returns(self, df):
        """Calculate daily log returns"""
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
        if not coin_data:
            print("Warning: No coin data to combine!")
            return
            
        # Get the date range from the first coin with data
        ref_coin_id = metadata['reference_coin']
        if ref_coin_id not in coin_data and coin_data:
            ref_coin_id = list(coin_data.keys())[0]
            print(f"Warning: Reference coin {metadata['reference_coin']} not in data, using {ref_coin_id}")
        
        if ref_coin_id in coin_data:
            base_dates = coin_data[ref_coin_id][['date']].copy()
            combined_df = base_dates
        else:
            print(f"Error: No valid reference coin found")
            return
        
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
        
        combined_path = os.path.join(self.output_dir, "combined_data.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"Saved combined data to {combined_path}")
        print(f"Combined data shape: {combined_df.shape}")
    
    def run(self):
        """Main execution function"""
        print("Starting cryptocurrency data fetch pipeline...")
        print(f"Output directory: {os.path.abspath(self.output_dir)}")
        self.status.update_step("initialization", "success", "Pipeline initialized")
        
        try:
            # Step 1: Fetch top cryptocurrencies
            print("\n" + "="*50)
            print("Step 1: Fetching top cryptocurrencies")
            print("="*50)
            
            self.status.update_step("fetch_top_coins", "running")
            
            try:
                top_cryptos = self.get_top_cryptos(20)
            except Exception as e:
                print(f"Failed to fetch from API: {str(e)}")
                print("Using fallback list of top cryptocurrencies...")
                
                # Fallback list of top cryptos by market cap (as of 2024)
                top_cryptos = [
                    {'id': 'bitcoin', 'symbol': 'btc', 'name': 'Bitcoin', 'market_cap': 1000000000000},
                    {'id': 'ethereum', 'symbol': 'eth', 'name': 'Ethereum', 'market_cap': 500000000000},
                    {'id': 'tether', 'symbol': 'usdt', 'name': 'Tether', 'market_cap': 100000000000},
                    {'id': 'binancecoin', 'symbol': 'bnb', 'name': 'BNB', 'market_cap': 90000000000},
                    {'id': 'solana', 'symbol': 'sol', 'name': 'Solana', 'market_cap': 80000000000},
                ]
                
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
            
            print("\n" + "="*50)
            print("Step 2: Fetching historical data for each coin")
            print("="*50)
            
            for i, crypto in enumerate(top_cryptos):
                coin_id = crypto['id']
                symbol = crypto['symbol'].upper()
                name = crypto['name']
                market_cap = crypto.get('market_cap', 0)
                
                print(f"\nFetching data for {name} ({symbol})... [{i+1}/{len(top_cryptos)}]")
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
                    
                    # Rate limiting - more conservative
                    time.sleep(1.0 if self.api_key else 3.0)
                    
                except Exception as e:
                    error_msg = f"Error fetching data for {coin_id}: {str(e)}"
                    print(error_msg)
                    self.status.update_step(f"fetch_{coin_id}", "failed", 
                                          error=str(e))
                    self.status.add_error(error_msg, f"fetch_{coin_id}")
                    failed_coins.append(coin_id)
                    
                    # Don't fail the entire pipeline for individual coin failures
                    if len(coin_data) >= 2:  # Continue if we have at least 2 coins
                        continue
                    else:
                        print("Too many failures, stopping...")
                        break
            
            # Set reference coin
            if coin_data:
                # Use bitcoin if available, otherwise first successful coin
                if 'bitcoin' in coin_data:
                    metadata['reference_coin'] = 'bitcoin'
                else:
                    metadata['reference_coin'] = list(coin_data.keys())[0]
            else:
                raise Exception("No coin data was successfully fetched!")
                
            metadata['failed_coins'] = failed_coins
            metadata['successful_coins'] = len(coin_data)
            
            # Step 3: Save data
            print("\n" + "="*50)
            print("Step 3: Saving data")
            print("="*50)
            
            self.status.update_step("save_data", "running")
            self.save_data(coin_data, metadata)
            self.status.update_step("save_data", "success", 
                                  f"Saved data for {len(coin_data)} coins")
            
            # Final summary
            if failed_coins:
                self.status.add_warning(f"{len(failed_coins)} coins failed to fetch: {', '.join(failed_coins)}")
            
            print("\n" + "="*50)
            print("SUMMARY")
            print("="*50)
            print(f"Successfully fetched: {len(coin_data)} coins")
            print(f"Failed: {len(failed_coins)} coins")
            
            if coin_data:
                print(f"Reference coin: {metadata['reference_coin']}")
                print(f"Data saved to: {os.path.abspath(self.output_dir)}")
            
            # Save final status
            print("\n" + self.status.create_summary())
            
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