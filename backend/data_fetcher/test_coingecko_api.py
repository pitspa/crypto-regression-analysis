#!/usr/bin/env python3
"""
Test script to verify CoinGecko API connectivity and data format
"""

import requests # type: ignore
import json
import os

def test_coingecko_api():
    """Test basic CoinGecko API functionality"""
    
    api_key = os.environ.get('COINGECKO_API_KEY', '')
    
    print("Testing CoinGecko API...")
    print(f"API Key present: {'Yes' if api_key else 'No'}")
    print("-" * 50)
    
    # Test 1: Check API status
    try:
        ping_url = "https://api.coingecko.com/api/v3/ping"
        headers = {'x-cg-pro-api-key': api_key} if api_key else {}
        response = requests.get(ping_url, headers=headers)
        print(f"1. API Ping: {response.status_code}")
        if response.status_code == 200:
            print("   ✓ API is accessible")
        else:
            print("   ✗ API ping failed")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: Get top coins
    try:
        markets_url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 5,
            'page': 1
        }
        headers = {'x-cg-pro-api-key': api_key} if api_key else {}
        response = requests.get(markets_url, params=params, headers=headers)
        print(f"\n2. Top Coins: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("   ✓ Retrieved top coins:")
            for coin in data[:3]:
                print(f"     - {coin['name']} ({coin['symbol'].upper()}): ${coin['current_price']:,.2f}")
        else:
            print(f"   ✗ Failed: {response.text}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: Get historical data for Bitcoin
    try:
        btc_url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': 7,
            'interval': 'daily'
        }
        headers = {'x-cg-pro-api-key': api_key} if api_key else {}
        response = requests.get(btc_url, params=params, headers=headers)
        print(f"\n3. Bitcoin Historical: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if 'prices' in data and len(data['prices']) > 0:
                print(f"   ✓ Retrieved {len(data['prices'])} price points")
                print(f"     Latest: ${data['prices'][-1][1]:,.2f}")
            else:
                print("   ✗ No price data in response")
        else:
            print(f"   ✗ Failed: {response.text}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 4: Check rate limit headers
    try:
        print("\n4. Rate Limit Info:")
        if 'x-ratelimit-limit' in response.headers:
            print(f"   Rate Limit: {response.headers.get('x-ratelimit-limit')}")
            print(f"   Remaining: {response.headers.get('x-ratelimit-remaining')}")
        else:
            print("   No rate limit headers found (normal for public API)")
    except:
        pass
    
    print("\n" + "-" * 50)
    print("API Key Instructions:")
    print("1. Get a free API key at: https://www.coingecko.com/en/api/pricing")
    print("2. Set it as environment variable: export COINGECKO_API_KEY='your-key-here'")
    print("3. Or add it to GitHub Secrets as COINGECKO_API_KEY")

if __name__ == "__main__":
    test_coingecko_api()