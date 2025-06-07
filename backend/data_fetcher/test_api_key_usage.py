#!/usr/bin/env python3
"""
Test script to verify CoinGecko API key is being used correctly
"""

import requests
import os
from datetime import datetime, timedelta

def test_api_key_usage():
    api_key = os.environ.get('COINGECKO_API_KEY', '')
    
    if not api_key:
        print("ERROR: No API key found in environment variables")
        return
    
    print(f"API Key found: {api_key[:8]}...")
    print("=" * 60)
    
    # Test different header formats
    header_formats = [
        {'x-cg-demo-api-key': api_key},  # Demo/Free tier
        {'x-cg-pro-api-key': api_key},   # Pro tier
        {'X-Cg-Demo-Api-Key': api_key},  # Different case
        {'X-CG-DEMO-API-KEY': api_key},  # All caps
    ]
    
    base_url = "https://api.coingecko.com/api/v3"
    
    # Test 1: Check which header format works
    print("\nTest 1: Testing different header formats")
    print("-" * 40)
    
    for i, headers in enumerate(header_formats):
        header_name = list(headers.keys())[0]
        print(f"\nTrying header: {header_name}")
        
        # Test with ping endpoint
        response = requests.get(f"{base_url}/ping", headers=headers)
        print(f"  Ping status: {response.status_code}")
        
        # Test with simple/price endpoint to check rate limits
        response = requests.get(
            f"{base_url}/simple/price",
            params={'ids': 'bitcoin', 'vs_currencies': 'usd'},
            headers=headers
        )
        print(f"  Price check status: {response.status_code}")
        
        # Check rate limit headers
        if 'x-ratelimit-limit' in response.headers:
            print(f"  Rate limit: {response.headers.get('x-ratelimit-limit')}")
            print(f"  Remaining: {response.headers.get('x-ratelimit-remaining')}")
    
    # Test 2: Check historical data limits with the correct header
    print("\n\nTest 2: Testing historical data access")
    print("-" * 40)
    
    # Use the most likely correct header
    headers = {'x-cg-demo-api-key': api_key}
    
    test_days = [90, 180, 365, 730, 'max']
    
    for days in test_days:
        print(f"\nTesting {days} days of historical data:")
        
        url = f"{base_url}/coins/bitcoin/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': str(days),
            'interval': 'daily'
        }
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'prices' in data:
                    actual_days = len(data['prices'])
                    
                    # Get date range
                    if actual_days > 0:
                        start_date = datetime.fromtimestamp(data['prices'][0][0]/1000)
                        end_date = datetime.fromtimestamp(data['prices'][-1][0]/1000)
                        date_range = (end_date - start_date).days
                        
                        print(f"  ✓ Success: Got {actual_days} data points")
                        print(f"  Date range: {start_date.date()} to {end_date.date()} ({date_range} days)")
                    else:
                        print(f"  ✗ No data returned")
                else:
                    print(f"  ✗ Invalid response format")
            else:
                print(f"  ✗ Failed with status {response.status_code}")
                if response.status_code == 429:
                    print("    Rate limit exceeded")
                elif response.status_code == 401:
                    print("    Unauthorized - API key may be invalid")
        
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    # Test 3: Compare with and without API key
    print("\n\nTest 3: Comparing with and without API key")
    print("-" * 40)
    
    # Without API key
    print("\nWithout API key:")
    response = requests.get(
        f"{base_url}/coins/bitcoin/market_chart",
        params={'vs_currency': 'usd', 'days': '365', 'interval': 'daily'},
        headers={'Accept': 'application/json'}
    )
    
    if response.status_code == 200:
        data = response.json()
        if 'prices' in data:
            print(f"  Days available: {len(data['prices'])}")
    else:
        print(f"  Status: {response.status_code}")
    
    # With API key
    print("\nWith API key:")
    response = requests.get(
        f"{base_url}/coins/bitcoin/market_chart",
        params={'vs_currency': 'usd', 'days': '365', 'interval': 'daily'},
        headers=headers
    )
    
    if response.status_code == 200:
        data = response.json()
        if 'prices' in data:
            print(f"  Days available: {len(data['prices'])}")
    else:
        print(f"  Status: {response.status_code}")
    
    # Test 4: Check account/rate limit info if available
    print("\n\nTest 4: Checking API key tier")
    print("-" * 40)
    
    # Some API endpoints that might reveal tier info
    test_endpoints = [
        '/coins/list',
        '/global',
    ]
    
    for endpoint in test_endpoints:
        response = requests.get(f"{base_url}{endpoint}", headers=headers)
        if 'x-ratelimit-limit' in response.headers:
            print(f"\nEndpoint {endpoint}:")
            print(f"  Rate limit: {response.headers.get('x-ratelimit-limit')}")
            
            # Higher rate limits usually indicate a paid tier
            limit = response.headers.get('x-ratelimit-limit', '0')
            try:
                if int(limit) > 50:
                    print("  → Appears to be using API key (higher rate limit)")
                else:
                    print("  → May be using public tier")
            except:
                pass

if __name__ == "__main__":
    test_api_key_usage()