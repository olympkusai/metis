"""
Example usage of the Calculator Service gRPC client
Based on API.md use cases
"""

from app.grpc_client import CalculatorClient
from datetime import datetime, timezone


def example_calculate_returns():
    """Example: Calculate returns for BTCUSDT (API.md Example 1)"""
    print("=== Example 1: Calculate Returns ===")
    
    client = CalculatorClient(host='localhost', port=8081)
    
    try:
        # Request from API.md
        start_time = int(datetime(2024, 4, 19, 0, 0, tzinfo=timezone.utc).timestamp())
        end_time = int(datetime(2024, 4, 19, 0, 10, tzinfo=timezone.utc).timestamp())
        
        response = client.calculate(
            symbol="BTCUSDT",
            interval="1m",
            start_time=start_time,
            end_time=end_time,
            features=["returns"]
        )
        
        if response:
            print(f"Request ID: {response.request_id}")
            print(f"Symbol: {response.symbol}")
            print(f"Interval: {response.interval}")
            print(f"Features calculated: {len(response.features)}")
            for feature in response.features:
                print(f"  - {feature.name}: {len(feature.values)} values")
    finally:
        client.close()


def example_calculate_moving_averages():
    """Example: Calculate moving averages (API.md Example 2)"""
    print("\n=== Example 2: Calculate Moving Averages ===")
    
    client = CalculatorClient(host='localhost', port=8081)
    
    try:
        start_time = int(datetime(2024, 4, 19, 0, 0, tzinfo=timezone.utc).timestamp())
        end_time = int(datetime(2024, 4, 20, 0, 0, tzinfo=timezone.utc).timestamp())
        
        response = client.calculate(
            symbol="BTCUSDT",
            interval="1h",
            start_time=start_time,
            end_time=end_time,
            features=["ma_7", "ma_21", "ma_50"]
        )
        
        if response:
            print(f"Request ID: {response.request_id}")
            print(f"Symbol: {response.symbol}")
            print(f"Features calculated: {len(response.features)}")
            for feature in response.features:
                print(f"  - {feature.name}: {len(feature.values)} values")
    finally:
        client.close()


def example_calculate_volatility():
    """Example: Calculate volatility (API.md Example 3)"""
    print("\n=== Example 3: Calculate Volatility ===")
    
    client = CalculatorClient(host='localhost', port=8081)
    
    try:
        start_time = int(datetime(2024, 4, 19, 0, 0, tzinfo=timezone.utc).timestamp())
        end_time = int(datetime(2024, 4, 19, 6, 0, tzinfo=timezone.utc).timestamp())
        
        response = client.calculate(
            symbol="ETHUSDT",
            interval="15m",
            start_time=start_time,
            end_time=end_time,
            features=["volatility_7", "volatility_21"]
        )
        
        if response:
            print(f"Request ID: {response.request_id}")
            print(f"Symbol: {response.symbol}")
            print(f"Features calculated: {len(response.features)}")
            for feature in response.features:
                print(f"  - {feature.name}: {len(feature.values)} values")
    finally:
        client.close()


def example_calculate_ewma():
    """Example: Calculate EWMA (API.md Example 4)"""
    print("\n=== Example 4: Calculate EWMA ===")
    
    client = CalculatorClient(host='localhost', port=8081)
    
    try:
        start_time = int(datetime(2024, 4, 1, 0, 0, tzinfo=timezone.utc).timestamp())
        end_time = int(datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc).timestamp())
        
        response = client.calculate(
            symbol="BTCUSDT",
            interval="1d",
            start_time=start_time,
            end_time=end_time,
            features=["ewma_30d"]
        )
        
        if response:
            print(f"Request ID: {response.request_id}")
            print(f"Symbol: {response.symbol}")
            print(f"Features calculated: {len(response.features)}")
            for feature in response.features:
                print(f"  - {feature.name}: {len(feature.values)} values")
    finally:
        client.close()


def example_calculate_rsi():
    """Example: Calculate RSI (API.md Example 5)"""
    print("\n=== Example 5: Calculate RSI ===")
    
    client = CalculatorClient(host='localhost', port=8081)
    
    try:
        start_time = int(datetime(2024, 4, 19, 0, 0, tzinfo=timezone.utc).timestamp())
        end_time = int(datetime(2024, 4, 20, 0, 0, tzinfo=timezone.utc).timestamp())
        
        response = client.calculate(
            symbol="BTCUSDT",
            interval="1h",
            start_time=start_time,
            end_time=end_time,
            indicators=["rsi_14"]
        )
        
        if response:
            print(f"Request ID: {response.request_id}")
            print(f"Symbol: {response.symbol}")
            print(f"Indicators calculated: {len(response.indicators)}")
            for indicator in response.indicators:
                print(f"  - {indicator.name}: {len(indicator.values)} values")
    finally:
        client.close()


def example_calculate_macd():
    """Example: Calculate MACD (API.md Example 6)"""
    print("\n=== Example 6: Calculate MACD ===")
    
    client = CalculatorClient(host='localhost', port=8081)
    
    try:
        start_time = int(datetime(2024, 4, 19, 0, 0, tzinfo=timezone.utc).timestamp())
        end_time = int(datetime(2024, 4, 21, 0, 0, tzinfo=timezone.utc).timestamp())
        
        response = client.calculate(
            symbol="ETHUSDT",
            interval="4h",
            start_time=start_time,
            end_time=end_time,
            indicators=["macd", "macd_signal"]
        )
        
        if response:
            print(f"Request ID: {response.request_id}")
            print(f"Symbol: {response.symbol}")
            print(f"Indicators calculated: {len(response.indicators)}")
            for indicator in response.indicators:
                print(f"  - {indicator.name}: {len(indicator.values)} values")
    finally:
        client.close()


def example_calculate_bollinger_bands():
    """Example: Calculate Bollinger Bands (API.md Example 7)"""
    print("\n=== Example 7: Calculate Bollinger Bands ===")
    
    client = CalculatorClient(host='localhost', port=8081)
    
    try:
        start_time = int(datetime(2024, 4, 19, 0, 0, tzinfo=timezone.utc).timestamp())
        end_time = int(datetime(2024, 4, 20, 0, 0, tzinfo=timezone.utc).timestamp())
        
        response = client.calculate(
            symbol="BTCUSDT",
            interval="1h",
            start_time=start_time,
            end_time=end_time,
            indicators=["bb_upper", "bb_lower", "bb_width"]
        )
        
        if response:
            print(f"Request ID: {response.request_id}")
            print(f"Symbol: {response.symbol}")
            print(f"Indicators calculated: {len(response.indicators)}")
            for indicator in response.indicators:
                print(f"  - {indicator.name}: {len(indicator.values)} values")
    finally:
        client.close()


def example_calculate_complete():
    """Example: Complete calculation with multiple features and indicators (API.md Complete Example)"""
    print("\n=== Example 8: Complete Calculation ===")
    
    client = CalculatorClient(host='localhost', port=8081)
    
    try:
        start_time = int(datetime(2024, 4, 19, 0, 0, tzinfo=timezone.utc).timestamp())
        end_time = int(datetime(2024, 4, 20, 0, 0, tzinfo=timezone.utc).timestamp())
        
        response = client.calculate(
            symbol="BTCUSDT",
            interval="1h",
            start_time=start_time,
            end_time=end_time,
            features=["returns", "ma_21", "volatility_7"],
            indicators=["rsi_14", "macd", "bb_upper", "bb_lower"]
        )
        
        if response:
            print(f"Request ID: {response.request_id}")
            print(f"Symbol: {response.symbol}")
            print(f"Interval: {response.interval}")
            print(f"Start Time: {response.start_time}")
            print(f"End Time: {response.end_time}")
            print(f"Features calculated: {len(response.features)}")
            for feature in response.features:
                print(f"  - {feature.name}: {len(feature.values)} values")
            print(f"Indicators calculated: {len(response.indicators)}")
            for indicator in response.indicators:
                print(f"  - {indicator.name}: {len(indicator.values)} values")
            print(f"Calculated at: {response.calculated_at}")
    finally:
        client.close()


if __name__ == "__main__":
    # Run all examples
    # Note: These will fail if the gRPC server is not running
    print("Calculator Service gRPC Client Examples")
    print("=" * 50)
    print("Note: Make sure the gRPC server is running on localhost:8081")
    print()
    
    try:
        example_calculate_returns()
        example_calculate_moving_averages()
        example_calculate_volatility()
        example_calculate_ewma()
        example_calculate_rsi()
        example_calculate_macd()
        example_calculate_bollinger_bands()
        example_calculate_complete()
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure the gRPC server is running and accessible.")
