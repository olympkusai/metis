"""Batch calculator for concurrent processing."""
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .types import Candle, FeatureResult, IndicatorResult, FeatureCalculator, IndicatorCalculator


class BatchCalculator:
    """Handles batch processing of calculations with concurrency control."""
    
    def __init__(self, max_concurrent: int = 5, timeout: float = 30.0):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
    
    def calculate_batch(
        self,
        candles: list[Candle],
        calculators: list[FeatureCalculator | IndicatorCalculator]
    ) -> tuple[dict[str, list[FeatureResult]], dict[str, list[IndicatorResult]]]:
        """Calculate multiple features/indicators in batch with concurrency control.
        
        Args:
            candles: List of candles to calculate on
            calculators: List of calculator instances
            
        Returns:
            Tuple of (feature_results, indicator_results) dictionaries
        """
        feature_results: dict[str, list[FeatureResult]] = {}
        indicator_results: dict[str, list[IndicatorResult]] = {}
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = {
                executor.submit(self._calculate_single, calc, candles): calc
                for calc in calculators
            }
            
            for future in as_completed(futures, timeout=self.timeout):
                calc = futures[future]
                try:
                    result = future.result()
                    if isinstance(result, list) and result:
                        if isinstance(result[0], FeatureResult):
                            feature_results[calc.name()] = result
                        elif isinstance(result[0], IndicatorResult):
                            indicator_results[calc.name()] = result
                except Exception as e:
                    print(f"Error calculating {calc.name()}: {e}")
        
        return feature_results, indicator_results
    
    def _calculate_single(
        self,
        calculator: FeatureCalculator | IndicatorCalculator,
        candles: list[Candle]
    ) -> list[FeatureResult] | list[IndicatorResult]:
        """Calculate a single calculator."""
        return calculator.calculate(candles)


def default_batch_calculator() -> BatchCalculator:
    """Create a batch calculator with default settings."""
    return BatchCalculator(max_concurrent=5, timeout=30.0)


async def calculate_batch_async(
    candles: list[Candle],
    calculators: list[FeatureCalculator | IndicatorCalculator],
    max_concurrent: int = 5
) -> tuple[dict[str, list[FeatureResult]], dict[str, list[IndicatorResult]]]:
    """Async version of batch calculation.
    
    Args:
        candles: List of candles to calculate on
        calculators: List of calculator instances
        max_concurrent: Maximum concurrent calculations
        
    Returns:
        Tuple of (feature_results, indicator_results) dictionaries
    """
    feature_results: dict[str, list[FeatureResult]] = {}
    indicator_results: dict[str, list[IndicatorResult]] = {}
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def calculate_with_semaphore(calc: FeatureCalculator | IndicatorCalculator):
        async with semaphore:
            # Run synchronous calculation in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, calc.calculate, candles)
    
    tasks = [calculate_with_semaphore(calc) for calc in calculators]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for calc, result in zip(calculators, results):
        if isinstance(result, Exception):
            print(f"Error calculating {calc.name()}: {result}")
            continue
        
        if result and isinstance(result[0], FeatureResult):
            feature_results[calc.name()] = result
        elif result and isinstance(result[0], IndicatorResult):
            indicator_results[calc.name()] = result
    
    return feature_results, indicator_results
