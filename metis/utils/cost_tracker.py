"""LLM cost tracking for token and dollar calculation."""

from typing import Dict, Optional
from collections import defaultdict

# Preços por modelo (em USD por 1M tokens)
# Baseado em preços da OpenAI (GPT-4o, GPT-4o-mini, etc.)
MODEL_PRICES = {
    "gpt-4o": {
        "input": 2.50,      # $2.50 per 1M input tokens
        "output": 10.00,    # $10.00 per 1M output tokens
    },
    "gpt-4o-mini": {
        "input": 0.15,      # $0.15 per 1M input tokens
        "output": 0.60,     # $0.60 per 1M output tokens
    },
    "gpt-4-turbo": {
        "input": 10.00,
        "output": 30.00,
    },
    "gpt-3.5-turbo": {
        "input": 0.50,
        "output": 1.50,
    },
}


class CostTracker:
    """Track LLM usage costs across multiple calls."""
    
    def __init__(self):
        self.calls: list[Dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
    
    def add_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        node_name: str = "unknown",
    ) -> float:
        """Add a call to the tracker and return the cost in USD."""
        # Normalize model name
        model_lower = model.lower()
        if "gpt-4o-mini" in model_lower:
            price_key = "gpt-4o-mini"
        elif "gpt-4o" in model_lower:
            price_key = "gpt-4o"
        elif "gpt-4-turbo" in model_lower:
            price_key = "gpt-4-turbo"
        elif "gpt-3.5" in model_lower:
            price_key = "gpt-3.5-turbo"
        else:
            price_key = "gpt-4o"  # Default
        
        prices = MODEL_PRICES.get(price_key, MODEL_PRICES["gpt-4o"])
        
        # Calculate cost
        input_cost = (input_tokens / 1_000_000) * prices["input"]
        output_cost = (output_tokens / 1_000_000) * prices["output"]
        total_cost = input_cost + output_cost
        
        # Update totals
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += total_cost
        
        # Record call
        self.calls.append({
            "model": model,
            "node": node_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": total_cost,
        })
        
        return total_cost
    
    def get_summary(self) -> Dict:
        """Get a summary of all costs."""
        return {
            "total_calls": len(self.calls),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "calls_by_node": self._get_calls_by_node(),
            "calls_by_model": self._get_calls_by_model(),
        }
    
    def _get_calls_by_node(self) -> Dict:
        """Get calls grouped by node."""
        by_node = defaultdict(lambda: {"count": 0, "tokens": 0, "cost": 0.0})
        for call in self.calls:
            by_node[call["node"]]["count"] += 1
            by_node[call["node"]]["tokens"] += call["input_tokens"] + call["output_tokens"]
            by_node[call["node"]]["cost"] += call["cost_usd"]
        return {k: {"count": v["count"], "tokens": v["tokens"], "cost_usd": round(v["cost"], 6)} 
                for k, v in by_node.items()}
    
    def _get_calls_by_model(self) -> Dict:
        """Get calls grouped by model."""
        by_model = defaultdict(lambda: {"count": 0, "tokens": 0, "cost": 0.0})
        for call in self.calls:
            by_model[call["model"]]["count"] += 1
            by_model[call["model"]]["tokens"] += call["input_tokens"] + call["output_tokens"]
            by_model[call["model"]]["cost"] += call["cost_usd"]
        return {k: {"count": v["count"], "tokens": v["tokens"], "cost_usd": round(v["cost"], 6)} 
                for k, v in by_model.items()}
    
    def reset(self):
        """Reset all tracking."""
        self.calls = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0


# Global tracker instance
_global_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker


def reset_cost_tracker():
    """Reset the global cost tracker."""
    global _global_tracker
    _global_tracker = None
