"""Quick performance benchmark for DataBeak wrapper pattern improvements.

This is a simplified version of the full benchmark for faster testing.
"""

from __future__ import annotations

import statistics

# Need to add src to path or import from installed package
import sys
import timeit
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from databeak.models.tool_responses import (  # type: ignore[import-not-found]
    StatisticsResult,
    StatisticsSummary,
)


def generate_test_data(rows: int = 1000) -> pd.DataFrame:
    """Generate test data."""
    np.random.seed(42)

    data = {
        "sales": np.random.normal(1000, 300, rows),
        "profit": np.random.normal(150, 50, rows),
        "units": np.random.randint(1, 100, rows),
        "category": np.random.choice(["A", "B", "C"], rows),
    }

    return pd.DataFrame(data)


def old_pattern_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Simulate old pattern with extra processing layers."""
    numeric_df = df.select_dtypes(include=[np.number])

    # Multiple processing layers (old pattern overhead)
    stats_dict = {}
    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()

        # Step 1: Create intermediate dict
        temp_stats = {
            "count": int(col_data.count()),
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
        }

        # Step 2: Validation layer
        validated_stats = {k: v for k, v in temp_stats.items() if v is not None}

        # Step 3: Create Pydantic object with serialization
        pydantic_obj = StatisticsSummary(
            count=validated_stats["count"],
            mean=validated_stats["mean"],
            std=validated_stats["std"],
            min=validated_stats["min"],
            **{"25%": 0.0, "50%": 0.0, "75%": 0.0},
            max=validated_stats["max"],
        )

        # Step 4: Serialize and deserialize (wrapper pattern)
        serialized = pydantic_obj.model_dump()
        stats_dict[col] = StatisticsSummary(**serialized)

    result = StatisticsResult(
        session_id="test",
        statistics=stats_dict,
        column_count=len(stats_dict),
        numeric_columns=list(stats_dict.keys()),
        total_rows=len(df),
    ).model_dump()
    return dict(result)


def new_pattern_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Direct Pydantic creation."""
    numeric_df = df.select_dtypes(include=[np.number])

    stats = {}
    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()

        # Direct creation
        stats[col] = StatisticsSummary(
            count=int(col_data.count()),
            mean=float(col_data.mean()),
            std=float(col_data.std()),
            min=float(col_data.min()),
            **{"25%": 0.0, "50%": 0.0, "75%": 0.0},
            max=float(col_data.max()),
        )

    result = StatisticsResult(
        session_id="test",
        statistics=stats,
        column_count=len(stats),
        numeric_columns=list(stats.keys()),
        total_rows=len(df),
    ).model_dump()
    return dict(result)


def benchmark_function(func: Any, *args: Any, iterations: int = 20) -> dict[str, float]:
    """Benchmark a function."""
    times = []

    for _ in range(iterations):
        start = timeit.default_timer()
        func(*args)
        end = timeit.default_timer()
        times.append(end - start)

    return {
        "mean": statistics.mean(times),
        "std": statistics.stdev(times),
        "min": min(times),
        "max": max(times),
    }


def run_quick_benchmark() -> None:
    """Run a quick benchmark."""

    # Test with 1000 rows
    df = generate_test_data(1000)

    # Benchmark statistics
    old_results = benchmark_function(old_pattern_statistics, df)
    new_results = benchmark_function(new_pattern_statistics, df)

    improvement = ((old_results["mean"] - new_results["mean"]) / old_results["mean"]) * 100

    if 15 <= improvement <= 25 or improvement > 25:
        pass
    else:
        pass


if __name__ == "__main__":
    run_quick_benchmark()
