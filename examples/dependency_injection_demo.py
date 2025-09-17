"""Demonstration of dependency injection patterns for improved testability.

This script shows how the new dependency injection architecture improves testability and reduces
coupling between server modules and session management.
"""

import asyncio
from datetime import UTC, datetime

import pandas as pd

# Import the new service-based architecture
from databeak.models.session_service import (  # type: ignore[import-not-found]
    MockSessionManager,
    SessionServiceFactory,
)
from databeak.services.statistics_service import StatisticsService  # type: ignore[import-not-found]


async def demonstrate_before_after() -> None:
    """Demonstrate the improvements from dependency injection."""

    print("=== DataBeak Dependency Injection Demo ===\n")

    # === BEFORE: Tight coupling with global session manager ===
    print("BEFORE (Tight Coupling):")
    print("- Server functions directly call get_session_manager()")
    print("- Hard to test in isolation")
    print("- Session state is global and shared")
    print("- Difficult to mock or control for testing")
    print()

    # === AFTER: Dependency injection with testable services ===
    print("AFTER (Dependency Injection):")
    print("- Services receive session manager as constructor parameter")
    print("- Easy to test with mock session managers")
    print("- Session management is injected, not hardcoded")
    print("- Clean separation of concerns")
    print()

    # === Demonstration with Mock Data ===
    print("=== Live Demo with Mock Data ===\n")

    # Create test data
    sales_data = pd.DataFrame(
        {
            "product": ["A", "B", "C", "A", "B", "C", "A"],
            "sales": [100, 150, 200, 120, 180, 210, 110],
            "profit": [20, 30, 40, 25, 35, 42, 22],
            "region": ["North", "South", "East", "North", "South", "East", "West"],
        }
    )

    print("Sample Data:")
    print(sales_data)
    print()

    # === Demonstration 1: Service with Mock Session Manager ===
    print("1. Creating service with mock session manager...")
    mock_session_manager = MockSessionManager()
    service_factory = SessionServiceFactory(mock_session_manager)
    statistics_service = service_factory.create_service(StatisticsService)

    # Add data to mock session
    session_id = "demo_session_001"
    mock_session_manager.add_test_data(session_id, sales_data)

    print(f"   ✓ Created session: {session_id}")
    print(f"   ✓ Service type: {statistics_service.get_service_name()}")
    print()

    # === Demonstration 2: Statistics Analysis ===
    print("2. Running statistics analysis...")

    # Get overall statistics
    stats_result = await statistics_service.get_statistics(session_id)
    print(f"   ✓ Found {len(stats_result.statistics)} numeric columns")
    print(f"   ✓ Total rows analyzed: {stats_result.total_rows}")

    # Show statistics for sales column
    if "sales" in stats_result.statistics:
        sales_stats = stats_result.statistics["sales"]
        print(
            f"   ✓ Sales stats - Mean: {sales_stats.mean:.2f}, Min: {sales_stats.min}, Max: {sales_stats.max}"
        )

    print()

    # === Demonstration 3: Column-specific Analysis ===
    print("3. Column-specific analysis...")

    try:
        profit_stats = await statistics_service.get_column_statistics(session_id, "profit")
        print(f"   ✓ Profit column type: {profit_stats.data_type}")
        print(f"   ✓ Non-null values: {profit_stats.non_null_count}")
        print(f"   ✓ Profit mean: {profit_stats.statistics.mean:.2f}")
    except Exception as e:
        print(f"   ⚠ Column analysis skipped: {e}")
    print()

    # === Demonstration 4: Value Counts ===
    print("4. Value distribution analysis...")

    try:
        product_counts = await statistics_service.get_value_counts(session_id, "product")
        print(f"   ✓ Product unique values: {product_counts.unique_values}")
        print("   ✓ Product distribution:")
        for product, count in product_counts.value_counts.items():
            print(f"      - {product}: {count} sales")
    except Exception as e:
        print(f"   ⚠ Value counts skipped: {e}")
    print()

    # === Demonstration 5: Correlation Analysis ===
    print("5. Correlation analysis...")

    try:
        correlation_result = await statistics_service.get_correlation_matrix(session_id)
        print(f"   ✓ Correlation method: {correlation_result.method}")
        print(f"   ✓ Columns analyzed: {', '.join(correlation_result.columns_analyzed)}")

        if (
            "sales" in correlation_result.correlation_matrix
            and "profit" in correlation_result.correlation_matrix["sales"]
        ):
            sales_profit_corr = correlation_result.correlation_matrix["sales"]["profit"]
            print(f"   ✓ Sales-Profit correlation: {sales_profit_corr:.3f}")
    except Exception as e:
        print(f"   ⚠ Correlation analysis skipped: {e}")

    print()

    # === Demonstration 6: Multiple Independent Services ===
    print("6. Multiple independent services...")

    # Create second service with different data
    mock_session_manager_2 = MockSessionManager()
    service_factory_2 = SessionServiceFactory(mock_session_manager_2)
    statistics_service_2 = service_factory_2.create_service(StatisticsService)

    # Different dataset
    inventory_data = pd.DataFrame(
        {
            "item": ["Widget", "Gadget", "Tool", "Widget"],
            "quantity": [50, 75, 30, 45],
            "price": [10.50, 25.00, 15.75, 10.50],
        }
    )

    session_id_2 = "inventory_session"
    mock_session_manager_2.add_test_data(session_id_2, inventory_data)

    inventory_stats = await statistics_service_2.get_statistics(session_id_2)
    print(f"   ✓ Service 1 sessions: {len(mock_session_manager.sessions)}")
    print(f"   ✓ Service 2 sessions: {len(mock_session_manager_2.sessions)}")
    print(f"   ✓ Service 2 analyzed {inventory_stats.total_rows} inventory items")
    print("   ✓ Services are completely independent!")
    print()

    # === Benefits Summary ===
    print("=== Benefits Achieved ===")
    print("✓ Testability: Services can be tested with mock data")
    print("✓ Isolation: Multiple services don't interfere with each other")
    print("✓ Flexibility: Easy to inject different session management implementations")
    print("✓ Maintainability: Clean separation between business logic and infrastructure")
    print("✓ Backwards Compatibility: Existing server functions still work")
    print()

    # === Error Handling Demo ===
    print("7. Error handling demonstration...")

    try:
        await statistics_service.get_statistics("nonexistent_session")
    except Exception as e:
        print(f"   ✓ Proper error handling: {e}")

    try:
        await statistics_service.get_column_statistics(session_id, "nonexistent_column")
    except Exception as e:
        print(f"   ✓ Column validation: {e}")

    print()
    print("=== Demo Complete ===")
    print("The dependency injection pattern provides better testability,")
    print("maintainability, and separation of concerns while maintaining")
    print("full backwards compatibility with existing DataBeak functionality.")


def demonstrate_testing_patterns() -> None:
    """Show how the new architecture improves testing."""

    print("\n=== Testing Patterns Demo ===\n")

    print("OLD Testing Approach:")
    print("- Tests must set up full session management infrastructure")
    print("- Global state can cause test interference")
    print("- Hard to isolate business logic from infrastructure")
    print("- Requires complex mocking of global functions")
    print()

    print("NEW Testing Approach:")
    print("- Create MockSessionManager for each test")
    print("- Inject mock into service via constructor")
    print("- Test business logic in complete isolation")
    print("- No global state or complex setup required")
    print()

    # Show simple test setup
    print("Simple Test Setup Example:")
    print("""
    def test_statistics_calculation():
        # Create isolated test environment
        mock_manager = MockSessionManager()
        service = StatisticsService(mock_manager)

        # Add test data
        test_data = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
        mock_manager.add_test_data("test", test_data)

        # Test business logic directly
        result = await service.get_statistics("test")
        assert result.statistics['values'].mean == 3.0
    """)

    print("This approach makes tests:")
    print("✓ Faster (no infrastructure setup)")
    print("✓ More reliable (no shared state)")
    print("✓ Easier to understand (focused on business logic)")
    print("✓ Better isolated (each test is independent)")


if __name__ == "__main__":
    """Run the demonstration."""
    print(f"Starting DataBeak Dependency Injection Demo at {datetime.now(UTC)}")
    print()

    # Run the main demonstration
    asyncio.run(demonstrate_before_after())

    # Show testing benefits
    demonstrate_testing_patterns()

    print(f"\nDemo completed at {datetime.now(UTC)}")
