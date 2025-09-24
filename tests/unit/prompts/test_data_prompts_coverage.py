"""Comprehensive coverage tests for data_prompts module."""

from databeak.prompts.data_prompts import (
    analyze_csv_prompt,
    data_cleaning_prompt,
    suggest_transformations_prompt,
)


class TestDataPromptsCoverage:
    """Test all prompt generation functions for coverage."""

    def test_analyze_csv_prompt_basic(self) -> None:
        """Test basic CSV analysis prompt generation."""
        session_id = "test_session_123"
        analysis_type = "correlation"

        result = analyze_csv_prompt(session_id, analysis_type)

        assert isinstance(result, str)
        assert session_id in result
        assert analysis_type in result
        assert "Analyze CSV data" in result

    def test_analyze_csv_prompt_various_types(self) -> None:
        """Test prompt generation with different analysis types."""
        session_id = "session_456"

        test_cases = [
            "statistical_summary",
            "outlier_detection",
            "data_quality",
            "pattern_analysis",
            "",  # edge case: empty analysis type
        ]

        for analysis_type in test_cases:
            result = analyze_csv_prompt(session_id, analysis_type)
            assert isinstance(result, str)
            assert session_id in result

    def test_suggest_transformations_prompt_basic(self) -> None:
        """Test basic transformation suggestions prompt."""
        session_id = "transform_session"
        goal = "normalize data for machine learning"

        result = suggest_transformations_prompt(session_id, goal)

        assert isinstance(result, str)
        assert session_id in result
        assert goal in result
        assert "Suggest transformations" in result

    def test_suggest_transformations_prompt_various_goals(self) -> None:
        """Test transformation prompt with different goals."""
        session_id = "session_789"

        test_goals = [
            "data cleaning",
            "feature engineering",
            "aggregation and grouping",
            "data visualization prep",
            "",  # edge case: empty goal
        ]

        for goal in test_goals:
            result = suggest_transformations_prompt(session_id, goal)
            assert isinstance(result, str)
            assert session_id in result

    def test_data_cleaning_prompt_basic(self) -> None:
        """Test basic data cleaning prompt generation."""
        session_id = "cleaning_session"
        issues = ["missing values", "duplicate rows"]

        result = data_cleaning_prompt(session_id, issues)

        assert isinstance(result, str)
        assert session_id in result
        assert "missing values" in result
        assert "duplicate rows" in result
        assert "Suggest cleaning" in result

    def test_data_cleaning_prompt_single_issue(self) -> None:
        """Test data cleaning prompt with single issue."""
        session_id = "single_issue_session"
        issues = ["outliers"]

        result = data_cleaning_prompt(session_id, issues)

        assert isinstance(result, str)
        assert session_id in result
        assert "outliers" in result

    def test_data_cleaning_prompt_multiple_issues(self) -> None:
        """Test data cleaning prompt with multiple issues."""
        session_id = "multi_issue_session"
        issues = ["null values", "data type inconsistencies", "format issues", "encoding problems"]

        result = data_cleaning_prompt(session_id, issues)

        assert isinstance(result, str)
        assert session_id in result
        for issue in issues:
            assert issue in result

    def test_data_cleaning_prompt_empty_issues(self) -> None:
        """Test data cleaning prompt with empty issues list."""
        session_id = "empty_issues_session"
        issues: list[str] = []

        result = data_cleaning_prompt(session_id, issues)

        assert isinstance(result, str)
        assert session_id in result
        # Should handle empty list gracefully

    def test_data_cleaning_prompt_special_characters(self) -> None:
        """Test data cleaning prompt with special characters in issues."""
        session_id = "special_chars_session"
        issues = ["issues with 'quotes'", "issues & symbols", "unicode: 你好"]

        result = data_cleaning_prompt(session_id, issues)

        assert isinstance(result, str)
        assert session_id in result

    def test_all_prompts_return_strings(self) -> None:
        """Test that all prompt functions return string types."""
        session_id = "type_test_session"

        # Test analyze_csv_prompt
        result1 = analyze_csv_prompt(session_id, "test")
        assert isinstance(result1, str)

        # Test suggest_transformations_prompt
        result2 = suggest_transformations_prompt(session_id, "test goal")
        assert isinstance(result2, str)

        # Test data_cleaning_prompt
        result3 = data_cleaning_prompt(session_id, ["test issue"])
        assert isinstance(result3, str)

    def test_prompt_consistency(self) -> None:
        """Test that prompt functions are consistent in format."""
        session_id = "consistency_test"

        # All prompts should include session ID
        prompts = [
            analyze_csv_prompt(session_id, "analysis"),
            suggest_transformations_prompt(session_id, "goal"),
            data_cleaning_prompt(session_id, ["issue"]),
        ]

        for prompt in prompts:
            assert session_id in prompt
            assert len(prompt) > 0
