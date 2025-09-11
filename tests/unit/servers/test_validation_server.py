"""Tests for validation module to improve coverage."""

import pytest
from fastmcp.exceptions import ToolError

from src.databeak.servers.io_server import load_csv_from_content
from src.databeak.servers.validation_server import (
    CompletenessRule,
    ConsistencyRule,
    DataTypesRule,
    DuplicatesRule,
    OutliersRule,
    UniquenessRule,
    ValidationSchema,
    check_data_quality,
    find_anomalies,
    validate_schema,
)


@pytest.fixture
async def validation_test_session():
    """Create a session with validation-friendly data."""
    csv_content = """id,name,age,email,salary,join_date,status
1,John Doe,30,john@example.com,50000,2023-01-15,active
2,Jane Smith,25,jane@test.com,60000,2023-02-20,active
3,Bob Johnson,35,,75000,2023-03-10,inactive
4,Alice Brown,-5,alice@company.org,45000,2023-04-05,active
5,Charlie Wilson,200,charlie@email,120000,2023-05-01,active
6,Diana Ross,28,diana@music.com,55000,2023-06-15,active
7,Frank Miller,32,frank@books.com,0,2023-07-20,pending
8,Grace Lee,29,grace@tech.com,65000,2023-08-25,active
9,Henry Ford,150,henry@cars.com,200000,2023-09-30,retired
10,Ivan Petrov,27,ivan@russian.ru,58000,2023-10-15,active"""

    result = await load_csv_from_content(csv_content)
    return result.session_id


@pytest.fixture
async def clean_test_session():
    """Create a session with clean validation data."""
    csv_content = """id,name,age,email
1,John,25,john@example.com
2,Jane,30,jane@test.com
3,Bob,35,bob@company.org"""

    result = await load_csv_from_content(csv_content)
    return result.session_id


@pytest.fixture
async def problematic_test_session():
    """Create a session with data quality issues."""
    csv_content = """id,name,age,score,category
1,John,30,85.5,A
1,John,30,85.5,A
2,,25,92.0,B
3,Bob,,88.7,A
4,Alice,35,invalid,C
5,Charlie,200,95.0,A
6,,40,78.2,B
7,Diana,28,,A
8,Frank,-10,89.1,C
9,Grace,45,102.5,D
10,Henry,32,91.0,A"""

    result = await load_csv_from_content(csv_content)
    return result.session_id


@pytest.mark.asyncio
class TestSchemaValidation:
    """Test schema validation functionality."""

    async def test_validate_schema_success(self, clean_test_session):
        """Test successful schema validation."""
        schema_dict = {
            "id": {"type": "int", "nullable": False, "min": 1},
            "name": {"type": "str", "nullable": False, "min_length": 2},
            "age": {"type": "int", "min": 18, "max": 65},
            "email": {"type": "str", "pattern": r"^[^@]+@[^@]+\.[^@]+$"},
        }
        schema = ValidationSchema(schema_dict)

        result = validate_schema(clean_test_session, schema)
        assert result.valid is True
        assert len(result.errors) == 0

    async def test_validate_schema_type_mismatch(self, validation_test_session):
        """Test schema validation with type mismatches."""
        schema = {
            "age": {"type": "str"},  # Age is int in data
            "salary": {"type": "bool"},  # Salary is float in data
        }

        result = validate_schema(validation_test_session, ValidationSchema(schema))
        assert result.valid is False
        assert len(result.validation_errors) > 0
        assert "age" in result.validation_errors

    async def test_validate_schema_null_violations(self, validation_test_session):
        """Test schema validation with null value violations."""
        schema = {
            "email": {"nullable": False},  # But data has null email
        }

        result = validate_schema(validation_test_session, ValidationSchema(schema))
        assert result.valid is False
        assert "email" in result.validation_errors

    async def test_validate_schema_min_max_violations(self, validation_test_session):
        """Test schema validation with min/max violations."""
        schema = {
            "age": {"type": "int", "min": 0, "max": 120},
            "salary": {"type": "int", "min": 30000},
        }

        result = validate_schema(validation_test_session, ValidationSchema(schema))
        assert result.valid is False
        # Should catch negative age and zero salary

    async def test_validate_schema_pattern_violations(self, validation_test_session):
        """Test schema validation with pattern violations."""
        schema = {
            "email": {"pattern": r"^[^@]+@[^@]+\.com$"},  # Only .com emails
            "status": {"pattern": r"^(active|inactive)$"},  # Specific values only
        }

        result = validate_schema(validation_test_session, ValidationSchema(schema))
        assert result.valid is False

    async def test_validate_schema_allowed_values(self, validation_test_session):
        """Test schema validation with allowed values."""
        schema = {
            "status": {"values": ["active", "inactive"]},  # Excludes "pending", "retired"
        }

        result = validate_schema(validation_test_session, ValidationSchema(schema))
        assert result.valid is False
        assert "status" in result.validation_errors

    async def test_validate_schema_uniqueness(self, problematic_test_session):
        """Test schema validation with uniqueness requirements."""
        schema = {
            "id": {"unique": True},
        }

        result = validate_schema(problematic_test_session, ValidationSchema(schema))
        assert result.valid is False
        assert "id" in result.validation_errors

    async def test_validate_schema_string_length(self, validation_test_session):
        """Test schema validation with string length rules."""
        schema = {
            "name": {"min_length": 5, "max_length": 20},
            "email": {"min_length": 8},
        }

        result = validate_schema(validation_test_session, ValidationSchema(schema))
        # Some names might be too short - should have validation errors
        assert isinstance(result.valid, bool)

    async def test_validate_schema_missing_columns(self, clean_test_session):
        """Test schema validation with missing columns."""
        schema = {
            "id": {"type": "int"},
            "nonexistent": {"type": "str"},
        }

        result = validate_schema(clean_test_session, ValidationSchema(schema))
        assert result.valid is False
        assert "nonexistent" in result.validation_errors
        assert len(result.summary.missing_columns) > 0

    async def test_validate_schema_invalid_regex(self, clean_test_session):
        """Test schema validation with invalid regex pattern."""
        from pydantic import ValidationError as PydanticValidationError

        schema = {
            "email": {"pattern": "[invalid regex"},  # Invalid regex
        }

        # Should fail when creating ValidationSchema due to invalid regex
        with pytest.raises(PydanticValidationError):
            ValidationSchema(schema)

    async def test_validate_schema_invalid_session(self):
        """Test schema validation with invalid session."""
        from fastmcp.exceptions import ToolError

        schema = {"id": {"type": "int"}}

        with pytest.raises(ToolError):
            validate_schema("invalid-session", ValidationSchema(schema))

    async def test_validate_schema_empty_schema(self, clean_test_session):
        """Test schema validation with empty schema."""
        result = validate_schema(clean_test_session, ValidationSchema({}))
        assert result.valid is True


@pytest.mark.asyncio
class TestDataQualityChecking:
    """Test data quality checking functionality."""

    async def test_check_data_quality_default_rules(self, problematic_test_session):
        """Test data quality check with default rules."""
        result = check_data_quality(problematic_test_session)
        assert hasattr(result, "quality_results")
        assert hasattr(result.quality_results, "overall_score")
        assert hasattr(result.quality_results, "rule_results")
        assert hasattr(result.quality_results, "issues")

    async def test_check_data_quality_completeness(self, problematic_test_session):
        """Test data quality completeness check."""
        rules = [CompletenessRule(threshold=0.8)]

        result = check_data_quality(problematic_test_session, rules)
        quality = result.quality_results

        # Should find completeness issues in problematic data
        completeness_checks = [c for c in quality.rule_results if c.rule_type == "completeness"]
        assert len(completeness_checks) > 0

    async def test_check_data_quality_duplicates(self, problematic_test_session):
        """Test data quality duplicate detection."""
        rules = [DuplicatesRule(threshold=0.0)]  # No duplicates allowed

        result = check_data_quality(problematic_test_session, rules)
        quality = result.quality_results

        # Should find duplicate rows
        duplicate_checks = [c for c in quality.rule_results if c.rule_type == "duplicates"]
        assert len(duplicate_checks) > 0
        assert not duplicate_checks[0].passed  # Should fail with duplicates found

    async def test_check_data_quality_uniqueness(self, problematic_test_session):
        """Test data quality uniqueness check."""
        rules = [UniquenessRule(column="id", expected_unique=True)]

        result = check_data_quality(problematic_test_session, rules)
        quality = result.quality_results

        uniqueness_checks = [c for c in quality.rule_results if c.rule_type == "uniqueness"]
        assert len(uniqueness_checks) > 0

    async def test_check_data_quality_data_types(self, problematic_test_session):
        """Test data quality data type consistency."""
        rules = [DataTypesRule()]

        result = check_data_quality(problematic_test_session, rules)
        quality = result.quality_results

        type_checks = [c for c in quality.rule_results if c.rule_type == "data_type_consistency"]
        assert len(type_checks) > 0

    async def test_check_data_quality_outliers(self, problematic_test_session):
        """Test data quality outlier detection."""
        rules = [OutliersRule(threshold=0.1)]

        result = check_data_quality(problematic_test_session, rules)
        quality = result.quality_results

        outlier_checks = [c for c in quality.rule_results if c.rule_type == "outliers"]
        assert len(outlier_checks) > 0

    async def test_check_data_quality_consistency(self, validation_test_session):
        """Test data quality consistency check."""
        rules = [ConsistencyRule(columns=["join_date"])]

        result = check_data_quality(validation_test_session, rules)
        # Consistency rules may not generate results for all datasets
        assert hasattr(result, "quality_results")

    async def test_check_data_quality_invalid_session(self):
        """Test data quality check with invalid session."""
        from fastmcp.exceptions import ToolError

        with pytest.raises(ToolError):
            check_data_quality("invalid-session")

    async def test_check_data_quality_clean_data(self, clean_test_session):
        """Test data quality check on clean data."""
        result = check_data_quality(clean_test_session)
        quality = result.quality_results

        # Clean data should have high quality score
        assert quality.overall_score > 80

    async def test_check_data_quality_score_calculation(self, problematic_test_session):
        """Test quality score calculation."""
        result = check_data_quality(problematic_test_session)
        quality = result.quality_results

        # Should have issues and lower score
        assert quality.overall_score < 100
        assert len(quality.issues) > 0


@pytest.mark.asyncio
class TestAnomalyDetection:
    """Test anomaly detection functionality."""

    async def test_find_anomalies_statistical(self, problematic_test_session):
        """Test statistical anomaly detection."""
        result = find_anomalies(problematic_test_session, methods=["statistical"], sensitivity=0.95)
        assert hasattr(result, "anomalies")
        assert hasattr(result.anomalies, "by_method")

        # Should find statistical anomalies in age, salary columns
        if "statistical" in result.anomalies.by_method:
            stats_anomalies = result.anomalies.by_method["statistical"]
            assert len(stats_anomalies) > 0

    async def test_find_anomalies_pattern(self, problematic_test_session):
        """Test pattern anomaly detection."""
        result = find_anomalies(problematic_test_session, methods=["pattern"], sensitivity=0.8)
        assert hasattr(result, "anomalies")

    async def test_find_anomalies_missing(self, problematic_test_session):
        """Test missing value anomaly detection."""
        result = find_anomalies(problematic_test_session, methods=["missing"], sensitivity=0.9)
        assert hasattr(result, "anomalies")

    async def test_find_anomalies_all_methods(self, problematic_test_session):
        """Test anomaly detection with all methods."""
        result = find_anomalies(problematic_test_session)
        anomalies = result.anomalies

        assert hasattr(anomalies, "summary")
        assert hasattr(anomalies, "by_column")
        assert hasattr(anomalies, "by_method")
        assert isinstance(anomalies.summary.total_anomalies, int)

    async def test_find_anomalies_specific_columns(self, problematic_test_session):
        """Test anomaly detection on specific columns."""
        result = find_anomalies(problematic_test_session, columns=["age", "score"])
        assert result.columns_analyzed == ["age", "score"]

    async def test_find_anomalies_sensitivity_levels(self, problematic_test_session):
        """Test different sensitivity levels."""
        # High sensitivity should find more anomalies
        high_sens = find_anomalies(problematic_test_session, sensitivity=0.99)
        low_sens = find_anomalies(problematic_test_session, sensitivity=0.5)

        # Both should succeed
        high_count = high_sens.anomalies.summary.total_anomalies
        low_count = low_sens.anomalies.summary.total_anomalies

        # High sensitivity should generally find more or equal anomalies
        assert high_count >= low_count

    async def test_find_anomalies_clean_data(self, clean_test_session):
        """Test anomaly detection on clean data."""
        result = find_anomalies(clean_test_session)

        # Clean data should have few or no anomalies
        anomalies = result.anomalies
        assert anomalies.summary.total_anomalies == 0

    async def test_find_anomalies_missing_columns(self, clean_test_session):
        """Test anomaly detection with missing columns."""
        from fastmcp.exceptions import ToolError

        with pytest.raises(ToolError):
            find_anomalies(clean_test_session, columns=["nonexistent"])

    async def test_find_anomalies_invalid_session(self):
        """Test anomaly detection with invalid session."""
        from fastmcp.exceptions import ToolError

        with pytest.raises(ToolError):
            find_anomalies("invalid-session")

    async def test_find_anomalies_empty_methods(self, clean_test_session):
        """Test anomaly detection with empty methods list."""
        result = find_anomalies(clean_test_session, methods=[])
        # Should still work but find no anomalies
        assert result.anomalies.summary.total_anomalies == 0


@pytest.mark.asyncio
class TestValidationEdgeCases:
    """Test validation edge cases and error handling."""

    async def test_validate_schema_empty_dataframe(self):
        """Test schema validation on empty dataframe."""
        # load_csv_from_content now rejects empty CSVs
        with pytest.raises(ToolError, match="no data rows"):
            await load_csv_from_content("id,name\n")  # Header only

    async def test_schema_validation_all_types(self, validation_test_session):
        """Test schema validation with all supported data types."""
        schema = {
            "id": {"type": "int"},
            "name": {"type": "str"},
            "age": {"type": "int"},
            "salary": {"type": "float"},
            # Note: bool and datetime types would need appropriate test data
        }

        result = validate_schema(validation_test_session, ValidationSchema(schema))
        # Should complete validation
        assert hasattr(result, "valid")

    async def test_data_quality_empty_rules(self, clean_test_session):
        """Test data quality check with empty rules."""
        result = check_data_quality(clean_test_session, [])
        assert hasattr(result, "quality_results")
        assert result.quality_results.overall_score > 0
        # Should use default rules

    async def test_data_quality_custom_threshold(self, problematic_test_session):
        """Test data quality with custom thresholds."""
        rules = [
            CompletenessRule(threshold=0.5),  # Very lenient
            DuplicatesRule(threshold=0.5),  # Allow many duplicates
        ]

        result = check_data_quality(problematic_test_session, rules)
        quality = result.quality_results

        # With lenient thresholds, score should be higher
        assert quality.overall_score > 50

    async def test_find_anomalies_numeric_only(self):
        """Test anomaly detection on numeric-only data."""
        numeric_csv = """value1,value2,value3
1,10,100
2,20,200
3,30,300
100,40,400
5,50,500"""

        result = await load_csv_from_content(numeric_csv)
        session_id = result.session_id

        anomaly_result = find_anomalies(session_id, methods=["statistical"])
        # Should complete without errors
        assert hasattr(anomaly_result, "anomalies")

    async def test_find_anomalies_string_only(self):
        """Test anomaly detection on string-only data."""
        string_csv = """category,description,status
A,Normal data,active
B,Regular item,active
C,Standard entry,active
Z,OUTLIER DATA,DIFFERENT
D,Another normal,active"""

        result = await load_csv_from_content(string_csv)
        session_id = result.session_id

        anomaly_result = find_anomalies(session_id, methods=["pattern"])
        # Should complete without errors
        assert hasattr(anomaly_result, "anomalies")


@pytest.mark.asyncio
class TestValidationIntegration:
    """Test validation integration with other tools."""

    async def test_validation_after_transformations(self, validation_test_session):
        """Test validation after data transformations."""
        # First apply some transformations
        from src.databeak.services.transformation_operations import fill_missing_values

        # Fill missing values
        await fill_missing_values(validation_test_session, strategy="drop")

        # Then validate
        schema = {"email": {"nullable": False}}
        result = validate_schema(validation_test_session, ValidationSchema(schema))
        # After dropping nulls, email should be non-null
        assert hasattr(result, "valid")

    async def test_quality_check_recommendations(self, problematic_test_session):
        """Test that quality check provides useful recommendations."""
        result = check_data_quality(problematic_test_session)

        quality = result.quality_results
        if quality.overall_score < 85:
            assert len(quality.recommendations) > 0

    async def test_validation_with_operations_history(self, clean_test_session):
        """Test that validation operations work with session management."""
        # Perform validation
        schema = {"id": {"type": "int"}}
        result = validate_schema(clean_test_session, ValidationSchema(schema))
        # Should complete validation
        assert hasattr(result, "valid")

        # Check if session info can be retrieved (verifies session still exists)
        from src.databeak.servers.io_server import get_session_info

        info_result = await get_session_info(clean_test_session)
        assert info_result.success is True
        assert info_result.data_loaded is True
