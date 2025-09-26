# Test Fixtures

This directory contains sample CSV files used for integration testing.

## Files

### `sample_data.csv`

Basic employee data with:

- name, age, city, salary columns
- 5 rows of clean data
- Good for basic loading and operations testing

### `sales_data.csv`

Sales transaction data with:

- date, product, quantity, price, customer_id columns
- 6 rows of sales data
- Useful for date/time operations and aggregation testing

### `empty.csv`

Empty CSV with headers only:

- id, name, value columns
- No data rows
- Tests handling of empty datasets

### `missing_values.csv`

Data with missing values:

- name, age, city, score columns
- 6 rows with various missing values (empty strings and missing fields)
- Tests data quality and missing value handling

## Usage

Use the `get_fixture_path()` helper function from `conftest.py` to get absolute
paths:

```python
from tests.integration.conftest import get_fixture_path

# Get real absolute path to fixture
csv_path = get_fixture_path("sample_data.csv")
result = await server.call_tool("load_csv", {"file_path": csv_path})
```

This ensures that DataBeak receives the full absolute path, avoiding any
confusion about relative paths during testing.

### Security Features

The `get_fixture_path()` function includes security validations to prevent
directory traversal attacks:

- **Path separator validation**: Rejects fixture names containing `/`, `\`, or
  other path separators
- **Relative path protection**: Blocks `..`, `.`, and other relative path
  components
- **Directory containment**: Ensures resolved paths stay within the
  `tests/fixtures/` directory
- **Input sanitization**: Rejects empty strings and whitespace-only names

**Examples of rejected inputs:**

- `../../../etc/passwd` (directory traversal)
- `subdir/file.csv` (path separators)
- `.hidden` (relative path component)
- `""` (empty string)
