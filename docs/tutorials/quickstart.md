---
sidebar_position: 1
title: Quick Start
---

## Quick Start Tutorial

Learn how to use DataBeak in 10 minutes with this hands-on tutorial. We'll
process a sample sales dataset using natural language commands.

## Prerequisites

- DataBeak installed and configured ([Installation Guide](../installation.md))
- An AI assistant client (Claude Desktop, Continue, etc.) configured with
  DataBeak

## Step 1: Load Your Data

Ask your AI assistant to load data from a URL or paste CSV content:

> "Load the sales data from this URL: <https://example.com/sales.csv>"

Or provide CSV content directly:

> "Load this CSV data: name,price,quantity\\nWidget,10.99,5\\nGadget,25.50,3"

The AI will use the `load_csv_from_url` or `load_csv_from_content` tool to
create a new session and load your data. You'll see a response with:

- Session ID for tracking
- Data shape (rows × columns)
- Column names and types
- Memory usage information

## Step 2: Explore the Data

Get an overview of your dataset:

> "Show me basic statistics for this data"

This uses `get_statistics` to provide:

- Row and column counts
- Data types summary
- Missing values count
- Memory usage

For detailed column analysis:

> "Get detailed statistics for the price and quantity columns"

## Step 3: Clean the Data

### Remove Duplicates

> "Remove any duplicate rows from this dataset"

### Handle Missing Values

> "Fill missing quantity values with 0 and missing customer_id values with
> 'UNKNOWN'"

### Fix Data Types

> "Convert the date column to datetime format"

## Step 4: Transform the Data

### Filter Data

> "Show me only Electronics products with price greater than $100"

### Add Calculated Columns

> "Add a total_value column that multiplies quantity by price"

### Group and Summarize

> "Group by category and show total sales and average price for each"

## Step 5: Analyze the Data

### Statistical Analysis

> "Calculate correlation between price and quantity"

### Outlier Detection

> "Find any outliers in the price column using the IQR method"

### Data Quality

> "Check the overall data quality and give me a quality score"

## Step 6: Save Results

DataBeak processes data in memory for web-based hosting security. To save
results, export them through your AI assistant which can save files on your
behalf.

## Advanced Features

### Undo/Redo Operations

Made a mistake? No problem:

> "Undo the last operation" "Show me the operation history" "Restore to the
> state before I added the total_value column"

### Data Retrieval

Get processed data back as CSV content for further use:

> "Show me the cleaned data as CSV content"

### Session Management

Work with multiple datasets:

> "Create a new session for the customer data" "List all my active sessions"
> "Close the sales data session"

## Real-World Examples

### Data Cleaning Workflow

```python
# Natural language commands:
"Load customer data from URL: https://example.com/customers.csv"

"Remove duplicate rows"
"Fill missing email addresses with 'no-email@domain.com'"
"Standardize the phone number format"
"Remove rows where age is negative or over 120"
"Show me the cleaned data preview"
```

### Analysis Pipeline

```python
# Business intelligence workflow:
"Load quarterly sales data from URL: https://example.com/q1-sales.csv"

"Filter for completed transactions only"
"Group by product category and month"
"Calculate total revenue and average order value"
"Find the top 10 selling products"
"Create correlation matrix for price vs quantity vs revenue"
"Show me the summary statistics"
```

### Data Validation

```python
# Quality assurance workflow:
"Load data from this CSV content: [paste CSV here]"

"Validate against the expected schema"
"Check data quality score"
"Find any statistical anomalies"
"Generate a data profiling report"
"Show me any quality issues found"
```

## Tips for Success

### 1. **Be Specific**

Instead of "analyze the data", try "calculate descriptive statistics for numeric
columns and show correlation matrix"

### 2. **Use Session IDs**

For multiple datasets, specify which session: "In session ABC123, filter rows
where status equals 'active'"

### 3. **Chain Operations**

"Load sales data from URL, remove duplicates, filter for 2024 data, then
calculate monthly totals"

### 4. **Work with Web Data**

DataBeak is designed for web-based hosting, so it works with URLs and in-memory
data without accessing your local file system

### 5. **Explore History**

Use DataBeak's stateless design to experiment with different approaches -
retrieve results when needed

## Next Steps

- **[API Reference](../api/index.md)** - Complete tool documentation
- **[Examples](https://github.com/jonpspri/databeak/tree/main/examples)**
  - More real-world scenarios
- **[GitHub Repository](https://github.com/jonpspri/databeak)** - Source code
  and community

______________________________________________________________________

**Congratulations!** You now know how to use DataBeak to transform your AI
assistant into a powerful data analyst.
