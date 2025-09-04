---
sidebar_position: 1
title: Quick Start
---

# Quick Start Tutorial

Learn how to use CSV Editor in 10 minutes with this hands-on tutorial.
We'll process a sample sales dataset using natural language commands.

## Prerequisites

- CSV Editor installed and configured ([Installation Guide](../installation))
- An AI assistant client (Claude Desktop, Continue, etc.) configured with
  CSV Editor

## Step 1: Load Your Data

Ask your AI assistant:

> "Load the sales data from my CSV file"

The AI will use the `load_csv` tool to create a new session and load your data. You'll see a response with:

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
>
> "Remove any duplicate rows from this dataset"

### Handle Missing Values
>
> "Fill missing quantity values with 0 and missing customer_id values
> with 'UNKNOWN'"

### Fix Data Types
>
> "Convert the date column to datetime format"

## Step 4: Transform the Data

### Filter Data
>
> "Show me only Electronics products with price greater than $100"

### Add Calculated Columns
>
> "Add a total_value column that multiplies quantity by price"

### Group and Summarize
>
> "Group by category and show total sales and average price for each"

## Step 5: Analyze the Data

### Statistical Analysis
>
> "Calculate correlation between price and quantity"

### Outlier Detection
>
> "Find any outliers in the price column using the IQR method"

### Data Quality
>
> "Check the overall data quality and give me a quality score"

## Step 6: Export Results

> "Export this cleaned and analyzed data as an Excel file named 'sales_analysis.xlsx'"

## Advanced Features

### Undo/Redo Operations

Made a mistake? No problem:

> "Undo the last operation"
> "Show me the operation history"
> "Restore to the state before I added the total_value column"

### Auto-Save Configuration

Set up automatic saving:

> "Configure auto-save to create backups in a backup folder with a
> maximum of 5 backups"

### Session Management

Work with multiple datasets:

> "Create a new session for the customer data"
> "List all my active sessions"
> "Close the sales data session"

## Real-World Examples

### Data Cleaning Workflow

```python
# Natural language commands:
"Load the messy customer data"
"Remove duplicate rows"
"Fill missing email addresses with 'no-email@domain.com'"
"Standardize the phone number format"
"Remove rows where age is negative or over 120"
"Export the cleaned data"
```

### Analysis Pipeline

```python
# Business intelligence workflow:
"Load quarterly sales data"
"Filter for completed transactions only"
"Group by product category and month"
"Calculate total revenue and average order value"
"Find the top 10 selling products"
"Create correlation matrix for price vs quantity vs revenue"
"Export summary as Excel with charts"
```

### Data Validation

```python
# Quality assurance workflow:
"Load the new data batch"
"Validate against the expected schema"
"Check data quality score"
"Find any statistical anomalies"
"Generate a data profiling report"
"Flag any quality issues for review"
```

## Tips for Success

### 1. **Be Specific**

Instead of "analyze the data", try "calculate descriptive statistics for
numeric columns and show correlation matrix"

### 2. **Use Session IDs**

For multiple datasets, specify which session: "In session ABC123, filter
rows where status equals 'active'"

### 3. **Chain Operations**

"Load sales.csv, remove duplicates, filter for 2024 data, then calculate
monthly totals"

### 4. **Leverage Auto-Save**

CSV Editor automatically saves your work, so you can focus on analysis
without worrying about losing changes

### 5. **Explore History**

Use undo/redo freely to experiment with different approaches

## Next Steps

- **[API Reference](../api/overview)** - Complete tool documentation
- **[Examples](https://github.com/jonpspri/csv-editor/tree/main/examples)**
  - More real-world scenarios
- **[GitHub Repository](https://github.com/jonpspri/csv-editor)** - Source
  code and community

---

**Congratulations!** You now know how to use CSV Editor to transform your
AI assistant into a powerful data analyst.
