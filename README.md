# Amazon Reviews Data Processor
A Databricks notebook suite for processing Amazon review datasets, handling various data formats and structures.

## Data Source
The data used in this project comes from the Amazon Reviews Dataset, available at:
https://amazon-reviews-2023.github.io/

This dataset contains product reviews and metadata from Amazon, spanning various product categories. The dataset is used for research purposes to analyze customer feedback, sentiment analysis, and product categorization.

## Pipeline Workflow
![Amazon Data Processing Pipeline](pipeline_diagram.png)
*Diagram of the Amazon Reviews + Meta Data Processing Pipeline*

## Features
- Data ingestion from JSONL files
- Cleaning and preprocessing
- Join operations and sampling
- Visualization of data quality
- Export functionality

## Notebooks

### amazon_processor.py
- Processes a single Amazon dataset (reviews + metadata)
- Performs data cleaning and preprocessing
- Samples data by category
- Provides visualization and analysis

### amazon_unified_processor.py
- Processes multiple Amazon datasets in a unified manner
- Combines data across different categories and product types
- Creates a balanced dataset with samples from each category
- Generates unified exports and visualizations
- Supports large-scale data processing across multiple files

## Usage Instructions
1. Import these notebooks into your Databricks workspace
2. Set your file paths in the "Main Execution" section
3. Run the notebook
4. Use the provided export functionality to download processed data

## Required Libraries
- PySpark
- pandas
- matplotlib
- seaborn

## Setup Instructions

## Upload the Dataset:
- Download the Amazon reviews dataset for the "Software" category (e.g., from the Amazon Review Data repository).
- Upload the reviews and metadata JSONL files to your Databricks File System (DBFS):
- Go to the Databricks workspace.
- Use the "Upload Data" option to upload the files to a directory, e.g., /FileStore/tables/amazon_reviews/.

## Create a Databricks Cluster:
- In your Databricks workspace, create a cluster:
- Runtime: Databricks 10.4 LTS (or later) with Spark 3.x.
- Node Type: Choose a node with sufficient memory and compute (e.g., 8 GB RAM, 2 cores) to handle the 4.68M-row dataset.

## Start the cluster.
- Import the Notebook:
- Download the amazon_reviews_analysis_notebook.py file from this repository.
- In your Databricks workspace, go to "Workspace" > "Import".
- Upload the amazon_reviews_analysis_notebook.py file as a Python notebook.
- Alternatively, create a new Python notebook in Databricks and copy-paste the code from amazon_reviews_analysis_notebook.py.

## Configure the Notebook:
- Open the notebook in your Databricks workspace.
- Attach the notebook to the cluster you created.
- Update the file paths for reviews_path, metadata_path, and output_path at the top of the notebook to match the location of your uploaded dataset:

## Pipeline Details
The notebook implements the following pipeline in the process_dataset function:

## 1. Load Data:
- Load reviews and metadata JSONL files using PySpark.
## 2. Clean Data:
- Drop rows with missing asin or user_id in reviews.
- Remove duplicate reviews (dropDuplicates(["asin", "user_id"])).
- Handle the timestamp column (convert to string or drop).
- Drop rows with missing parent_asin in metadata.
- Ensure main_category exists (extract from categories or set a default).
## 3. Handle Outliers:
- Apply IQR-based outlier handling to:
- rating (min bound: 1.0).
- helpful_vote (min bound: 0.0).
- price (min bound: 0.0).
- rating_number (min bound: 0.0).
## 4. Join Data:
- Inner join reviews and metadata on asin and parent_asin.
- Select relevant columns and add a dataset column.
- Convert Categorical to Numerical:
- Convert verified_purchase to verified_purchase_numeric (1/0).
- Convert main_category to main_category_numeric using StringIndexer.
## 5. Compute Summary Statistics:
- Compute count, mean, stddev, min, and max for numerical columns.
- Extract Insights:
- Derive insights like average rating, most common category, etc.
## 6. Generate Visualizations:
- Bar plot of ratings.
- Bar plot of categories.
- Scatter plot of price.
- Scatter plot of rating vs. helpful votes.
- Pie chart of verified purchases.
- Correlation matrix heatmap.
## 7. Feature Analysis:
- Identify significant correlations between numerical features (|correlation| > 0.5).
- Standardize numerical features (rating, helpful_vote, price, etc.) using StandardScaler.
- Tokenize and vectorize review_text (if present) using Tokenizer and HashingTF.
## 8. Save Processed Data:
- Save as Parquet (partitioned by main_category_numeric).
- Save as CSV (with vector columns converted to strings).
- Create Downloadable CSV:
