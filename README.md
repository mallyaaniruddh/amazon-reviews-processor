# Amazon Reviews Data Processor

A Databricks notebook for processing Amazon review datasets, handling various data formats and structures.

## Data Source

The data used in this project comes from the Amazon Reviews Dataset, available at:
https://amazon-reviews-2023.github.io/

This dataset contains product reviews and metadata from Amazon, spanning various product categories. The dataset is used for research purposes to analyze customer feedback, sentiment analysis, and product categorization.

## Features
- Data ingestion from JSONL files
- Cleaning and preprocessing
- Join operations and sampling
- Visualization of data quality
- Export functionality

## Usage Instructions
1. Import this notebook into your Databricks workspace
2. Set your file paths in the "Main Execution" section
3. Run the notebook
4. Use the provided export functionality to download processed data

## Required Libraries
- PySpark
- pandas
- matplotlib
- seaborn
