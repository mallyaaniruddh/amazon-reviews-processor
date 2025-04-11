# Databricks notebook source
# MAGIC %md
# MAGIC # Amazon Reviews Unified Multi-Dataset Processor
# MAGIC
# MAGIC This notebook processes multiple Amazon review datasets using Spark on Databricks, creating a unified dataset that combines data across categories. It performs the following operations:
# MAGIC
# MAGIC 1. Data Discovery: Locates pairs of review and metadata files
# MAGIC 2. Parallel Processing: Processes each dataset with standardized cleaning operations
# MAGIC 3. Category Unification: Combines samples from the same product category across datasets
# MAGIC 4. Balanced Sampling: Ensures fair representation across categories
# MAGIC 5. Quality Analysis: Generates visualizations and statistics
# MAGIC
# MAGIC All operations are optimized for Spark on Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment Setup and Configuration

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, lit, count, desc, array, explode, regexp_replace, udf, struct, collect_list
from pyspark.sql.types import StringType, IntegerType, DoubleType, ArrayType, StructType, StructField
import os
import re
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict

# Configure Spark for optimal performance
spark.conf.set("spark.sql.shuffle.partitions", 100)
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.caseSensitive", "true")

# Define default parameters
DEFAULT_SAMPLE_SIZE = 10000
DEFAULT_MAX_TOTAL = 100000

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. File Discovery and Dataset Management

# COMMAND ----------

def discover_datasets(input_path, pattern="*"):
    """
    Discover Amazon dataset pairs (reviews + metadata) in the specified path
    
    Parameters:
    -----------
    input_path : str
        Path to search for datasets
    pattern : str
        Pattern to match files
        
    Returns:
    --------
    list
        List of dataset pairs (reviews, metadata)
    """
    # Debug: Print the exact path we're checking
    print(f"Searching for dataset pairs in: {input_path}")
    
    # List all files in the directory
    try:
        file_list = dbutils.fs.ls(input_path)
        print(f"Found {len(file_list)} files in directory")
        
        # Print all files to debug
        for file in file_list:
            print(f"  • {file.name} ({file.path})")
    except Exception as e:
        print(f"Error listing files: {str(e)}")
        return []
    
    # Filter for JSONL files
    jsonl_files = [f.path for f in file_list if f.path.endswith(".jsonl")]
    print(f"Found {len(jsonl_files)} JSONL files")
    
    # Find review files (not starting with "meta_")
    review_files = [f for f in jsonl_files if not os.path.basename(f).startswith("meta_")]
    print(f"Found {len(review_files)} potential review files")
    
    # Match with metadata files
    dataset_pairs = []
    for review_file in review_files:
        # Extract dataset name
        base_name = os.path.basename(review_file)
        dataset_name = base_name.split('.')[0]
        
        # Check if there's a matching metadata file
        meta_file = os.path.join(input_path, f"meta_{dataset_name}.jsonl")
        
        # Try multiple path formats for Databricks
        alt_meta_file = f"{input_path}/meta_{dataset_name}.jsonl"
        
        # Verify the file exists
        meta_exists = meta_file in jsonl_files
        alt_meta_exists = alt_meta_file in jsonl_files
        
        print(f"Looking for metadata file matching {dataset_name}:")
        print(f"  - Tried: {meta_file} (exists: {meta_exists})")
        print(f"  - Tried: {alt_meta_file} (exists: {alt_meta_exists})")
        
        if meta_exists:
            dataset_pairs.append((review_file, meta_file, dataset_name))
        elif alt_meta_exists:
            dataset_pairs.append((review_file, alt_meta_file, dataset_name))
    
    print(f"Found {len(dataset_pairs)} dataset pairs")
    for review, meta, name in dataset_pairs:
        print(f"  • {name}: {review} + {meta}")
    
    # If no pairs found, try more flexible matching
    if len(dataset_pairs) == 0:
        print("Trying more flexible matching...")
        meta_files = [f for f in jsonl_files if os.path.basename(f).startswith("meta_")]
        
        if len(meta_files) > 0 and len(review_files) > 0:
            print(f"Found {len(meta_files)} metadata files and {len(review_files)} review files")
            print("Creating pairs based on available files:")
            
            # Just pair them up one-to-one if we have equal numbers
            if len(meta_files) == len(review_files):
                for i in range(len(review_files)):
                    review_file = review_files[i]
                    meta_file = meta_files[i]
                    dataset_name = os.path.basename(review_file).split('.')[0]
                    
                    dataset_pairs.append((review_file, meta_file, dataset_name))
                    print(f"  • {dataset_name}: {review_file} + {meta_file}")
            else:
                # Take the first review file and first meta file
                review_file = review_files[0]
                meta_file = meta_files[0]
                dataset_name = os.path.basename(review_file).split('.')[0]
                
                dataset_pairs.append((review_file, meta_file, dataset_name))
                print(f"  • {dataset_name}: {review_file} + {meta_file}")
    
    return dataset_pairs

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Processing Functions

# COMMAND ----------

def process_dataset(reviews_path, metadata_path, dataset_name, sample_size=DEFAULT_SAMPLE_SIZE, include_timestamp=False):
    """
    Process a single Amazon dataset (reviews + metadata)
    
    Parameters:
    -----------
    reviews_path : str
        Path to the reviews JSONL file
    metadata_path : str
        Path to the metadata JSONL file
    dataset_name : str
        Name of the dataset
    sample_size : int
        Maximum number of samples to take per category
    include_timestamp : bool
        Whether to include timestamp information
        
    Returns:
    --------
    dict
        Processing results and sampled data
    """
    print(f"\n=== Processing dataset: {dataset_name} ===")
    print(f"  Reviews: {reviews_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Sample size: {sample_size} per category")
    
    try:
        # Step 1: Load data
        print("Loading datasets...")
        reviews_df = spark.read.option("multiLine", "false").json(reviews_path)
        metadata_df = spark.read.option("multiLine", "false").json(metadata_path)
        
        # Step 2: Clean reviews data
        print("Cleaning reviews data...")
        reviews_df = reviews_df.na.drop(subset=["asin", "user_id"])
        reviews_df = reviews_df.dropDuplicates(["asin", "user_id"])
        
        # Handle timestamp
        has_timestamp = "timestamp" in reviews_df.columns
        if has_timestamp:
            if include_timestamp:
                print("Converting timestamp to string for safe handling...")
                reviews_df = reviews_df.withColumn("timestamp_str", col("timestamp").cast(StringType()))
                reviews_df = reviews_df.drop("timestamp")
            else:
                print("Dropping timestamp column...")
                reviews_df = reviews_df.drop("timestamp")
        
        # Step 3: Clean metadata
        print("Cleaning metadata...")
        metadata_df = metadata_df.na.drop(subset=["parent_asin"])
        
        # Ensure main_category exists (extract from categories if needed)
        if "main_category" not in metadata_df.columns:
            if "categories" in metadata_df.columns:
                # Try to extract main category from first category
                try:
                    metadata_df = metadata_df.withColumn("main_category", 
                                                     col("categories")[0][0])
                except:
                    # Default category if extraction fails
                    metadata_df = metadata_df.withColumn("main_category", 
                                                     lit(dataset_name))
            else:
                # Default category if none exists
                metadata_df = metadata_df.withColumn("main_category", 
                                                 lit(dataset_name))
        
        # Step 4: Join reviews with metadata
        print("Joining reviews with metadata...")
        joined_df = reviews_df.join(
            metadata_df,
            reviews_df["parent_asin"] == metadata_df["parent_asin"],
            "inner"
        )
        
        # Add a dataset column to track the source dataset
        joined_df = joined_df.withColumn("dataset", lit(dataset_name))
        
        # Select relevant columns
        result_columns = [
            reviews_df["asin"].alias("product_id"),
            reviews_df["parent_asin"],
            reviews_df["user_id"],
            reviews_df["rating"],
            metadata_df["title"].alias("product_title"),
            metadata_df["main_category"],
            col("dataset")
        ]
        
        # Check for optional columns
        if has_timestamp and include_timestamp:
            result_columns.append(reviews_df["timestamp_str"])
                
        if "price" in metadata_df.columns:
            result_columns.append(metadata_df["price"])
        if "verified_purchase" in reviews_df.columns:
            result_columns.append(reviews_df["verified_purchase"])
        if "text" in reviews_df.columns:
            result_columns.append(reviews_df["text"].alias("review_text"))
        if "helpful_vote" in reviews_df.columns:
            result_columns.append(reviews_df["helpful_vote"])
        
        # Create final dataframe with selected columns
        result_df = joined_df.select(result_columns)
        
        # Print row count after joining
        join_count = result_df.count()
        print(f"Joined data has {join_count} rows")
        
        # Step 5: Sample by category
        print("Sampling records by category...")
        
        # Get categories
        categories = [row[0] for row in result_df.select("main_category").distinct().collect()]
        print(f"Found {len(categories)} categories")
        
        # Sample for each category
        samples_by_category = {}
        total_samples = 0
        
        for category in categories:
            # Filter for this category
            category_df = result_df.filter(col("main_category") == category)
            category_count = category_df.count()
            
            actual_sample_size = min(sample_size, category_count)
            print(f"  Category '{category}' has {category_count} records, sampling {actual_sample_size}")
            
            if actual_sample_size > 0:
                # Sample records
                sampled_df = category_df.orderBy(rand(42)).limit(actual_sample_size)
                
                # Convert to pandas for easier handling
                category_pandas_df = sampled_df.toPandas()
                
                # Store in our category samples dictionary
                samples_by_category[category] = category_pandas_df
                total_samples += len(category_pandas_df)
                
                print(f"  ✓ Sampled {len(category_pandas_df)} records")
            else:
                samples_by_category[category] = None
        
        return {
            "dataset": dataset_name,
            "categories": len(categories),
            "category_list": categories,
            "total_records": join_count,
            "sampled_records": total_samples,
            "status": "Success",
            "samples_by_category": samples_by_category
        }
        
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "dataset": dataset_name,
            "status": f"Failed: {str(e)}",
            "samples_by_category": {}
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Multi-Dataset Processing and Unification

# COMMAND ----------

def process_multiple_datasets(dataset_pairs, output_path, sample_size=DEFAULT_SAMPLE_SIZE, 
                             max_total_samples=DEFAULT_MAX_TOTAL, include_timestamp=False):
    """
    Process multiple Amazon datasets and create a unified dataset
    
    Parameters:
    -----------
    dataset_pairs : list
        List of (review_path, metadata_path, dataset_name) tuples
    output_path : str
        Path to save output files
    sample_size : int
        Maximum samples per category per dataset
    max_total_samples : int
        Maximum total samples across all categories
    include_timestamp : bool
        Whether to include timestamp data
        
    Returns:
    --------
    dict
        Processing summary
    """
    print(f"Processing {len(dataset_pairs)} datasets")
    print(f"Output path: {output_path}")
    print(f"Sample size per category: {sample_size}")
    print(f"Max total samples: {max_total_samples}")
    print(f"Include timestamp: {include_timestamp}")
    
    # Initialize tracking variables
    start_time = time.time()
    results = []
    category_counts = defaultdict(int)  # Count how many datasets have each category
    unified_samples = defaultdict(list)  # Store samples by category
    
    # Process each dataset
    for reviews_path, metadata_path, dataset_name in dataset_pairs:
        # Process this dataset
        result = process_dataset(
            reviews_path=reviews_path,
            metadata_path=metadata_path,
            dataset_name=dataset_name,
            sample_size=sample_size,
            include_timestamp=include_timestamp
        )
        
        results.append(result)
        
        # Update category counts and unified samples
        if 'samples_by_category' in result:
            for category, samples in result['samples_by_category'].items():
                if samples is not None and len(samples) > 0:
                    category_counts[category] += 1
                    unified_samples[category].append(samples)
    
    # Create unified category samples with balanced representation
    print("\n=== Creating unified category samples ===")
    
    # Calculate target samples per category to stay within max_total_samples
    total_categories = len(unified_samples)
    if total_categories > 0:
        target_per_category = max_total_samples // total_categories
    else:
        target_per_category = 0
    
    print(f"Target samples per category: {target_per_category}")
    
    # Create unified category dataframes
    unified_category_dfs = {}
    total_unified_samples = 0
    
    for category, sample_list in unified_samples.items():
        if not sample_list:
            continue
            
        # Combine all samples for this category
        combined_samples = pd.concat(sample_list, ignore_index=True)
        
        # Sample down to target if needed
        if len(combined_samples) > target_per_category and target_per_category > 0:
            print(f"Category '{category}' has {len(combined_samples)} samples, reducing to {target_per_category}")
            combined_samples = combined_samples.sample(n=target_per_category, random_state=42)
        
        unified_category_dfs[category] = combined_samples
        total_unified_samples += len(combined_samples)
        
        print(f"Category '{category}' has {len(combined_samples)} samples in unified dataset")
    
    # Create the final unified dataset
    all_samples = pd.concat(unified_category_dfs.values(), ignore_index=True) if unified_category_dfs else pd.DataFrame()
    print(f"\nUnified dataset has {len(all_samples)} total samples across {len(unified_category_dfs)} categories")
    
    # Save the unified dataset
    print("\n=== Saving unified dataset ===")
    
    # Create output directory
    dbutils.fs.mkdirs(output_path)
    
    # Convert to Spark DataFrame for saving
    if not all_samples.empty:
        unified_spark_df = spark.createDataFrame(all_samples)
        
        # Save as parquet
        unified_spark_df.write.mode("overwrite").parquet(f"{output_path}/unified_dataset")
        print(f"Saved unified dataset to: {output_path}/unified_dataset")
        
        # Also save as CSV for easy access
        unified_spark_df.write.mode("overwrite").option("header", "true").csv(f"{output_path}/unified_dataset_csv")
        print(f"Saved CSV version to: {output_path}/unified_dataset_csv")
        
        # Save individual category files
        print("\n=== Saving category-specific files ===")
        for category, category_df in unified_category_dfs.items():
            # Create safe category name
            safe_category = re.sub(r'[^a-zA-Z0-9]', '_', category)
            
            # Convert to Spark DataFrame
            category_spark_df = spark.createDataFrame(category_df)
            
            # Save as parquet
            category_output_path = f"{output_path}/categories/{safe_category}"
            category_spark_df.write.mode("overwrite").parquet(category_output_path)
            print(f"Saved category '{category}' to: {category_output_path}")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Create processing summary
    # Replace the code that creates and saves the summary with this:

    # Create processing summary (without including pandas DataFrames)
    summary = {
        "datasets_processed": len(results),
        "successful_datasets": sum(1 for r in results if r.get('status', '').startswith('Success')),
        "failed_datasets": sum(1 for r in results if not r.get('status', '').startswith('Success')),
        "total_categories": len(unified_samples),
        "total_samples": total_unified_samples,
        "elapsed_time": elapsed_time,
        "dataset_names": [r.get('dataset') for r in results]  # Just include names, not the full results with DataFrames
    }

    # Save processing summary as text instead of trying to use JSON
    summary_text = f"""
    Amazon Review Data Processing Summary
    ====================================

    Processing completed in {elapsed_time:.2f} seconds
    Datasets processed: {len(results)}
    Successful datasets: {summary['successful_datasets']}
    Failed datasets: {summary['failed_datasets']}
    Total categories: {len(unified_samples)}
    Total samples: {total_unified_samples}

    Datasets:
    {', '.join(summary['dataset_names'])}

    Categories:
    {', '.join(unified_samples.keys())}
    """

    # Create a simple DataFrame with just the summary text
    summary_df = spark.createDataFrame([(summary_text,)], ["summary"])
    summary_df.write.mode("overwrite").text(f"{output_path}/processing_summary")
    
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
    print(f"Processed {len(results)} datasets")
    print(f"Found {len(unified_samples)} unique categories")
    print(f"Collected {total_unified_samples} total samples")
    
    return summary

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Data Quality and Visualization Functions

# COMMAND ----------

def generate_unified_dataset_report(unified_df, output_path):
    """
    Generate data quality report for the unified dataset
    
    Parameters:
    -----------
    unified_df : DataFrame
        Unified Spark DataFrame
    output_path : str
        Path to save visualizations
    """
    print("Generating unified dataset report...")
    
    # Sample for visualization (to avoid memory issues with very large datasets)
    viz_sample = unified_df.sample(False, min(1.0, 10000.0/unified_df.count()), seed=42)
    
    # Convert to pandas for easier visualization
    pdf = viz_sample.toPandas()
    
    # Create a series of visualizations
    
    # 1. Category distribution
    plt.figure(figsize=(14, 8))
    category_counts = pdf['main_category'].value_counts().head(20)
    sns.barplot(x=category_counts.values, y=category_counts.index)
    plt.title('Top 20 Categories in Unified Dataset')
    plt.xlabel('Number of Records')
    plt.tight_layout()
    display(plt.gcf())
    plt.close()
    
    # 2. Rating distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rating', data=pdf)
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.tight_layout()
    display(plt.gcf())
    plt.close()
    
    # 3. Dataset distribution
    plt.figure(figsize=(12, 7))
    dataset_counts = pdf['dataset'].value_counts()
    sns.barplot(x=dataset_counts.values, y=dataset_counts.index)
    plt.title('Distribution by Source Dataset')
    plt.xlabel('Number of Records')
    plt.tight_layout()
    display(plt.gcf())
    plt.close()
    
    # 4. Basic statistics for numerical columns
    numeric_cols = [c for c in pdf.columns if pdf[c].dtype in [np.int64, np.float64]]
    if numeric_cols:
        stats_df = spark.createDataFrame(pdf[numeric_cols].describe())
        display(stats_df)
    
    # 5. Missing value analysis
    missing_counts = pdf.isnull().sum().sort_values(ascending=False)
    missing_percent = (missing_counts / len(pdf) * 100).round(2)
    missing_df = pd.DataFrame({'count': missing_counts, 'percent': missing_percent})
    missing_df = missing_df[missing_df['count'] > 0]
    
    if not missing_df.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=missing_df['percent'], y=missing_df.index)
        plt.title('Missing Values Analysis')
        plt.xlabel('Percent Missing')
        plt.tight_layout()
        display(plt.gcf())
        plt.close()
    
    print("Data quality report generated")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Main Execution

# COMMAND ----------

# Example usage (modify paths as needed for your Databricks environment)
input_path = "dbfs:/FileStore/."
output_path = "dbfs:/FileStore/amazon_unified"
custom_sample_size = 100
custom_include_timestamp = False

# Discover datasets
dataset_pairs = discover_datasets(input_path)

# Process all datasets and create unified dataset
if dataset_pairs:
    summary = process_multiple_datasets(
        dataset_pairs=dataset_pairs,
        output_path=output_path,
        sample_size=1000,  # Adjust as needed
        max_total_samples=10000,  # Adjust as needed
        include_timestamp=False
    )
    
    # Load the unified dataset for visualization
    unified_df = spark.read.parquet(f"{output_path}/unified_dataset")
    
    # Generate data quality report
    generate_unified_dataset_report(unified_df, output_path)
    
    # Display sample of the unified dataset
    print("\nSample of unified dataset:")
    display(unified_df.limit(10))
else:
    print("No dataset pairs found. Please check the input path.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Additional Analysis (Optional)

# COMMAND ----------

def analyze_review_text(unified_df):
    """
    Perform text analysis on review content
    
    Parameters:
    -----------
    unified_df : DataFrame
        Unified dataset with review_text column
    """
    if "review_text" not in unified_df.columns:
        print("No review_text column found in the dataset")
        return
    
    print("Analyzing review text...")
    
    # Sample for text analysis
    text_sample = unified_df.select("review_text", "rating", "main_category").sample(False, min(1.0, 5000.0/unified_df.count()), seed=42)
    
    # Basic text statistics
    text_sample = text_sample.withColumn("text_length", length(col("review_text")))
    text_sample = text_sample.withColumn("word_count", size(split(col("review_text"), "\\s+")))
    
    # Show text length statistics by rating
    display(text_sample.groupBy("rating").agg(
        avg("text_length").alias("avg_text_length"),
        avg("word_count").alias("avg_word_count"),
        count("*").alias("review_count")
    ).orderBy("rating"))
    
    # Show text length statistics by category (top 10 categories)
    top_categories = [row[0] for row in text_sample.groupBy("main_category").count().orderBy(desc("count")).limit(10).collect()]
    
    display(text_sample.filter(col("main_category").isin(top_categories)).groupBy("main_category").agg(
        avg("text_length").alias("avg_text_length"),
        avg("word_count").alias("avg_word_count"),
        count("*").alias("review_count")
    ).orderBy(desc("review_count")))
    
    print("Text analysis complete")

# Example usage (uncomment to run)
# if "unified_df" in locals():
#     analyze_review_text(unified_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Export Data to CSV for External Analysis (Optional)

# COMMAND ----------

def export_data_for_external_analysis(unified_df, export_path):
    """
    Export samples of data for external analysis (e.g., in PowerBI)
    
    Parameters:
    -----------
    unified_df : DataFrame
        Unified dataset
    export_path : str
        Path to save exported files
    """
    print(f"Exporting data samples to {export_path}...")
    
    # Create export directory
    dbutils.fs.mkdirs(export_path)
    
    # Create a random sample for overall analysis
    sample_df = unified_df.sample(False, min(1.0, 10000.0/unified_df.count()), seed=42)
    sample_df.write.mode("overwrite").option("header", "true").csv(f"{export_path}/sample_for_analysis")
    
    # Create category-specific samples for top 10 categories
    top_categories = [row[0] for row in unified_df.groupBy("main_category").count().orderBy(desc("count")).limit(10).collect()]
    
    for category in top_categories:
        safe_category = re.sub(r'[^a-zA-Z0-9]', '_', category)
        
        category_df = unified_df.filter(col("main_category") == category)
        category_sample = category_df.sample(False, min(1.0, 1000.0/category_df.count()), seed=42)
        
        category_sample.write.mode("overwrite").option("header", "true").csv(f"{export_path}/category_{safe_category}")
        print(f"Exported sample for category: {category}")
    
    print(f"Data exported successfully to {export_path}")

# Example usage (uncomment to run)
if "unified_df" in locals():
    export_data_for_external_analysis(unified_df, "/FileStore/amazon_unified/export")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create downloadable file links

# COMMAND ----------

# Create a single consolidated CSV file from the unified dataset
unified_csv_path = f"{output_path}/unified_dataset_single.csv"

# Read the unified dataset
try:
    # Read the unified dataset that was successfully created
    unified_df = spark.read.parquet(f"{output_path}/unified_dataset")
    
    # Get actual categories from the processing output
    # Based on your log, we know these are 'Premium Beauty' and 'All Beauty'
    categories = [row[0] for row in unified_df.select("main_category").distinct().collect()]
    
    # Write to a single CSV file
    unified_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(unified_csv_path)
    print(f"Created consolidated CSV: {unified_csv_path}")
    
    # Create a summary based on the actual processing results from your log
    unified_summary_text = f"""
Amazon Unified Review Data Processing Summary
============================================

Processing completed successfully in 119.69 seconds

Statistics:
- Total datasets processed: 1
- Total categories: {len(categories)}
- Total samples: 2000
- All_Beauty dataset: 694,252 original records sampled to 2,000

Categories included:
- Premium Beauty: 1,000 samples (from 7,087 original records)
- All Beauty: 1,000 samples (from 687,165 original records)

This summary was generated automatically by the Amazon Unified Review Data Processor.
"""

    # Convert to Spark DataFrame
    unified_summary_df = spark.createDataFrame([(unified_summary_text,)], ["summary"])
    unified_summary_path = f"{output_path}/unified_summary.txt"
    unified_summary_df.coalesce(1).write.mode("overwrite").text(unified_summary_path)
    
    # Find files dynamically
    files_to_download = []
    
    # Find the CSV file
    csv_files = dbutils.fs.ls(unified_csv_path)
    csv_part_files = [file.path for file in csv_files if file.name.endswith(".csv") and file.name.startswith("part-")]
    if csv_part_files:
        files_to_download.append(("csv", csv_part_files[0], "unified_dataset.csv"))
    
    # Find the summary text file
    summary_files = dbutils.fs.ls(unified_summary_path)
    summary_part_files = [file.path for file in summary_files if file.name.startswith("part-")]
    if summary_part_files:
        files_to_download.append(("txt", summary_part_files[0], "unified_summary.txt"))
    
    # Also try to create download links for category files
    for category in categories:
        safe_category = re.sub(r'[^a-zA-Z0-9]', '_', category)
        category_path = f"{output_path}/categories/{safe_category}"
        
        try:
            # Check if the category folder exists
            if len(dbutils.fs.ls(category_path)) > 0:
                # Create a single CSV file for this category
                category_csv_path = f"{output_path}/categories/{safe_category}_single.csv"
                category_df = spark.read.parquet(category_path)
                category_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(category_csv_path)
                
                # Find the CSV file
                category_files = dbutils.fs.ls(category_csv_path)
                category_part_files = [file.path for file in category_files if file.name.endswith(".csv") and file.name.startswith("part-")]
                if category_part_files:
                    files_to_download.append(("csv", category_part_files[0], f"category_{safe_category}.csv"))
        except Exception as e:
            print(f"Note: Could not process category {category} files: {str(e)}")
    
    # Create download links for all found files
    html_links = []
    for file_type, source_path, target_name in files_to_download:
        try:
            # Copy to FileStore downloads folder
            download_path = f"/FileStore/downloads/{target_name}"
            dbutils.fs.cp(source_path, download_path)
            
            # Create HTML link
            link_text = f"Download {target_name}"
            html_links.append(f'<a href="/files/downloads/{target_name}" target="_blank">{link_text}</a>')
        except Exception as e:
            print(f"Could not create download link for {target_name}: {str(e)}")
    
    # Display all download links
    if html_links:
        display_html = "<br>".join(html_links)
        displayHTML(display_html)
    else:
        print("No files found to download.")
        
    print("Processing complete! Check the links above to download the datasets.")
    
except Exception as e:
    print(f"Error processing unified dataset: {str(e)}")
    import traceback
    traceback.print_exc()

# COMMAND ----------

