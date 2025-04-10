# Databricks notebook source
# This notebook processes Amazon review data using Spark on Databricks. It performs the following operations:
# 1. Data Ingestion: Loads Amazon review and metadata JSONL files
# 2. Data Cleaning: Handles missing values, duplicates, and timestamp issues
# 3. Join Operations: Combines reviews with product metadata
# 4. Sampling: Creates balanced samples across product categories
# 5. Output: Saves processed data in a structured format

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment Setup and Configuration

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, lit, count, desc, avg, year, month, regexp_replace, explode, array
from pyspark.sql.types import StringType, IntegerType, DoubleType, ArrayType, StructType, StructField
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configure Spark for optimal performance
spark.conf.set("spark.sql.shuffle.partitions", 100)
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.caseSensitive", "true")

# Define default parameters
DEFAULT_SAMPLE_SIZE = 10000
DEFAULT_INCLUDE_TIMESTAMP = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define Core Functions for Data Processing

# COMMAND ----------


def process_dataset(reviews_path, metadata_path, output_path, sample_size=DEFAULT_SAMPLE_SIZE, include_timestamp=DEFAULT_INCLUDE_TIMESTAMP):
    """
    Process a single Amazon dataset (reviews + metadata)
    
    Parameters:
    -----------
    reviews_path : str
        Path to the reviews JSONL file
    metadata_path : str
        Path to the metadata JSONL file
    output_path : str
        Path where processed data will be saved
    sample_size : int
        Maximum number of samples to take per category
    include_timestamp : bool
        Whether to include timestamp information
        
    Returns:
    --------
    dict
        Processing results summary
    """
    # Extract dataset name from filename
    dataset_name = os.path.basename(reviews_path).split('.')[0]
    
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"  Reviews: {reviews_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Output: {output_path}")
    print(f"  Sample size: {sample_size} per category")
    print(f"  Include timestamp: {include_timestamp}")
    
    try:
        # Step 1: Load data
        print("Loading datasets...")
        reviews_df = spark.read.option("multiLine", "false").json(reviews_path)
        metadata_df = spark.read.option("multiLine", "false").json(metadata_path)
        
        # Display basic statistics
        print(f"Reviews dataset shape: {(reviews_df.count(), len(reviews_df.columns))}")
        print(f"Metadata dataset shape: {(metadata_df.count(), len(metadata_df.columns))}")
        
        # Step 2: Clean reviews data
        print("\nCleaning reviews data...")
        
        # Display schema before cleaning
        print("Reviews schema before cleaning:")
        reviews_df.printSchema()
        
        # Check for missing values in key columns
        print("\nMissing values in key columns (reviews):")
        reviews_df.select([count(col(c)).alias(c) for c in ["asin", "user_id", "rating"]]).show()
        
        # Drop rows with missing values in key columns
        reviews_df = reviews_df.na.drop(subset=["asin", "user_id"])
        
        # Drop duplicate reviews for the same product by the same user
        reviews_counts_before = reviews_df.count()
        reviews_df = reviews_df.dropDuplicates(["asin", "user_id"])
        reviews_counts_after = reviews_df.count()
        print(f"Removed {reviews_counts_before - reviews_counts_after} duplicate reviews")
        
        # Handle timestamp based on parameter
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
        print("\nCleaning metadata...")
        
        # Display schema before cleaning
        print("Metadata schema before cleaning:")
        metadata_df.printSchema()
        
        # Check for missing values in key columns
        print("\nMissing values in key columns (metadata):")
        metadata_df.select([count(col(c)).alias(c) for c in ["parent_asin", "title"]]).show()
        
        # Drop rows with missing parent_asin
        metadata_df = metadata_df.na.drop(subset=["parent_asin"])
        
        # [D] Ensure main_category exists (extract from categories if needed)
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
        print("\nJoining reviews with metadata...")
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
        
        # Basic statistics after joining
        print("\nColumn statistics after joining:")
        result_df.describe().show()
        
        # Step 5: Sample by category
        print("\nSampling records by category...")
        
        # Get categories
        categories = [row[0] for row in result_df.select("main_category").distinct().collect()]
        print(f"Found {len(categories)} categories")
        
        # Display category distribution
        print("\nCategory distribution before sampling:")
        result_df.groupBy("main_category").count().orderBy(desc("count")).show(10)
        
        # Sample for each category
        sampled_dfs = []
        total_samples = 0
        
        for category in categories:
            # Filter for this category
            category_df = result_df.filter(col("main_category") == category)
            category_count = category_df.count()
            
            if category_count <= sample_size:
                print(f"  Category '{category}' has {category_count} records, keeping all")
                sampled_dfs.append(category_df)
                total_samples += category_count
            else:
                # Use random sampling
                print(f"  Category '{category}' has {category_count} records, sampling {sample_size}")
                sampled_df = category_df.orderBy(rand(42)).limit(sample_size)
                sampled_dfs.append(sampled_df)
                total_samples += sample_size
        
        # Union all sampled dataframes
        if sampled_dfs:
            final_df = sampled_dfs[0]
            for df in sampled_dfs[1:]:
                final_df = final_df.union(df)
        else:
            final_df = result_df
        
        final_count = final_df.count()
        print(f"Final dataset has {final_count} rows")
        
        # Check category distribution after sampling
        print("\nCategory distribution after sampling:")
        final_df.groupBy("main_category").count().orderBy(desc("count")).show(10)
        
        # Step 6: Save processed data
        print("\nSaving processed data...")
        
        # Create output path if it doesn't exist
        dbutils.fs.mkdirs(output_path)
        
        # Save as parquet (efficient Spark format)
        parquet_path = f"{output_path}/{dataset_name}"
        final_df.write.mode("overwrite").parquet(parquet_path)
        print(f"Saved processed data to: {parquet_path}")
        
        # Also save CSV for easy access (may be slower for very large datasets)
        csv_path = f"{output_path}/{dataset_name}_csv"
        final_df.write.mode("overwrite").option("header", "true").csv(csv_path)
        print(f"Saved CSV data to: {csv_path}")
        
        # Return processing summary
        return {
            "dataset": dataset_name,
            "categories": len(categories),
            "total_records": join_count,
            "sampled_records": total_samples,
            "status": "Success",
            "output_path": parquet_path
        }
        
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "dataset": dataset_name,
            "status": f"Failed: {str(e)}"
        }


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Quality Visualizations

# COMMAND ----------

def visualize_data_quality(df, dataset_name):
    """
    Create data quality visualizations for a dataset
    
    Parameters:
    -----------
    df : DataFrame
        Spark DataFrame to visualize
    dataset_name : str
        Name of the dataset
    """
    # Convert to pandas for visualization
    pdf = df.toPandas()
    
    # Set up the figure
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Rating distribution
    plt.subplot(2, 2, 1)
    sns.countplot(x='rating', data=pdf)
    plt.title(f'Rating Distribution - {dataset_name}')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    
    # Plot 2: Categories distribution (top 10)
    plt.subplot(2, 2, 2)
    category_counts = pdf['main_category'].value_counts().head(10)
    sns.barplot(x=category_counts.values, y=category_counts.index)
    plt.title(f'Top 10 Categories - {dataset_name}')
    plt.xlabel('Count')
    
    # Plot 3: Missing values heatmap
    plt.subplot(2, 2, 3)
    missing = pdf.isnull().sum() / len(pdf) * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        sns.barplot(x=missing.values, y=missing.index)
        plt.title(f'Missing Values (%) - {dataset_name}')
        plt.xlabel('Percent Missing')
    else:
        plt.text(0.5, 0.5, 'No missing values', horizontalalignment='center',
                 verticalalignment='center', transform=plt.gca().transAxes)
        plt.title(f'Missing Values (%) - {dataset_name}')
    
    # Plot 4: Verified purchase distribution (if available)
    plt.subplot(2, 2, 4)
    if 'verified_purchase' in pdf.columns:
        sns.countplot(x='verified_purchase', data=pdf)
        plt.title(f'Verified Purchase Distribution - {dataset_name}')
    else:
        plt.text(0.5, 0.5, 'No verified_purchase field', horizontalalignment='center',
                 verticalalignment='center', transform=plt.gca().transAxes)
        plt.title('Verified Purchase Distribution')
    
    plt.tight_layout()
    display(plt.gcf())
    plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Main Execution 

# COMMAND ----------

# Example usage (modify paths as needed for your Databricks environment)
reviews_path = "dbfs:/FileStore/Digital_Music.jsonl"
metadata_path = "dbfs:/FileStore/meta_Digital_Music.jsonl"
output_path = "dbfs:/FileStore/amazon_processed"
custom_sample_size = 100
custom_include_timestamp = False

# Process a single dataset
results = process_dataset(
    reviews_path=reviews_path,
    metadata_path=metadata_path,
    output_path=output_path,
    sample_size=custom_sample_size,  # Adjust as needed
    include_timestamp=custom_include_timestamp
)

# Load the processed data for visualization
if "output_path" in results:
    processed_df = spark.read.parquet(results["output_path"])
    
    # Display data quality visualizations
    visualize_data_quality(processed_df, results["dataset"])
    
    # Display sample of the processed data
    print("\nSample of processed data:")
    display(processed_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Advanced Data Cleaning (Optional)

# COMMAND ----------

def advanced_cleaning(df):
    """
    Perform advanced data cleaning operations
    
    Parameters:
    -----------
    df : DataFrame
        Spark DataFrame to clean
        
    Returns:
    --------
    DataFrame
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original
    result_df = df
    
    # Clean text fields - remove special characters and extra whitespace
    if "review_text" in df.columns:
        result_df = result_df.withColumn(
            "review_text_clean", 
            regexp_replace(col("review_text"), "[^a-zA-Z0-9\\s]", " ")
        )
        result_df = result_df.withColumn(
            "review_text_clean", 
            regexp_replace(col("review_text_clean"), "\\s+", " ")
        )
    
    # Convert price to numeric (if exists)
    if "price" in df.columns:
        # First check if it's already numeric
        if str(df.schema["price"].dataType) == "StringType":
            # Remove currency symbol and convert to double
            result_df = result_df.withColumn(
                "price_numeric", 
                regexp_replace(col("price"), "[$,]", "").cast(DoubleType())
            )
        else:
            # Already numeric, just rename
            result_df = result_df.withColumn("price_numeric", col("price"))
    
    # Normalize rating to 0-1 scale (for ML)
    if "rating" in df.columns:
        result_df = result_df.withColumn(
            "rating_normalized", 
            col("rating") / 5.0
        )
    
    return result_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Saving Processing Summary

# COMMAND ----------

def save_processing_summary(results, output_path):
    """
    Save processing summary as a text file
    
    Parameters:
    -----------
    results : dict
        Processing results
    output_path : str
        Path to save summary
    """
    summary_text = f"""
Amazon Review Data Processing Summary
====================================

Processing completed for dataset: {results.get('dataset', 'Unknown')}
Status: {results.get('status', 'Unknown')}

Statistics:
- Categories: {results.get('categories', 'N/A')}
- Total records: {results.get('total_records', 'N/A')}
- Sampled records: {results.get('sampled_records', 'N/A')}
- Output path: {results.get('output_path', 'N/A')}

Processing Parameters:
- Sample size per category: {DEFAULT_SAMPLE_SIZE}
- Included timestamp: {DEFAULT_INCLUDE_TIMESTAMP}

This summary was generated automatically by the Amazon Review Data Processor.
    """
    
    # Convert to Spark DataFrame
    summary_df = spark.createDataFrame([(summary_text,)], ["summary"])
    
    # Save as text file
    summary_path = f"{output_path}/processing_summary.txt"
    summary_df.write.mode("overwrite").text(summary_path)
    print(f"Saved processing summary to: {summary_path}")

# Uncomment to save summary
save_processing_summary(results, output_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create downloadable file links

# COMMAND ----------

# Create a single consolidated CSV file
dataset_name = os.path.basename(reviews_path).split('.')[0]
csv_df = spark.read.parquet(f"{output_path}/{dataset_name}")
single_csv_path = f"{output_path}/{dataset_name}_single.csv"
csv_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(single_csv_path)

# Generate and save processing summary
summary_text = f"""
Amazon Review Data Processing Summary
====================================

Processing completed for dataset: {dataset_name}
Status: Success

Statistics:
- Categories: {results.get('categories', 'N/A')}
- Total records: {results.get('total_records', 'N/A')}
- Sampled records: {results.get('sampled_records', 'N/A')}
- Output path: {results.get('output_path', 'N/A')}

Processing Parameters:
- Sample size per category: {custom_sample_size}
- Included timestamp: {custom_include_timestamp}

This summary was generated automatically by the Amazon Review Data Processor.
"""

# Convert to Spark DataFrame
summary_df = spark.createDataFrame([(summary_text,)], ["summary"])
summary_single_path = f"{output_path}/{dataset_name}_summary.txt"
summary_df.coalesce(1).write.mode("overwrite").text(summary_single_path)

# Find files dynamically
files_to_download = []

# Find the CSV file
csv_files = dbutils.fs.ls(single_csv_path)
csv_part_files = [file.path for file in csv_files if file.name.endswith(".csv") and file.name.startswith("part-")]
if csv_part_files:
    files_to_download.append(("csv", csv_part_files[0], f"{dataset_name}_data.csv"))

# Find the summary text file
summary_files = dbutils.fs.ls(summary_single_path)
summary_part_files = [file.path for file in summary_files if file.name.startswith("part-")]
if summary_part_files:
    files_to_download.append(("txt", summary_part_files[0], f"{dataset_name}_summary.txt"))

# Create download links for all found files
html_links = []
for file_type, source_path, target_name in files_to_download:
    # Copy to FileStore downloads folder
    download_path = f"/FileStore/downloads/{target_name}"
    dbutils.fs.cp(source_path, download_path)
    
    # Create HTML link
    link_text = f"Download {dataset_name} {file_type.upper()} File"
    html_links.append(f'<a href="/files/downloads/{target_name}" target="_blank">{link_text}</a>')

# Display all download links
if html_links:
    display_html = "<br>".join(html_links)
    displayHTML(display_html)
else:
    print("No files found to download.")

# COMMAND ----------

