from ingestor import DataIngestor
from preprocessor import PreprocessingPipeline
import os
import argparse

def orchestrate_loading(raw_input, processed_output):
    """Workflow to ingest and preprocess the dataset."""
    print("Step 1: Ingesting Raw Data...")
    ingestor = DataIngestor(raw_input)
    df = ingestor.load_sgcc()
    ingestor.validate_schema(df)
    
    print("\nStep 2: Preprocessing & Cleaning...")
    # Preprocessor expects to read from file, so we ensure raw is in place
    pipeline = PreprocessingPipeline(raw_input, processed_output)
    pipeline.load_csv()
    pipeline.clip_outliers()
    pipeline.impute_gaps()
    pipeline.normalize_per_household()
    pipeline.save_processed_data()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/sgcc/data set.csv")
    parser.add_argument("--out", default="data/processed/clean_sgcc.csv")
    args = parser.parse_args()
    
    orchestrate_loading(args.raw, args.out)
