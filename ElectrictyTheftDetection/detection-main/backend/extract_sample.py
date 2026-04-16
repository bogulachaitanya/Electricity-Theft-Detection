import pandas as pd

# Load the dataset
# Skip the second row which contains numeric indices (not data)
try:
    df = pd.read_csv("Electricity_Theft_Data.csv", skiprows=[1])
    
    # Extract first 100 rows
    sample_100 = df.head(100)
    
    # Save to a temporary file
    sample_100.to_csv("sample_100_test_data.csv", index=False)
    print("Successfully extracted 100 rows to sample_100_test_data.csv")
    
    # Also provide a small preview for the user
    # Showing just a few columns for brevity
    cols_to_show = ['CONS_NO'] + list(df.columns[1:6]) + ['CHK_STATE']
    print("\nPreview of first 5 rows (subset of columns):")
    print(sample_100[cols_to_show].head().to_string(index=False))

except Exception as e:
    print(f"Error: {e}")
