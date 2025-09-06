import os
import glob
import pandas as pd


# DIR

path = "./data"
# Sample DataFrame


def combine_csv_files(directory):
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {directory}")

    # Read and combine all CSV files
    dfs = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)

    # Ensure 'Date' column is in datetime format
    if 'Date' in combined_df.columns:
        combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='%m/%d/%Y')

    # Convert numerical columns to float (assuming they are yields or rates)
    numerical_columns = [col for col in combined_df.columns if col not in ['Date']]
    combined_df[numerical_columns] = combined_df[numerical_columns].astype(float)

    # Re order columns
    order = ['Date', '1 Mo', '1.5 Month', '2 Mo', '3 Mo', '4 Mo', '6 Mo',
             '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
    combined_df = combined_df[order]
    # Rename columns for consistency
    combined_df.columns = combined_df.columns.str.replace('Month', 'Mo')
    combined_df.columns = combined_df.columns.str.replace('1.5 Mo', '1_5 Mo')
    combined_df.columns = combined_df.columns.str.replace(' ', '_')

    print(f"Combined {len(csv_files)} CSV files with {len(combined_df)} rows")
    return combined_df


if __name__ == "__main__":
    # Example usage
    try:
        combined_df = combine_csv_files(path)
        print(combined_df.head())
    except ValueError as e:
        print(e)
