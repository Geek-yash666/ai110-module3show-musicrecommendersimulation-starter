import os
import sys
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

def convert_csv_to_parquet(csv_file_path, parquet_file_path, chunk_size=100000):
    """
    Converts a CSV file to Parquet format.
    Uses chunking to handle large files efficiently and avoid out-of-memory errors.
    """
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at '{csv_file_path}'")
        sys.exit(1)

    print(f"Converting CSV: '{csv_file_path}' -> Parquet: '{parquet_file_path}'")
    
    if not HAS_PYARROW:
        print("Warning: 'pyarrow' is not installed. Attempting full file conversion in-memory with pandas fallback...")
        print("To run chunked/low-memory conversion, please install pyarrow: pip install pyarrow")
        try:
            df = pd.read_csv(csv_file_path, low_memory=False)
            df.to_parquet(parquet_file_path, index=False)
            print("Conversion completed successfully (in-memory).")
            return
        except Exception as e:
            print(f"Error during in-memory conversion: {e}")
            sys.exit(1)

    # Chunked conversion using pyarrow ParquetWriter
    writer = None
    chunk_reader = pd.read_csv(csv_file_path, chunksize=chunk_size, low_memory=False)
    
    try:
        for i, chunk in enumerate(chunk_reader):
            print(f"Processing chunk {i + 1}...")
            # Convert pandas DataFrame to pyarrow Table
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            
            # Initialize ParquetWriter on the first chunk
            if writer is None:
                writer = pq.ParquetWriter(parquet_file_path, table.schema, compression='snappy')
            
            writer.write_table(table)
        
        if writer is not None:
            writer.close()
        print("Conversion completed successfully (chunked).")
        
    except Exception as e:
        if writer is not None:
            writer.close()
        print(f"Error during chunked conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Default paths matching project files
    default_csv = os.path.join(os.path.dirname(__file__), "tracks.csv")
    default_parquet = os.path.join(os.path.dirname(__file__), "tracks.parquet")
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else default_csv
    parquet_path = sys.argv[2] if len(sys.argv) > 2 else default_parquet

    convert_csv_to_parquet(csv_path, parquet_path)
