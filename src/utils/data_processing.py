import pandas as pd
from dateutil.parser import parse

def process_raw_logs(input_file, output_file):
    """
    Processes raw Home Assistant logs into a clean dataset.

    Args:
        input_file (str): Path to the raw CSV file.
        output_file (str): Path to save the processed data.
    """
    # Define columns to keep
    columns_to_keep = [
        "entity_id", "state", "last_changed",
        "current_temperature", "hvac_action", "target_temp_high",
        "target_temp_low", "temperature"
    ]

    # Read the large CSV file in chunks
    chunk_size = 100000  # Adjust for memory efficiency
    processed_chunks = []

    for chunk in pd.read_csv(input_file, usecols=columns_to_keep, chunksize=chunk_size, low_memory=False):
        # Convert `last_changed` to datetime
        chunk["last_changed"] = pd.to_datetime(chunk["last_changed"], errors="coerce")

        # Drop rows with missing critical data
        chunk = chunk.dropna(subset=["entity_id", "last_changed"])

        # Normalize `state` values with specific rules for entity types
        def normalize_state(value, entity_id):
            if pd.isna(value) or value == '':
                # Handle missing or empty states
                if entity_id.startswith(("sensor.", "device_tracker.", "person.", "climate.", "button.")):
                    return "unknown"  # Default for these specific entity types
                return "0"  # Default as string for other entity types

            # Retain location-based strings or operational modes for climate entities
            if entity_id.startswith("climate."):
                return str(value).strip().lower()  # Standardize as lowercase string
            
            # Handle datetime-like strings
            if isinstance(value, str) and ("T" in value and ("Z" in value or "+" in value)):
                try:
                    return parse(value).isoformat()  # Standardize to ISO 8601
                except ValueError:
                    return str(value).strip()  # Keep as string if parsing fails

            # Handle binary states
            if value in ["on", "true"]:
                return "1"
            elif value in ["off", "false"]:
                return "0"
            
            # Retain as string for location-based entities
            if entity_id.startswith(("person.", "device_tracker.")):
                return str(value).strip()

            # Default numeric and string handling
            return str(value).strip()

        # Apply normalization to `state`
        chunk["state"] = chunk.apply(lambda row: normalize_state(row["state"], row["entity_id"]), axis=1)

        # Append processed chunk
        processed_chunks.append(chunk)

    # Concatenate all processed chunks
    processed_data = pd.concat(processed_chunks)

    # Fill missing `state` values for special cases
    processed_data["state"] = processed_data["state"].fillna("unknown")  # Default for any remaining NaN

    # Save the processed data to a new CSV file
    processed_data.to_csv(output_file, index=False)
    print(f"Processed logs saved to: {output_file}")

if __name__ == "__main__":
    input_file = "data/raw_logs/history.csv"
    output_file = "data/processed_logs/processed_data.csv"
    process_raw_logs(input_file, output_file)
