import pandas as pd
import numpy as np

def classify_action_space(data_file, output_file, unknown_conditions_file):
    """
    Classifies each entity in the processed data as requiring discrete or continuous actions.

    Args:
        data_file (str): Path to the processed CSV file.
        output_file (str): Path to save the action space mapping.
        unknown_conditions_file (str): Path to save unknown conditions for debugging.
    """
    # Load the processed data
    print(f"Loading data from {data_file}...")
    data = pd.read_csv(data_file)

    # Dictionary to store the action space classification
    action_space = {}
    unknown_conditions = []

    # Group data by entity_id
    grouped = data.groupby("entity_id")

    for entity, group in grouped:
        # Analyze the `state` column
        states = group["state"].dropna().unique()

        # Check domain type from entity_id
        if entity.startswith("automation") or entity.startswith("media_player"):
            # Automation and media players with on/off states
            if set(states).issubset({"on", "off", "unavailable"}):
                action_space[entity] = "discrete"
            else:
                action_space[entity] = "none"
                unknown_conditions.append((entity, states))
        elif entity.startswith("input_button"):
            # Buttons are always discrete (triggered events)
            action_space[entity] = "discrete"
        elif entity.startswith("binary_sensor"):
            # Binary states (on/off)
            if set(states).issubset({"on", "off"}):
                action_space[entity] = "discrete"
            else:
                action_space[entity] = "none"
                unknown_conditions.append((entity, states))
        elif all(isinstance(state, (int, float, np.float64)) for state in states):
            # Numeric states
            if len(states) > 2:  # Assume continuous if there are more than 2 unique states
                action_space[entity] = "continuous"
            else:  # Otherwise, treat as discrete
                action_space[entity] = "discrete"
        else:
            # Non-numeric or other categorical states
            action_space[entity] = "discrete"
            unknown_conditions.append((entity, states))

    # Save the action space mapping to a CSV file
    action_space_df = pd.DataFrame(action_space.items(), columns=["entity_id", "action_space"])
    action_space_df.to_csv(output_file, index=False)
    print(f"Action space mapping saved to {output_file}")

    # Save unknown conditions for debugging
    if unknown_conditions:
        unknown_conditions_df = pd.DataFrame(unknown_conditions, columns=["entity_id", "states"])
        unknown_conditions_df.to_csv(unknown_conditions_file, index=False)
        print(f"Unknown conditions saved to {unknown_conditions_file}")
    else:
        print("No unknown conditions detected.")

if __name__ == "__main__":
    # Input and output file paths
    input_file = "data/processed_logs/processed_data.csv"  # Path to the processed data
    output_file = "data/processed_logs/action_space_mapping.csv"  # Path to save the mapping
    unknown_conditions_file = "data/processed_logs/unknown_conditions.csv"  # Path to save unknown conditions

    # Run the classification
    classify_action_space(input_file, output_file, unknown_conditions_file)
