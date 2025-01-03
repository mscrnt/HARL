# path: src/environments/ha_simulator.py

import pandas as pd
import numpy as np
from datetime import timedelta
from src.utils import get_logger

LOGGER = get_logger("HomeAssistantProcessor")

class HomeAssistantProcessor:
    def __init__(self, log_file, action_mapping_file=None, epoch_start=None, epoch_end=None):
        """
        Initialize and process Home Assistant logs.

        :param log_file: Path to the log file (CSV).
        :param action_mapping_file: Path to action space mapping file (CSV).
        :param epoch_start: Optional start of the epoch (datetime or string).
        :param epoch_end: Optional end of the epoch (datetime or string).
        """
        LOGGER.info("Initializing HomeAssistantProcessor...")
        self.log_file = log_file
        self.action_mapping_file = action_mapping_file

        # Load data and mappings
        self.log_data = self._load_and_normalize_logs(log_file)
        self.action_mapping = self._load_action_mapping() if action_mapping_file else None

        # Dynamically populate entity-specific binary states
        self.entity_state_mappings = self._generate_entity_state_mappings()

        self.epoch_start = pd.to_datetime(epoch_start) if epoch_start else self.log_data["last_changed"].min()
        self.epoch_end = pd.to_datetime(epoch_end) if epoch_end else self.epoch_start + timedelta(days=1)
        self.time_index = pd.date_range(self.epoch_start, self.epoch_end, freq="1s", inclusive="left")
        LOGGER.info(f"Epoch range set: {self.epoch_start} to {self.epoch_end}.")

        # Classify keys directly into observations and actions
        self.classify_keys()

    def get_sensor_state(self, entity_id):
        """
        Retrieve the latest state of the specified entity.

        :param entity_id: The entity ID for which to retrieve the state.
        :return: The latest state value for the specified entity or None if not found.
        """
        if entity_id not in self.log_data["entity_id"].values:
            LOGGER.warning(f"Entity '{entity_id}' not found in log data.")
            return None

        # Retrieve the most recent state for the entity
        latest_state_row = self.log_data[self.log_data["entity_id"] == entity_id].iloc[-1]
        return latest_state_row["state"]

    def _load_and_normalize_logs(self, log_file):
        """
        Load logs and normalize timestamps to the nearest second.
        """
        LOGGER.debug(f"Loading logs from: {log_file}")
        data = pd.read_csv(log_file, low_memory=False)

        # Normalize timestamps and sort data
        data["last_changed"] = pd.to_datetime(data["last_changed"], errors="coerce").dt.floor("s")
        data = data.dropna(subset=["last_changed"]).sort_values("last_changed")
        LOGGER.debug(f"Loaded and normalized {len(data)} rows of log data.")
        return data

    def _load_action_mapping(self):
        """
        Load action space mapping file if provided.
        """
        try:
            mapping = pd.read_csv(self.action_mapping_file)
            LOGGER.debug(f"Loaded {len(mapping)} rows of action space mapping.")
            return mapping
        except Exception as e:
            LOGGER.error(f"Failed to load action mapping file: {e}")
            return None

    def _generate_entity_state_mappings(self):
        """
        Generate entity-specific binary state mappings based on the log data.
        """
        LOGGER.info("Generating entity-specific state mappings...")
        predefined_states = {
            "on": 1.0, "off": 0.0, "true": 1.0, "false": 0.0,
            "locked": 1.0, "unlocked": 0.0, "home": 1.0, "not_home": 0.0
        }

        entity_state_mappings = {}

        for _, row in self.log_data.iterrows():
            entity_id = row["entity_id"]
            state = str(row["state"]).strip().lower()

            if entity_id not in entity_state_mappings:
                entity_state_mappings[entity_id] = predefined_states.copy()

            if state not in entity_state_mappings[entity_id]:
                if state.isnumeric() or state.replace('.', '', 1).isdigit():
                    entity_state_mappings[entity_id][state] = float(state)
                else:
                    LOGGER.warning(f"Unmapped state for entity '{entity_id}': '{state}'. Adding as 0.0 by default.")
                    entity_state_mappings[entity_id][state] = 0.0

        LOGGER.info(f"Generated state mappings for {len(entity_state_mappings)} entities.")
        return entity_state_mappings

    def classify_keys(self):
        """
        Classify raw data into observation and action keys based on prefixes.
        """
        if self.log_data.empty:
            LOGGER.warning("Log data is empty. Ensure the log file contains valid data.")
            self.observation_data = pd.DataFrame()
            self.action_data = pd.DataFrame()
            return

        self.log_data["entity_type"] = self.log_data["entity_id"].apply(
            lambda x: self._flag_keys(x) if pd.notna(x) else "other"
        )

        # Normalize all state data
        self.log_data["state"] = self.log_data.apply(
            lambda row: self._normalize_state(row["state"], row["entity_id"]),
            axis=1
        )

        # Handle missing or unexpected columns in the action data
        action_columns = ["hvac_action", "target_temp_high", "target_temp_low", "temperature"]
        for col in action_columns:
            if col not in self.log_data.columns:
                LOGGER.warning(f"Missing column '{col}' in log data. Filling with default values.")
                self.log_data[col] = None  # Default to None for missing columns

        # Split into observations and actions
        self.observation_data = self.log_data[self.log_data["entity_type"] == "observation"]
        self.action_data = self.log_data[self.log_data["entity_type"] == "action"]

        # Ensure no empty DataFrame issues
        if self.observation_data.empty:
            LOGGER.warning("No observation data found. Check the log file for valid observation entities.")
        if self.action_data.empty:
            LOGGER.warning("No action data found. Check the log file for valid action entities.")

        LOGGER.debug(f"Extracted {len(self.observation_data)} observations and {len(self.action_data)} actions.")

    def _flag_keys(self, entity_id):
        """
        Classify entity IDs into observation or action keys based on prefixes.
        """
        observation_keys = ["automation", "binary_sensor", "camera", "device_tracker", "sensor", "time"]
        action_keys = ["button", "climate", "input_boolean", "light", "lock", "media_player"]

        if any(entity_id.startswith(prefix) for prefix in observation_keys):
            return "observation"
        elif any(entity_id.startswith(prefix) for prefix in action_keys):
            return "action"
        return "other"

    def _normalize_state(self, state, entity_id):
        """
        Normalize state values using entity-specific mappings.

        :param state: State value from the log.
        :param entity_id: Corresponding entity ID for context.
        :return: Normalized state value.
        """
        if pd.isna(state) or state == "":
            return 0.0

        state = str(state).strip().lower()
        if entity_id in self.entity_state_mappings and state in self.entity_state_mappings[entity_id]:
            return self.entity_state_mappings[entity_id][state]

        LOGGER.warning(f"State '{state}' for entity '{entity_id}' is not mapped. Defaulting to 0.0.")
        return 0.0

    @property
    def actionable_entities(self):
        """
        Extract unique actionable entities from action data.
        """
        if self.action_data.empty:
            return []
        return self.action_data["entity_id"].unique().tolist()

    def pad_environment(self):
        """
        Pad the environment to create a full 24-hour dataset.
        """
        LOGGER.debug("Padding environment for the full 24-hour epoch...")

        # Generate a full 24-hour time index
        padded_environment = pd.DataFrame({"last_changed": self.time_index})

        # Merge observations and actions
        observations = self.observation_data[["last_changed", "entity_id", "state"]]
        actions = self.action_data[["last_changed", "entity_id", "state"]]
        combined_data = pd.concat([observations, actions], ignore_index=True)

        # Merge into the padded environment
        padded_environment = padded_environment.merge(combined_data, on="last_changed", how="left")

        # Normalize states and classify entities
        padded_environment["state"] = padded_environment.apply(
            lambda row: self._normalize_state(row["state"], row["entity_id"]), axis=1
        )
        padded_environment["entity_type"] = padded_environment["entity_id"].apply(
            lambda x: self._flag_keys(x) if pd.notna(x) else "other"
        )

        LOGGER.info(f"Padded environment created with {len(padded_environment)} rows.")
        return padded_environment.fillna({"state": 0.0, "entity_id": "unknown", "entity_type": "other"})

    def get_observation_space(self):
        """
        Extract observation space directly without padding.
        """
        if self.observation_data.empty:
            LOGGER.warning("Observation space data is empty. Returning an empty DataFrame.")
            return pd.DataFrame()
        return self.observation_data

    def get_action_space(self):
        """
        Extract action space directly without padding.
        """
        if self.action_data.empty:
            LOGGER.warning("Action space data is empty. Returning an empty DataFrame.")
            return pd.DataFrame()
        return self.action_data
