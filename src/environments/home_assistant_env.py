# path: src/environments/home_assistant_env.py

from gymnasium import spaces
from src.utils import get_logger
import gymnasium as gym
import numpy as np
import pandas as pd

# Initialize the logger for this module
LOGGER = get_logger("HomeAssistantEnv")


class HomeAssistantEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, observation_space_data, action_space_data):
        """
        Initialize the Home Assistant environment.

        :param observation_space_data: DataFrame containing the observation space (e.g., sensors).
        :param action_space_data: DataFrame containing the action space (e.g., switches, lights, locks).
        """
        LOGGER.info("Starting initialization of HomeAssistantEnv...")

        # Validate input data
        if observation_space_data is None or observation_space_data.empty:
            raise ValueError("Observation space data is empty or None. Check the processor input.")
        if action_space_data is None or action_space_data.empty:
            raise ValueError("Action space data is empty or None. Check the processor input.")

        self.observation_space_data = observation_space_data
        self.action_space_data = action_space_data

        # Encode string columns into numerical features
        self._encode_observation_space()
        self._encode_action_space()

        # Validate timestamp data
        self.current_step = 0
        self.timestamp_index = pd.Series(pd.to_datetime(self.observation_space_data["last_changed"].unique())).sort_values()
        if self.timestamp_index.empty:
            raise ValueError("Timestamp index is empty. Ensure observation_space_data contains valid timestamps.")
        LOGGER.info(f"Environment initialized with {len(self.timestamp_index)} time steps.")

    def _encode_observation_space(self):
        """
        Encode textual data into numerical features and define observation space.
        """
        self.observation_columns = self.observation_space_data.columns.difference(["last_changed"]).tolist()

        # Initialize string encoders for categorical columns
        self.string_encoders = {}
        for col in self.observation_columns:
            if pd.api.types.is_string_dtype(self.observation_space_data[col]):
                # Handle unique values and map to integers
                unique_values = self.observation_space_data[col].dropna().unique()
                if len(unique_values) > 0:
                    self.string_encoders[col] = {v: i for i, v in enumerate(unique_values, start=1)}
                    self.observation_space_data.loc[:, col] = (
                        self.observation_space_data[col]
                        .map(self.string_encoders[col])
                        .fillna(0)
                        .astype(int)
                    )
                else:
                    LOGGER.warning(f"Column '{col}' has no unique string values. Defaulting to a single class.")
                    self.string_encoders[col] = {None: 0}
                    self.observation_space_data.loc[:, col] = 0
            elif pd.api.types.is_numeric_dtype(self.observation_space_data[col]):
                # Ensure numeric columns have valid values
                self.observation_space_data.loc[:, col] = (
                    self.observation_space_data[col]
                    .fillna(0)
                    .astype(np.float32)
                )
            else:
                LOGGER.warning(f"Unexpected data type in column '{col}'. Treating as categorical data.")
                unique_values = self.observation_space_data[col].dropna().astype(str).unique()
                self.string_encoders[col] = {v: i for i, v in enumerate(unique_values, start=1)}
                self.observation_space_data.loc[:, col] = (
                    self.observation_space_data[col]
                    .astype(str)
                    .map(self.string_encoders[col])
                    .fillna(0)
                    .astype(int)
                )

        # Define observation space
        self.feature_spaces = {
            col: spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            if pd.api.types.is_numeric_dtype(self.observation_space_data[col])
            else spaces.Discrete(len(self.string_encoders.get(col, {None: 0})) + 1)
            for col in self.observation_columns
        }
        self.observation_space = spaces.Dict(self.feature_spaces)
        LOGGER.debug(f"Observation space initialized with keys: {list(self.feature_spaces.keys())}")

    def _encode_action_space(self):
        """
        Encode textual action space data into numerical features and define action space.
        """
        self.action_columns = self.action_space_data.columns.difference(["last_changed"]).tolist()

        # Initialize encoders for categorical columns
        self.action_string_encoders = {}
        discrete_action_ranges = []

        for col in self.action_columns:
            if pd.api.types.is_string_dtype(self.action_space_data[col]):
                unique_values = self.action_space_data[col].dropna().unique()
                if len(unique_values) > 0:
                    self.action_string_encoders[col] = {v: i for i, v in enumerate(unique_values, start=1)}
                    self.action_space_data.loc[:, col] = (
                        self.action_space_data[col]
                        .map(self.action_string_encoders[col])
                        .fillna(0)
                        .astype(int)
                    )
                    discrete_action_ranges.append(len(self.action_string_encoders[col]) + 1)
                else:
                    LOGGER.warning(f"Column '{col}' has no unique string values. Defaulting to a single class.")
                    self.action_string_encoders[col] = {None: 0}
                    self.action_space_data.loc[:, col] = 0
                    discrete_action_ranges.append(1)
            elif pd.api.types.is_numeric_dtype(self.action_space_data[col]):
                self.action_space_data.loc[:, col] = (
                    self.action_space_data[col]
                    .fillna(0)
                    .astype(np.float32)
                )
                discrete_action_ranges.append(1)
            else:
                LOGGER.warning(f"Unexpected data type in column '{col}'. Treating as categorical data.")
                unique_values = self.action_space_data[col].dropna().astype(str).unique()
                self.action_string_encoders[col] = {v: i for i, v in enumerate(unique_values, start=1)}
                self.action_space_data.loc[:, col] = (
                    self.action_space_data[col]
                    .astype(str)
                    .map(self.action_string_encoders[col])
                    .fillna(0)
                    .astype(int)
                )
                discrete_action_ranges.append(len(self.action_string_encoders[col]) + 1)

        # Use MultiDiscrete for the combined action space
        if discrete_action_ranges:
            self.action_space = spaces.MultiDiscrete(discrete_action_ranges)
        else:
            raise ValueError("Action space could not be initialized. No valid actions found.")

        LOGGER.debug(f"Action space initialized as MultiDiscrete with ranges: {discrete_action_ranges}")

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state and select a random day for training.
        """
        LOGGER.debug("Resetting environment...")
        super().reset(seed=seed)

        # Ensure timestamp_index is converted to datetime and normalized
        normalized_timestamps = pd.to_datetime(self.timestamp_index).dt.normalize()
        unique_days = normalized_timestamps.unique()

        # Select a random day
        random_day = np.random.choice(unique_days)
        self.timestamp_index = self.timestamp_index[normalized_timestamps == random_day]

        # Reset state tracking
        self.current_step = 0
        LOGGER.debug(f"Environment reset with data from {random_day}.")
        LOGGER.debug("Environment state reset to initial step.")

        # Get initial observation
        observation = self._get_observation()
        LOGGER.debug(f"Initial observation retrieved: {observation}")
        return observation, {}


    def step(self, actions):
        """
        Perform the given actions for all actionable entities.

        :param actions: List of actions corresponding to all actionable entities.
        """
        LOGGER.debug(f"Executing step with actions: {actions}")

        rewards = []
        for entity_idx, action in enumerate(actions):
            entity_id = self.action_columns[entity_idx]
            actual_state = self._get_current_entity_state(entity_id)
            rewards.append(1.0 if action == actual_state else -1.0)

        total_reward = sum(rewards)
        self.current_step += 1
        done = self.current_step >= len(self.timestamp_index)
        truncated = False

        LOGGER.debug(f"Step {self.current_step}: Total Reward={total_reward}, Done={done}")

        return self._get_observation(), total_reward, done, truncated, {}

    def _get_current_entity_state(self, entity_id):
        """
        Retrieve the current state of an entity.
        """
        if self.current_step < len(self.timestamp_index):
            timestamp = self.timestamp_index.iloc[self.current_step]
            state_data = self.action_space_data[(self.action_space_data["entity_id"] == entity_id) &
                                                (self.action_space_data["last_changed"] <= timestamp)]
            if not state_data.empty:
                return state_data.iloc[-1]["state"]
        return 0  # Default state if no data available
    def _get_observation(self):
        """
        Retrieve the current observation for the environment.
        """
        if self.current_step >= len(self.timestamp_index):
            LOGGER.warning("Attempted to access observation beyond available timestamps.")
            return {col: self._get_default_value(col) for col in self.observation_columns}

        timestamp = self.timestamp_index.iloc[self.current_step]
        current_data = self.observation_space_data[self.observation_space_data["last_changed"] == timestamp]

        # Build observation dictionary
        observation = {
            col: (
                np.array([current_data[col].iloc[0]], dtype=np.float32)
                if not current_data.empty and pd.notna(current_data[col].iloc[0])
                else np.array([self._get_default_value(col)], dtype=np.float32)
            )
            for col in self.observation_columns
        }
        return observation

    def _get_default_value(self, column):
        """
        Get the default value for a given observation column based on its data type.
        """
        if pd.api.types.is_numeric_dtype(self.observation_space_data[column]):
            return 0.0  # Default to 0 for numeric types
        return 0  # Default to 0 for encoded text types

    def _get_current_entity_state(self, entity_id):
        """
        Retrieve the current state of an entity.
        """
        if self.current_step < len(self.timestamp_index):
            timestamp = self.timestamp_index.iloc[self.current_step]
            state_data = self.action_space_data[
                (self.action_space_data["entity_id"] == entity_id) &
                (self.action_space_data["last_changed"] <= timestamp)
            ]
            if not state_data.empty:
                return state_data.iloc[-1]["state"]
        return 0  # Default state if no data available

    def render(self, mode="human"):
        """
        Render the environment (no-op for now).
        """
        LOGGER.info("Rendering environment (no-op).")

    def close(self):
        """
        Close the environment.
        """
        LOGGER.info("Closing environment.")
