from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
from src.utils import get_logger
import numpy as np


# Linear scheduler for RL agent parameters
def linear_schedule(initial_value, final_value=0.0):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0), "linear_schedule work only with positive decreasing values"

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return final_value + progress * (initial_value - final_value)

    return func

# AutoSave Callback
class AutoSave(BaseCallback):
    """
    Callback for saving a model, it is saved every ``check_freq`` steps

    :param check_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :filename_prefix: (str) Filename prefix
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, num_envs: int, save_path: str, filename_prefix: str="", verbose: int=1):
        super(AutoSave, self).__init__(verbose)
        self.check_freq = int(check_freq / num_envs)
        self.num_envs = num_envs
        self.save_path_base = Path(save_path)
        self.filename = filename_prefix + "autosave_"

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if self.verbose > 0:
                print("Saving latest model to {}".format(self.save_path_base))
            # Save the agent
            self.model.save(self.save_path_base / (self.filename + str(self.n_calls * self.num_envs)))

        return True

from stable_baselines3.common.callbacks import BaseCallback
from src.utils import get_logger
import numpy as np


class RewardsCallback(BaseCallback):
    def __init__(self, processor, automations, penalty_weight=0.5, reward_scale=1.0, verbose=0):
        """
        Callback for managing and shaping rewards based on automations and actionable entities.

        :param processor: HomeAssistantProcessor instance.
        :param automations: List of automations parsed into triggers and actions.
        :param penalty_weight: Weight for penalties in the reward calculation.
        :param reward_scale: Scaling factor for rewards.
        :param verbose: Verbosity level.
        """
        super(RewardsCallback, self).__init__(verbose)
        self.LOGGER = get_logger("RewardsCallback")
        self.processor = processor
        self.automations = automations
        self.penalty_weight = penalty_weight
        self.reward_scale = reward_scale
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_length = 0

    def _evaluate_action_against_automations(self, action_label):
        """
        Evaluate the agent's action against the automation rules.

        :param action_label: The label of the action performed by the agent.
        :return: Match count based on automation evaluation (no penalties).
        """
        match_count = 0
        for automation in self.automations:
            triggers = automation.get("triggers", [])
            actions = automation.get("actions", [])

            # Check if action_label matches an automation action
            if action_label in [action.get("entity_id") for action in actions]:
                # Validate that triggers match the current state
                trigger_match = all(
                    self.processor.get_sensor_state(trigger["entity_id"]) == trigger.get("to")
                    for trigger in triggers if "entity_id" in trigger and "to" in trigger
                )
                if trigger_match:
                    match_count += 1
        return match_count

    def _on_step(self) -> bool:
        actions = self.locals["actions"]  # List of actions for all actionable entities
        done = self.locals["dones"]

        # Get current states for all actionable entities
        current_states = {
            entity_id: self.processor.get_sensor_state(entity_id)
            for entity_id in self.processor.actionable_entities
        }

        match_count = 0
        penalty_count = 0
        missed_count = 0
        total_entities = len(current_states)

        # Evaluate each action against its respective entity
        step_rewards = []
        for entity_idx, action in enumerate(actions[0]):  # Actions array corresponds to all entities
            actionable_entity_label = (
                self.processor.actionable_entities[entity_idx]
                if entity_idx < len(self.processor.actionable_entities)
                else None
            )

            if actionable_entity_label and actionable_entity_label in current_states:
                actual_state = current_states[actionable_entity_label]
                if action == actual_state:
                    match_count += 1
                    step_rewards.append(1.0)  # Positive reward for correct action
                else:
                    penalty_count += 1
                    step_rewards.append(-1.0)  # Penalty for incorrect action
                    self.LOGGER.debug(
                        f"Mismatch: Actioned '{actionable_entity_label}' to state {action}, "
                        f"but actual state is {actual_state}."
                    )
            elif actionable_entity_label:
                self.LOGGER.debug(f"Ignoring invalid action: '{actionable_entity_label}'")
                step_rewards.append(0.0)  # No reward for invalid actions

        # Count missed states
        for entity_id, actual_state in current_states.items():
            if entity_id not in self.processor.actionable_entities or actual_state != 1.0:
                missed_count += 1

        # Aggregate step rewards and scale
        step_reward = sum(step_rewards) * self.reward_scale

        # Update environment's reward
        self.locals["rewards"][0] += step_reward

        # Track episode metrics
        self.current_episode_reward += step_reward
        self.episode_length += 1

        # Log detailed debug info
        self.LOGGER.debug(
            f"Step {self.num_timesteps}: Matches={match_count}, Penalties={penalty_count}, Missed={missed_count}, "
            f"Step Reward={step_reward:.2f}, Updated Reward={self.locals['rewards'][0]:.2f}"
        )

        # Handle episode end
        if done[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.LOGGER.debug(
                f"Episode ended. Total Reward: {self.current_episode_reward:.2f}, Length: {self.episode_length}"
            )
            self.current_episode_reward = 0
            self.episode_length = 0

        return True
