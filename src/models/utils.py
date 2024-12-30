from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
from src.utils import get_logger


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

class RewardsCallback(BaseCallback):
    def __init__(self, processor, automations, actionable_entities, penalty_weight=0.5, verbose=0):
        """
        Callback for managing and shaping rewards based on automations and actionable entities.

        :param processor: HomeAssistantProcessor instance.
        :param automations: List of automations parsed into triggers and actions.
        :param actionable_entities: List of actionable entities (e.g., switches, lights, locks).
        :param penalty_weight: Weight for penalties in the reward calculation.
        :param verbose: Verbosity level.
        """
        super(RewardsCallback, self).__init__(verbose)
        self.LOGGER = get_logger("RewardCallback")
        self.processor = processor  # Use processor for environment state data
        self.automations = automations
        self.actionable_entities = actionable_entities
        self.penalty_weight = penalty_weight
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

            # Check if the action matches any valid action in the automation
            if action_label in [action.get("entity_id") for action in actions]:
                # Validate if the current state matches the automation trigger
                trigger_match = all(
                    self._evaluate_trigger(trigger)
                    for trigger in triggers if "entity_id" in trigger and "to" in trigger
                )
                if trigger_match:
                    match_count += 1

        return match_count

    def _evaluate_trigger(self, trigger):
        """
        Evaluate a single trigger condition.

        :param trigger: A trigger dictionary containing 'entity_id' and 'to' conditions.
        :return: True if the trigger matches the processor state, otherwise False.
        """
        entity_ids = trigger["entity_id"]
        target_state = trigger.get("to")

        # Handle cases where 'entity_id' is a list
        if isinstance(entity_ids, list):
            return all(
                self.processor.get_sensor_state(entity_id) == target_state
                for entity_id in entity_ids
            )

        # Handle single entity_id
        return self.processor.get_sensor_state(entity_ids) == target_state

    def _on_step(self) -> bool:
        action = self.locals["actions"]
        done = self.locals["dones"]

        # Decode the actionable entity
        actionable_entity_index = action[0]
        actionable_entity_label = (
            self.actionable_entities[actionable_entity_index]
            if actionable_entity_index < len(self.actionable_entities)
            else None
        )

        # Retrieve current states from the processor
        current_states = {
            entity_id: self.processor.get_sensor_state(entity_id)
            for entity_id in self.actionable_entities
        }

        # Reward metrics
        match_count = 0
        penalty_count = 0
        missed_count = 0
        total_entities = len(current_states)

        # Evaluate the selected action
        if actionable_entity_label and actionable_entity_label in current_states:
            predicted_state = 1  # Assume discrete action sets the entity to "on"
            actual_state = current_states[actionable_entity_label]
            if predicted_state == actual_state:
                match_count += 1
            else:
                penalty_count += 1
        elif actionable_entity_label:
            self.LOGGER.debug(f"Ignoring invalid action: '{actionable_entity_label}'")

        # Count missed states
        for entity_id, actual_state in current_states.items():
            if entity_id != actionable_entity_label and actual_state != 1.0:  # Example: Missed "on" state
                missed_count += 1

        # Calculate ratios
        match_ratio = match_count / total_entities if total_entities else 0
        penalty_ratio = penalty_count / total_entities if total_entities else 0
        missed_ratio = missed_count / total_entities if total_entities else 0

        # Reward shaping
        step_reward = match_ratio - (self.penalty_weight * (penalty_ratio + missed_ratio))
        self.locals["rewards"][0] += step_reward  # Add to the environment's reward

        # Update callback metrics
        self.current_episode_reward += step_reward
        self.episode_length += 1

        # Debug output
        self.LOGGER.debug(
            f"Step {self.num_timesteps}: Matches={match_count}, Penalties={penalty_count}, Missed={missed_count}, "
            f"Match Ratio={match_ratio:.2f}, Penalty Ratio={penalty_ratio:.2f}, Missed Ratio={missed_ratio:.2f}, "
            f"Step Reward={step_reward:.2f}, Updated Reward={self.locals['rewards'][0]:.2f}"
        )

        # Handle episode end
        if done[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.LOGGER.info(
                f"Episode ended. Total Reward: {self.current_episode_reward:.2f}, Length: {self.episode_length}"
            )
            self.current_episode_reward = 0
            self.episode_length = 0

        return True
