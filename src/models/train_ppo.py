# path: src/models/train_ppo.py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from src.environments.home_assistant_env import HomeAssistantEnv
from src.models.utils import AutoSave, linear_schedule, RewardsCallback
from stable_baselines3.common.callbacks import CallbackList
from src.utils import get_logger
from src.environments.ha_simulator import HomeAssistantProcessor
import yaml
import torch as th

# Initialize logger for this module
LOGGER = get_logger("PPOTrainer")


def make_env(processor, rank):
    """
    Factory function for creating HomeAssistantEnv instances.
    :param processor: Shared HomeAssistantProcessor instance.
    :param rank: Rank of the environment (used for seeding, if needed).
    """
    def _init():
        LOGGER.info(f"Creating environment instance for rank: {rank}")
        observation_space_data = processor.get_observation_space()
        action_space_data = processor.get_action_space()

        if observation_space_data.empty:
            LOGGER.error(f"Observation space is empty for rank {rank}. Cannot create environment.")
            raise ValueError(f"Observation space is empty for rank {rank}.")
        if action_space_data.empty:
            LOGGER.error(f"Action space is empty for rank {rank}. Cannot create environment.")
            raise ValueError(f"Action space is empty for rank {rank}.")

        LOGGER.debug(f"Environment {rank} created with {len(observation_space_data)} observations and {len(action_space_data)} actions.")
        return HomeAssistantEnv(observation_space_data, action_space_data)
    return _init


def parse_automations(automation_file):
    """
    Parse automations from a YAML file into a usable format.
    :param automation_file: Path to the YAML file containing automations.
    :return: List of parsed automations.
    """
    try:
        with open(automation_file, "r") as file:
            automations = yaml.safe_load(file)
        LOGGER.info(f"Successfully parsed {len(automations)} automations.")
        return [
            {"triggers": automation.get("trigger", []), "actions": automation.get("action", [])}
            for automation in automations
        ]
    except Exception as e:
        LOGGER.error(f"Failed to parse automations from {automation_file}: {e}")
        raise


if __name__ == "__main__":
    # Define paths and configuration settings
    log_file = "data/processed_logs/processed_data.csv"
    action_mapping_file = "data/processed_logs/action_space_mapping.csv"
    automation_file = "data/automations.yaml"
    log_dir = "logs/tensorboard"
    save_dir = "models"
    save_freq = 1382400  # Frequency of saving the model (in steps)
    num_envs = 2  # Number of parallel environments
    total_timesteps = 31_536_000  # Total timesteps for training
    n_epochs = 4  # Number of epochs for PPO updates
    n_steps = 1024  # Steps per environment before policy update

    LOGGER.info("Starting training script...")
    LOGGER.debug(f"Log file: {log_file}")
    LOGGER.debug(f"Configuration: num_envs={num_envs}, total_timesteps={total_timesteps}, n_steps={n_steps}, n_epochs={n_epochs}")

    # Initialize shared processor with action mapping
    processor = HomeAssistantProcessor(
        log_file=log_file,
        action_mapping_file=action_mapping_file
    )
    LOGGER.info("Shared HomeAssistantProcessor instance created.")

    # Parse automations
    try:
        automations = parse_automations(automation_file)
    except Exception as e:
        LOGGER.error("Error parsing automations.", exc_info=True)
        raise

    # Create a vectorized environment with SubprocVecEnv
    try:
        LOGGER.info("Creating SubprocVecEnv for training...")
        env = SubprocVecEnv([make_env(processor, i) for i in range(num_envs)])
        LOGGER.info("Vectorized environment created.")
    except Exception as e:
        LOGGER.error("Error during environment creation.", exc_info=True)
        raise

    # Initialize PPO model with custom hyperparameters and network architecture
    try:
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),  # Separate actor and critic networks
            activation_fn=th.nn.ReLU,  # Use ReLU activation
        )

        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=linear_schedule(0.0003, 0.00005),  # Linear decay for learning rate
            clip_range=linear_schedule(0.2, 0.03),  # Linear decay for clipping range
            n_steps=n_steps,  # Steps per environment per policy update
            batch_size=(num_envs * n_steps) // n_epochs,  # Minibatch size for optimization
            n_epochs=n_epochs,  # Number of optimization epochs per update
            gamma=0.2,  # Discount factor (focus on immediate rewards)
            gae_lambda=0.1,  # GAE parameter (step-wise optimization)
            max_grad_norm=0.5,  # Maximum gradient norm for clipping
            ent_coef=0.02,  # Entropy coefficient to encourage exploration
            device="cpu",  # Use CPU for training
            tensorboard_log=log_dir,  # TensorBoard logging directory
            policy_kwargs=policy_kwargs,  # Custom network architecture
        )
        LOGGER.info("PPO model initialized.")
    except Exception as e:
        LOGGER.error("Error initializing PPO model.", exc_info=True)
        raise

    # Setup callbacks
    try:
        autosave_callback = AutoSave(
            check_freq=save_freq * num_envs,  # Adjust for multiple environments
            num_envs=num_envs,
            save_path=save_dir,
            filename_prefix="ppo_",
            verbose=1
        )
        LOGGER.info(f"AutoSave callback configured: {autosave_callback}")

        rewards_callback = RewardsCallback(
            processor=processor,
            automations=automations,
            actionable_entities=processor.actionable_entities,
            penalty_weight=0.5,
            verbose=1 
        )
        LOGGER.info("RewardsCallback configured.")

        callback_list = CallbackList([autosave_callback, rewards_callback])
        LOGGER.info("Callback list created.")
    except Exception as e:
        LOGGER.error("Error configuring callbacks.", exc_info=True)
        raise

    # Train the PPO model
    try:
        LOGGER.info("Starting model training...")
        model.learn(total_timesteps=total_timesteps, callback=callback_list)
        LOGGER.info("Model training complete.")
    except Exception as e:
        LOGGER.error("Error during training.", exc_info=True)
        raise

    # Save the final trained model
    try:
        model.save(f"{save_dir}/ppo_final")
        LOGGER.info(f"Final model saved at: {save_dir}/ppo_final")
    except Exception as e:
        LOGGER.error("Error saving final model.", exc_info=True)
        raise
