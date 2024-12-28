# Home Assistant Reinforcement Learning (Home Assistant RL)

This project applies reinforcement learning (RL) techniques to optimize and automate behaviors in a Home Assistant environment. Using logs and user interactions as a training dataset, we aim to create adaptive and intelligent automations for smart homes.

---

## **Folder Structure**
The project is designed to be modular and robust, following Python best practices and ensuring future extensibility.

```plaintext
home_assistant_rl/
├── data/                        # Contains exported Home Assistant logs and other datasets
│   ├── raw_logs/                # Raw logs directly exported from HA
│   ├── processed_logs/          # Processed logs ready for training
│   └── config.yaml              # Configuration file for data preprocessing (e.g., features to extract)
├── src/                         # Core source code for the RL project
│   ├── __init__.py              # Marks src as a package
│   ├── main.py                  # Entry point of the project
│   ├── environments/            # Custom Gymnasium environments
│   │   ├── __init__.py
│   │   └── home_assistant_env.py # Custom hybrid Gym environment
│   ├── models/                  # RL models and training scripts
│   │   ├── __init__.py
│   │   ├── ppo.py               # PPO-specific training logic
│   │   └── utils.py             # Helper functions for model handling
│   ├── utils/                   # Utility functions and shared logic
│   │   ├── __init__.py
│   │   ├── data_processing.py   # Functions for processing HA logs
│   │   ├── logger.py            # Logging setup using IceCream (IC)
│   │   └── config.py            # Global configuration and constants
├── tests/                       # Unit and integration tests
│   ├── __init__.py
│   ├── test_environment.py      # Tests for the custom environment
│   ├── test_model.py            # Tests for PPO training logic
│   └── test_utils.py            # Tests for utility functions
├── notebooks/                   # Jupyter notebooks for exploratory data analysis and visualization
│   └── data_analysis.ipynb      # Analyze HA logs and preprocess features
├── docker/                      # Docker setup for reproducibility
│   ├── Dockerfile               # Dockerfile for creating the container
│   ├── docker-compose.yaml      # Docker Compose for managing services
├── docs/                        # Documentation for the project
│   ├── setup.md                 # Guide to setting up the project
│   ├── usage.md                 # Usage instructions
│   └── api_reference.md         # API documentation
├── .env                         # Environment variables (not committed to version control)
├── .gitignore                   # Git ignore file
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project metadata and dependency management
└── README.md                    # Project overview and quick start guide
```

---

## **Key Components**

### `data/`
- Organizes raw and processed logs.
- Includes a `config.yaml` for customizable preprocessing options.

### `src/`
- Centralized source code with subdirectories for environments, models, and utilities.
- **Environments**: Houses the custom Gymnasium environment.
- **Models**: Contains training scripts and PPO-specific logic.
- **Utils**: Includes shared utilities like data processing and logging.

### `tests/`
- Ensures the project is tested rigorously.

### `notebooks/`
- For experimenting with data and visualizations.

### `docker/`
- Facilitates reproducibility and easy environment setup.

### Top-Level Files
- `.env`: Stores sensitive configurations.
- `requirements.txt`: Lists Python dependencies.
- `pyproject.toml`: Manages project metadata and dependencies (useful with `poetry`).

---

## **Technologies Used**
- **Python 3.10+**: Core programming language.
- **Stable-Baselines3**: RL library for implementing PPO.
- **Gymnasium**: Modernized Gym framework for creating custom RL environments.
- **IceCream (IC)**: For enhanced debugging and logging.
- **Docker**: Ensures reproducibility across setups.

---

## **Next Steps**
1. **Data Preparation**:
   - Export Home Assistant logs to `data/raw_logs/`.
   - Process them using utilities in `src/utils/data_processing.py`.

2. **Environment Setup**:
   - Implement the custom Gymnasium environment in `src/environments/home_assistant_env.py`.

3. **Model Training**:
   - Use the PPO implementation in `src/models/ppo.py` to train on the processed data.

4. **Testing and Debugging**:
   - Write unit tests for each component in the `tests/` directory.
   - Debug using the IceCream logging setup in `src/utils/logger.py`.

5. **Deployment**:
   - Package the application using Docker.
   - Integrate with Home Assistant via APIs for live testing.

---

## **Contributing**
Contributions are welcome! Please follow Python PEP-8 standards and write tests for any new functionality. Check `docs/setup.md` for developer setup instructions.

---

## **License**
This project is licensed under the MIT License. See `LICENSE` for details.

