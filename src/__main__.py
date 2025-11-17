from src.classes.EvolutionRunner import EvolutionRunner
from src.methods.utils import load_yaml_config

if __name__ == "__main__":
    config = load_yaml_config("config.yaml")
    runner = EvolutionRunner(config)
    runner.evolve()
