import torch
import yaml

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config




# Load config in initial setup (this will be global)
def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config





#this config will be used throughout
config= load_config(config_path="config.yaml")
device = torch.device('cpu')
torch.set_num_threads(16)  # Use all 16 cores
torch.set_num_interop_threads(8)
print(f"Configured device: {device}")