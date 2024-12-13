import os
import yaml


def get_config(mode: str) -> dict:
    config_filepath = None
    cwd = os.getcwd()
    if mode == 'exog':
        config_filepath = os.path.join(cwd, "exog.config.yaml")
    elif mode == 'data':
        config_filepath = os.path.join(cwd, "data.config.yaml")
    elif mode == 'model':
        config_filepath = os.path.join(cwd, "model.config.yaml")
    else:
        raise ValueError(f"There is no config file for mode {mode}")
        
    if os.path.exists(config_filepath):
        config_dict = None
        with open(config_filepath) as config_file:
            config_dict = yaml.safe_load(config_file)
        return config_dict
    else:
        raise Exception("Config file does not exist")
