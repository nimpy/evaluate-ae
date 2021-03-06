import wandb
import os
from pathlib import Path

import utilities
from unite_sweeps_into_one import sweep_one_sweep_to_rule_them_all

if __name__ == '__main__':
    sweep_config = {
        'method': 'grid',
        'project': 'evaluate-ae',
        'metric': {
            'goal': 'minimize',
            'name': 'hpatches_overall'
        },
        'parameters': {
            'data_augm_level': {
                'values': [0, 1, 2, 3]
            },
            'activation_fn': {
                'values': ['elu', 'relu']
            },
            'loss_fn': {
                'values': ['bce', 'msssim']
            },
            'vae_beta_norm': {
                'values': [0.01, 0.001, 0.0001, 0.00001, 0]  # 0 means it's AE
            },
            'learning_rate': {
                'values': [0.0001]
            }
        }
    }

    sweep_version = 'sweep__one_sweep_to_rule_them_all_v2022_3'  # TODO change in both files!!! (TODO make it a parameter)
    sweep_dir = os.path.join('/scratch/image_datasets/3_65x65/ready/weights', sweep_version)
    Path(sweep_dir).mkdir(parents=True, exist_ok=True)

    # Set the logger
    utilities.set_logger(os.path.join(sweep_dir, 'train.log'))

    sweep_id = wandb.sweep(sweep_config, project="evaluate-ae")  # TODO!!!
    wandb.agent(sweep_id, function=sweep_one_sweep_to_rule_them_all)