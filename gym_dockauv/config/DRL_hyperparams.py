# For reference from PPO description:
PPO_HYPER_PARAMS_DEFAULT = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "target_kl": None,
    "tensorboard_log": None,
    "create_eval_env": False,
    "policy_kwargs": None,
    "verbose": 0,
    "seed": None,
    "device": "auto",
    "_init_setup_model": True
}

PPO_HYPER_PARAMS_TEST = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 8192,
    "n_epochs": 10,
    "gamma": 0.97,
    "gae_lambda": 0.90,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "target_kl": None,
    "tensorboard_log": "tb_logs",  # Changed to save tensorboard logs
    "create_eval_env": False,
    "policy_kwargs": None,
    "verbose": 0,
    "seed": None,
    "device": "auto",
    "_init_setup_model": True
}

