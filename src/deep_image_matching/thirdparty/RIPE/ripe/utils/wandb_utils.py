import omegaconf


def get_flattened_wandb_cfg(conf_dict):
    flattened = {}

    def _flatten(cfg, prefix=""):
        for k, v in cfg.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, omegaconf.dictconfig.DictConfig):
                _flatten(v, new_key)
            else:
                flattened[new_key] = v

    _flatten(conf_dict)
    return flattened
