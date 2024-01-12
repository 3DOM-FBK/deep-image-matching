import yaml

config_file = "config.yaml"

cfg = yaml.safe_load(open(config_file, "r"))

print(cfg)


# dim_cfg = Config()
