import json
import os
import warnings


def parse_json_config(args, config=None, first=False):
    if config is None:
        config = args.config

    with open(config, 'r') as f:
        output = json.load(f)

    def fix_base_cfg_path(base_config_path, is_relative):
        if is_relative:
            return os.path.join(os.path.dirname(config), base_config_path)
        return base_config_path

    # Option to override base config path for the first thing.
    # Warning: must follow the same relative rule of the config.
    if first and args.base_config_path:
        output["base_config_path"] = args.base_config_path

    # option to load a base config, reducing code duplication.
    if "base_config_path" in output:
        base_config_path = output.get("base_config_path")
        is_relative = output.get("base_config_path_is_relative", True)
        if isinstance(base_config_path, list):
            for i in base_config_path:
                parse_json_config(args,
                                  config=fix_base_cfg_path(i, is_relative))
        else:
            parse_json_config(args,
                              config=fix_base_cfg_path(base_config_path,
                                                       is_relative))
        # NOTE: it can be changed by children:
        if args.base_config_path != base_config_path:
            warnings.warn("Config path changed by child")

    if not os.path.exists(config):
        raise ValueError(f"Config {config} does not exists")

    add_parsed_config_to_args(args, output)


def add_parsed_config_to_args(args, output:dict):
    for key, value in output.items():

        # Allow skipping some options and loading them from cmd.
        # Example: seed_from_cmd
        if output.get(f'{key}_from_cmd', False) or getattr(
                args, f'{key}_from_cmd', False):
            if not hasattr(args, key):
                raise RuntimeError(f"-W- {key}_from_cmd=True but not set")
            continue

        key: str
        if key.endswith("_from_cmd") and hasattr(args, key) and getattr(args, key):
            warnings.warn(f"Taking {key} from cmd_args instead of overriding from json")
            continue
        # Replace
        setattr(args, key, value)
