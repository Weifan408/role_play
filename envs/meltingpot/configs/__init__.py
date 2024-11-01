from collections.abc import Mapping, Sequence

import importlib
from typing import Any

from ml_collections import config_dict


def _validated(build):
    """And adds validation checks to build function."""

    def lab2d_settings_builder(
        *,
        config: config_dict.ConfigDict,
        roles: Sequence[str],
    ) -> Mapping[str, Any]:
        invalid_roles = set(roles) - config.valid_roles
        if invalid_roles:
            raise ValueError(
                f"Invalid roles: {invalid_roles!r}. Must be one of "
                f"{config.valid_roles!r}"
            )
        return build(config=config, roles=roles)

    return lab2d_settings_builder


def get_config(substrate: str):
    path = f"{__name__}.{substrate}"
    module = importlib.import_module(path)
    config = module.get_config()
    with config.unlocked():
        config.lab2d_settings_builder = _validated(module.build)
    return config.lock()
