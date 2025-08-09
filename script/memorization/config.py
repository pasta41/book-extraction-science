import os
import yaml
from dataclasses import dataclass
from typing import List

###########################################################################
#
# Decoding and suffix start configs for running probabilistic extraction
# Loaded from the CLI
#
###########################################################################

@dataclass
class ConfigItem:
    """Single config item for probabilistic extraction."""
    name: str
    temperature: float
    k: int
    suffix_start_index: int

    def __post_init__(self):
        if self.temperature <= 0:
            raise ValueError(
                f"temperature={self.temperature} (must be > 0).")
        if self.k < 1:
            raise ValueError(f"k={self.k} (must be >= 1).")
        if self.suffix_start_index < 1:
            raise ValueError(
                f"suffix_start_index={self.suffix_start_index} (must be > 0)"
            )

NAMED_CONFIGS = {
    "greedy-basic": ConfigItem("greedy-basic", 1.0, 1, 50),
    "top40-basic": ConfigItem("top40-basic", 1.0, 40, 50)
}

class ConfigListInput:
    """
    Container for a list of ConfigItem loaded from either
    a yaml file or preset names.
    """
    __tyro_excluded__ = True

    def __init__(self, input_str: str):
        if os.path.exists(input_str):
            is_yaml = input_str.endswith(".yaml") or input_str.endswith(".yml")
            if not is_yaml:
                raise ValueError(f"Expected a YAML file: {input_str}")
            with open(input_str, "r") as f:
                data = yaml.safe_load(f) or []
            self.configs: List[ConfigItem] = [ConfigItem(**item) for item in data]

        else:
            names = input_str.replace(",", " ").split()
            self.configs: List[ConfigItem] = []
            for name in names:
                if name not in NAMED_CONFIGS:
                    raise ValueError(f"Unknown config preset: '{name}'")
                self.configs.append(NAMED_CONFIGS[name])

    def __iter__(self):
        return iter(self.configs)

    def __getitem__(self, i):
        return self.configs[i]

    def __repr__(self):
        return f"ConfigListInput({self.configs})"

    def names(self) -> List[str]:
        """Return the list of config names."""
        return [c.name for c in self.configs]
