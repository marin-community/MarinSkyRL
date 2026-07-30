"""Root `__init__` of the gym module setting the `__all__` of skyrl modules."""

from skyrl_gym.core import Env
from skyrl_gym import error

from skyrl_gym.envs.registration import (
    make,
    spec,
    register,
    registry,
    pprint_registry,
)
from skyrl_gym import tools, envs
from skyrl_gym.envs.data_contracts import get_data_contract

# Define __all__ to control what's exposed
__all__ = [
    # core classes
    "Env",
    # registration
    "make",
    "spec",
    "register",
    "registry",
    "pprint_registry",
    "get_data_contract",
    # module folders
    "envs",
    "tools",
    "error",
]
__version__ = "0.0.0"
