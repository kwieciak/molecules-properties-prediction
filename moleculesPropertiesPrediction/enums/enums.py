from enum import Enum

class Normalization(str, Enum):
    STANDARD = "standard",
    MINMAX = "minmax"