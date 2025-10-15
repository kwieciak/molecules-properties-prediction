from enum import Enum


class Normalization(str, Enum):
    STANDARD = "standard",
    MINMAX = "minmax"


class TaskType(str, Enum):
    BINARY = "binary",
    MULTICLASS = "multiclass",
    REGRESSION = "regression"
