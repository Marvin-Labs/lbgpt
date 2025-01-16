from typing import Any

from openai._compat import model_dump


def convert_to_dictionary(value: Any) -> dict:
    if isinstance(value, dict):
        return value

    try:
        return model_dump(value)
    except AttributeError:
        raise NotImplementedError(f"Cannot convert {value} to dictionary")

