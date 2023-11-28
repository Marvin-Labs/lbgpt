import random
from typing import Sequence, Optional
from lbgpt.base import _BaseGPT


def random_allocation_function(gpts: list[_BaseGPT], weights=Optional[Sequence[float]], **kwargs) -> _BaseGPT:
    return random.choices(gpts, weights=weights)[0]
