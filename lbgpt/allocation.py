import asyncio
import random
from typing import Sequence, Optional
from lbgpt.base import _BaseGPT


async def random_allocation_function(gpts: list[_BaseGPT], weights=Optional[Sequence[float]], **kwargs) -> _BaseGPT:
    return random.choices(gpts, weights=weights)[0]


async def max_headroom_allocation_function(gpts: list[_BaseGPT], overallocate: bool = False, **kwargs) -> _BaseGPT:
    """
    Returns the model with the most headroom. If overallocate is set to true return the model with the most headroom
    even if there is no allocation left available. Otherwise, we are waiting here until the overallocation is resolved
    """

    best_alternative = max(gpts, key=lambda gpt: gpt.headroom())

    if overallocate:
        return best_alternative

    else:
        if best_alternative.headroom() > 0:
            return best_alternative
        else:
            await asyncio.sleep(1)
            return await max_headroom_allocation_function(gpts, overallocate=overallocate, **kwargs)

