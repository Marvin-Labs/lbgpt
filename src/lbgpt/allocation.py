import asyncio
import random
from logging import getLogger
from typing import Optional, Sequence

import tenacity

from lbgpt.base import _BaseGPT

logger = getLogger(__name__)


async def random_allocation_function(
    gpts: list[_BaseGPT], weights=Optional[Sequence[float]], **kwargs
) -> _BaseGPT:
    return random.choices(gpts, weights=weights)[0]


async def max_headroom_allocation_function(
    gpts: list[_BaseGPT], overallocate: bool = False, **kwargs
) -> _BaseGPT:
    """
    Returns the model with the most headroom. If overallocate is set to true return the model with the most headroom
    even if there is no allocation left available. Otherwise, we are waiting here until the overallocation is resolved
    """
    gpts_with_headroom = [(await gpt.headroom(), gpt) for gpt in gpts]
    best_headroom, best_alternative = max(
        gpts_with_headroom, key=lambda gpt: (gpt[0], random.random())
    )

    if overallocate:
        return best_alternative

    else:
        if best_headroom > 0:
            return best_alternative
        else:
            await asyncio.sleep(1)
            logger.info(
                "waiting for overallocation to resolve, best alternative: %s",
                best_alternative,
            )
            return await max_headroom_allocation_function(
                gpts, overallocate=overallocate, **kwargs
            )
