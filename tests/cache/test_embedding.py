from os import environ

import numpy as np
import pytest
from cachetools import Cache
from sklearn.metrics.pairwise import cosine_similarity

from lbgpt.lbgpt import LiteLlmRouter


@pytest.fixture
def cache() -> Cache:
    return Cache(maxsize=1_000)


@pytest.mark.vcr(filter_query_parameters=["key"])
async def test_embedding_cache(cache: Cache):
    model_name = 'openai/text-embedding-ada-002'
    lb = LiteLlmRouter(model_list=[{
        'model_name': model_name,
        'litellm_params': {
            'model': model_name,
            'max_retries': 0,
            'api_key': environ['OPEN_AI_API_KEY']
        }

    }], cache=cache)

    input_cached = 'hello world'

    # running the non-caching variation
    res_empty_cache = await lb.cached_embedding(model=model_name, input=[input_cached])
    assert res_empty_cache.data[0].is_cached is False

    res_cache = await lb.cached_embedding(model=model_name, input=[input_cached])
    assert res_cache.data[0].is_cached is True

    res_partial_cache = await lb.cached_embedding(model=model_name,
                                                  input=[input_cached, 'test 1', 'test 2', input_cached, 'test 3'])
    assert res_partial_cache.data[0].is_cached is True
    assert res_partial_cache.data[1].is_cached is False
    assert res_partial_cache.data[2].is_cached is False
    assert res_partial_cache.data[3].is_cached is True
    assert res_partial_cache.data[4].is_cached is False

    res_baseline = await lb.embedding(model=model_name,
                                      input=[input_cached, 'test 1', 'test 2', input_cached, 'test 3'])

    # checking if we have the correct embeddings
    for cached, uncached in zip(res_partial_cache.data, res_baseline.data):
        np.isclose(cosine_similarity(np.array([cached.embedding]), np.array([uncached.embedding]))[0][0], 1.0)
