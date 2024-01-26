# -*- coding: utf-8 -*-
import os
from typing import Any, Dict

import pytest
import vcr  # type: ignore[import]
import vcr.request  # type: ignore[import]
import vcr.util  # type: ignore[import]

vcr_match_on = ("method", "scheme", "host", "port", "path", "query", "body")


@pytest.fixture()
def vcr_cassette_dir(request):  # type: ignore[no-untyped-def]
    return os.path.join(
        os.path.dirname(request.module.__file__), "cassette", request.module.__name__
    )


@pytest.fixture()
def vcr_config() -> Dict[str, Any]:
    return {
        "record_mode": "once",  # new_episodes
        "match_on": vcr_match_on,
        "filter_headers": {"authorization": "DUMMY_AUTHORIZATION"},
        "filter_query_parameters": ["token"],
        "ignore_hosts": ["localhost", "testserver"],
    }
