import os

import pytest

from lbgpt import ChatGPT


@pytest.mark.vcr
async def test_preserve_order():
    requests = [
        {
            "model": "o1-preview",
            "messages": [
                {"role": "user", "content": "please just return the word cat"}
            ],
        },
        {
            "model": "thisisabadmodelname",
            "messages": [
                {"role": "system", "content": "return error"},
                {"role": "user", "content": "please just return the word cat"},
            ],
        },
        {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "please just return the word dog"}
            ],
        },
    ]
    lb = ChatGPT(
        api_key=os.environ["OPEN_AI_API_KEY"],
        stop_after_attempts=1,
        stop_on_exception=True,
        auto_cache=True,
    )

    res = lb.chat_completion_list(requests)
    assert res[0].choices[0].message.content == "cat"
    assert isinstance(res[1], Exception)
    assert res[2].choices[0].message.content == "dog"
