import os

import pytest

from lbgpt import ChatGPT


@pytest.mark.vcr
async def test_image_url():
    requests = [
        {'model': 'gpt-4o',
         'messages': [
             {'role': 'system', 'content': 'You are a helpful assistant.'},
             {'role': 'user',
              'content': [
                  {
                      'type': 'text', 'text': 'What’s in this image?'
                  },
                  {
                      'type': 'image_url',
                      'image_url': {
                          'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg'}
                  }
              ]
              }
         ]
         }
    ]
    lb = ChatGPT(
        api_key=os.environ["OPEN_AI_API_KEY"],
        stop_after_attempts=1,
        stop_on_exception=True,
        auto_cache=True,
    )

    res = lb.chat_completion_list(requests)
    for r in res:
        assert not(isinstance(r, Exception))

