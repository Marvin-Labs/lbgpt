# -*- coding: utf-8 -*-
import hashlib
import uuid
from collections.abc import Iterable
from typing import Any

from openai.types import CompletionCreateParams, EmbeddingCreateParams
from openai.types.chat import ChatCompletionMessageParam


def non_message_parameters_from_create(
        chat_completion_create: CompletionCreateParams | dict[str, Any]
) -> dict[str, Any]:
    return dict(
        model=chat_completion_create.get(
            "model_name_cache_alias", chat_completion_create["model"]
        ),
        frequency_penalty=chat_completion_create.get("frequency_penalty", 0.0),
        logit_bias=chat_completion_create.get("logit_bias"),
        n=chat_completion_create.get("n", 1),
        presence_penalty=chat_completion_create.get("presence_penalty"),
        response_format=chat_completion_create.get("response_format"),
        seed=chat_completion_create.get("seed"),
        stop=chat_completion_create.get("stop"),
        stream=chat_completion_create.get("stream", False),
        temperature=chat_completion_create.get("temperature", 1.0),
        top_p=chat_completion_create.get("top_p", 1.0),
        tools=chat_completion_create.get("tools"),
        tool_choice=chat_completion_create.get("tool_choice"),
        function_call=chat_completion_create.get("function_call"),
        functions=chat_completion_create.get("functions"),
    )


def convert_message(message: ChatCompletionMessageParam):
    role = message["role"]

    # content needs to be a string in the end
    raw_content = message["content"]

    if isinstance(raw_content, str):
        content = raw_content
    elif raw_content is None:
        content = ""
    elif isinstance(raw_content, Iterable):
        content = ""
        for c in raw_content:
            if isinstance(c, dict):
                if "text" in c:
                    content += c["text"] + "\n\n"
                elif "image_url" in c:
                    image_url = c["image_url"]
                    url = image_url["url"]
                    detail = image_url.get("detail")

                    content += f"![image]({url}) {detail}\n\n"

    else:
        content = str(raw_content)

    if content == "":
        # if the content fails, we want to create a random key in order to avoid caching
        content = str(uuid.uuid4())
    return {
        "content": content.strip(),
        "role": role,
    }


def make_hash_chatgpt_request(
        chat_completion_create: CompletionCreateParams,
        include_messages: bool = True,
) -> str:
    """Converting a chatgpt request to a hash for caching and deduplication purposes"""

    non_message_parameters = non_message_parameters_from_create(
        chat_completion_create=chat_completion_create
    )

    messages = [
        convert_message(message) for message in chat_completion_create["messages"]
    ]

    hasher = hashlib.sha256()
    hasher.update(repr(make_hashable(non_message_parameters)).encode())
    if include_messages:
        hasher.update(repr(make_hashable(messages)).encode())

    return f"lbgpt_{hasher.digest().hex()}"


def _make_hash_single_embedding(
        massaged_embedding_create: dict,
        input_text: str
) -> str:
    hasher = hashlib.sha256()

    hasher.update(repr(make_hashable(massaged_embedding_create)).encode())
    hasher.update(input_text.encode())

    return f"lbgpte_{hasher.digest().hex()}"


def make_hash_embedding_request(request: EmbeddingCreateParams) -> list[str]:
    """Converting an embedding request to a hash for caching and deduplication purposes"""
    massaged_request = dict(request)
    input_texts = massaged_request.pop("input")

    return [
        _make_hash_single_embedding(massaged_request, input_text=input_text)
        for input_text in input_texts
    ]



def make_hashable(o):
    if isinstance(o, (tuple, list)):
        return tuple((make_hashable(e) for e in o))

    if isinstance(o, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in o.items()))

    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))

    return o
