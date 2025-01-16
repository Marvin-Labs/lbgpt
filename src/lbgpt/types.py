import sys

from litellm.types.utils import EmbeddingResponse, ModelResponse, Usage
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ConfigDict

if sys.version_info >= (3, 11):
    from typing import Optional, Self
else:
    from typing_extensions import Self


class ChatCompletionAddition(ChatCompletion):
    is_exact: bool = True
    is_cached: bool = False
    is_semantic_cached: bool = False
    model_class: Optional[str] = None

    model_config = ConfigDict(
        protected_namespaces=(), extra="ignore"  # Allow fields starting with "model_"
    )

    @classmethod
    def from_chat_completion(
        cls,
        chat_completion: ChatCompletion,
        model_class: str,
        is_exact: bool = True,
    ) -> Self:
        return cls(
            **chat_completion.model_dump(), is_exact=is_exact, model_class=model_class
        )

    @classmethod
    def from_litellm_model_response(
        cls, chat_completion: ModelResponse, model_class: str, is_exact: bool = True
    ):
        res = chat_completion.model_dump()

        # update finish reason to replace eos with finished
        choices = []
        for choice in res["choices"]:
            if choice["finish_reason"] == "eos":
                choice["finish_reason"] = "stop"
            choices.append(choice)
        res["choices"] = choices

        return cls(**res, is_exact=is_exact, model_class=model_class)


class EmbeddingAddition(BaseModel):
    embedding: list[float]
    is_cached: bool = True
    index: int


class EmbeddingResponseAddition(CreateEmbeddingResponse):
    model_class: Optional[str] = None
    object: str = 'embedding'
    data: list[EmbeddingAddition]
    usage: Usage

    model_config = ConfigDict(
        protected_namespaces=(),  # Allow fields starting with "model_"
        extra='ignore'
    )

    @classmethod
    def from_litellm_model_response(
        cls,
        embedding_response: EmbeddingResponse,
        model_class: str,
    ) -> Self:
        res = embedding_response.model_dump()
        res.pop('object', None)

        data = [
            EmbeddingAddition(embedding=data['embedding'], is_cached=False, index=data['index']) for data in res.pop('data')
        ]

        return cls(**res, model_class=model_class, data=data)


ResponseTypes = ChatCompletionAddition | EmbeddingResponseAddition
