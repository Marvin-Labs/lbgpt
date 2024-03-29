import abc
import logging
from typing import Any, Iterable, Optional

from langchain_core.embeddings import Embeddings
from openai.types.chat import ChatCompletion, CompletionCreateParams
from openai.types.completion_create_params import (
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
)

from lbgpt.cache import non_message_parameters_from_create
from lbgpt.types import ChatCompletionAddition

logger = logging.getLogger(__name__)


def get_completion_create_params(**query) -> dict[str, Any]:
    if query.get("stream", False) is True:
        return CompletionCreateParamsStreaming(**query)
    else:
        return CompletionCreateParamsNonStreaming(**query)


class _SemanticCacheBase(abc.ABC):
    def __init__(
        self,
        embedding_model: Embeddings,
        cosine_similarity_threshold: float = 0.99,
        converted_message_roles: Iterable[str] = ("user",),
    ):
        self.embeddings_model = embedding_model
        self.cosine_similarity_threshold = cosine_similarity_threshold
        self.converted_message_roles = converted_message_roles

    def message_to_text(self, message: [str, Any]) -> str:
        return f'[{message["role"]}]: {message["content"].strip()}\n\n'

    def messages_to_text(self, messages: list[dict[str, Any]]) -> str:
        txt = ""
        for message in messages:
            if message["role"] in self.converted_message_roles:
                txt += self.message_to_text(message)

        return txt

    def embed_messages(self, messages: list[dict[str, Any]]) -> list[float]:
        txt = self.messages_to_text(messages)
        return self.embeddings_model.embed_documents([txt])[0]

    async def aembed_messages(self, messages: list[dict[str, Any]]) -> list[float]:
        txt = self.messages_to_text(messages)
        return await self.embeddings_model.aembed_query(txt)


    def non_message_dict(
        self,
        chat_completion_create: CompletionCreateParams | dict[str, Any],
        allowed_types: Optional[Iterable] = None,
        convert_not_allowed_to_empty: bool = True,
    ) -> dict[str, Any]:
        res = non_message_parameters_from_create(
            chat_completion_create=chat_completion_create
        )

        if allowed_types is not None:
            if convert_not_allowed_to_empty:
                res = {
                    k: v if isinstance(v, tuple(allowed_types)) else ""
                    for k, v in res.items()
                }
            else:
                res = {
                    k: v for k, v in res.items() if isinstance(v, tuple(allowed_types))
                }

        return res

    @abc.abstractmethod
    async def query_cache(
        self, query: CompletionCreateParams | dict[str, Any]
    ) -> Optional[ChatCompletionAddition]:
        return None

    @abc.abstractmethod
    async def add_cache(
        self, query: CompletionCreateParams | dict[str, Any], response: ChatCompletion
    ) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def count(self) -> int:
        raise NotImplementedError()
