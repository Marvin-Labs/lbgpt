import abc
import logging
from typing import Any, Callable, Iterable, Optional

from langchain_core.embeddings import Embeddings
from openai.types.chat import ChatCompletion, CompletionCreateParams
from openai.types.completion_create_params import (
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
)

from lbgpt.cache import non_message_parameters_from_create
from lbgpt.semantic_cache import encoding
from lbgpt.types import ChatCompletionAddition

logger = logging.getLogger(__name__)


def get_completion_create_params(**query) -> dict[str, Any]:
    if query.get("stream", False) is True:
        return CompletionCreateParamsStreaming(**query)
    else:
        return CompletionCreateParamsNonStreaming(**query)


ENCODING_METHODS: dict[str, Callable[[list[encoding.RoleMessage]], str]] = {
    "user_only": encoding.user_only,
    "all": encoding.all_,
    "system_only": encoding.system_only,
}


class _SemanticCacheBase(abc.ABC):
    def __init__(
        self,
        embedding_model: Embeddings,
        cosine_similarity_threshold: float = 0.99,
    ):
        self.embeddings_model = embedding_model
        self.cosine_similarity_threshold = cosine_similarity_threshold

    def encode_messages(
        self, messages: list[dict[str, Any]], encoding_method: str
    ) -> str:
        if encoding_method not in ENCODING_METHODS:
            raise ValueError(f"Unknown encoding method: {encoding_method}")

        em = ENCODING_METHODS[encoding_method]
        rm = [
            encoding.RoleMessage(role=m["role"], content=m["content"].strip())
            for m in messages
        ]

        return em(rm)

    def embed_messages(
        self, messages: list[dict[str, Any]], encoding_method: str
    ) -> list[float]:
        txt = self.encode_messages(messages, encoding_method)
        return self.embeddings_model.embed_documents([txt])[0]

    async def aembed_messages(
        self, messages: list[dict[str, Any]], encoding_method: str
    ) -> list[float]:
        txt = self.encode_messages(messages, encoding_method)
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
        self,
        query: CompletionCreateParams | dict[str, Any],
        semantic_cache_encoding_method: Optional[str],
    ) -> Optional[ChatCompletionAddition]:
        return None

    @abc.abstractmethod
    async def add_cache(
        self,
        query: CompletionCreateParams | dict[str, Any],
        response: ChatCompletion,
        semantic_cache_encoding_method: Optional[str],
    ) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def count(self) -> int:
        raise NotImplementedError()
