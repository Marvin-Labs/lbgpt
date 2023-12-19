import abc
from typing import Any, Iterable, Optional

from langchain_core.embeddings import Embeddings
import logging

from openai.types.chat import CompletionCreateParams, ChatCompletion

from lbgpt.cache import non_message_parameters_from_create
from lbgpt.types import ChatCompletionAddition

logger = logging.getLogger(__name__)


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
        txt = self.message_to_text(messages)
        return self.embeddings_model.embed_documents([txt])[0]

    def non_message_dict(
        self, chat_completion_create: CompletionCreateParams | dict[str, Any]
    ) -> dict[str, Any]:
        return non_message_parameters_from_create(
            chat_completion_create=chat_completion_create
        )

    @abc.abstractmethod
    def query_cache(self, query: CompletionCreateParams | dict[str, Any]) -> Optional[ChatCompletionAddition]:
        return None

    @abc.abstractmethod
    def add_cache(
            self, query: CompletionCreateParams | dict[str, Any], response: ChatCompletion
    ) -> None:
        raise NotImplementedError()


