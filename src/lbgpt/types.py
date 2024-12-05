import sys

from litellm.types.utils import ModelResponse
from openai.types.chat import ChatCompletion
from pydantic import ConfigDict

if sys.version_info >= (3, 11):
    from typing import Self, Optional
else:
    from typing_extensions import Self


class ChatCompletionAddition(ChatCompletion):
    is_exact: bool = True
    is_cached: bool = False
    is_semantic_cached: bool = False
    model_class: Optional[str] = None

    model_config = ConfigDict(
        protected_namespaces=(),  # Allow fields starting with "model_"
        extra='ignore'
    )

    @classmethod
    def from_chat_completion(
            cls, chat_completion: ChatCompletion, model_class: str, is_exact: bool = True,
    ) -> Self:
        return cls(**chat_completion.model_dump(), is_exact=is_exact, model_class=model_class)

    @classmethod
    def from_litellm_model_response(cls, chat_completion: ModelResponse, model_class: str, is_exact: bool = True):
        return cls(**chat_completion.model_dump(), is_exact=is_exact, model_class=model_class)