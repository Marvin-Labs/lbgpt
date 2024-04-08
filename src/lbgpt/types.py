import sys

from openai.types.chat import ChatCompletion

if sys.version_info >= (3, 11):
    from typing import Self, Optional
else:
    from typing_extensions import Self


class ChatCompletionAddition(ChatCompletion):
    is_exact: bool = True
    is_cached: bool = False
    is_semantic_cached: bool = False
    model_class: Optional[str] = None

    @classmethod
    def from_chat_completion(
            cls, chat_completion: ChatCompletion, model_class: str, is_exact: bool = True,
    ) -> Self:
        return cls(**chat_completion.model_dump(), is_exact=is_exact)
