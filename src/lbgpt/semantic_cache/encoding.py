# Converting the prompt into something that will then be embedded with an embedding model
from pydantic import BaseModel


class RoleMessage(BaseModel):
    role: str
    content: str

    def to_text(self) -> str:
        return f"[{self.role}]: {self.content.strip()}"


def user_only(role_messages: list[RoleMessage]) -> str:
    return "\n\n".join(
        [message.to_text() for message in role_messages if message.role == "user"]
    )


def all_(role_messages: list[RoleMessage]) -> str:
    return "\n\n".join([message.to_text() for message in role_messages])


def system_only(role_messages: list[RoleMessage]) -> str:
    return "".join(
        [message.content for message in role_messages if message.role == "system"]
    )
