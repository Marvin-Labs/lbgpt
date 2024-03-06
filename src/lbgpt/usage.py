import dataclasses
import datetime
import sys

if sys.version_info >= (3, 11):
    from typing import Optional, Self
else:
    from typing_extensions import Self


@dataclasses.dataclass
class Usage:
    input_tokens: int
    output_tokens: int

    start_datetime: datetime.datetime
    end_datetime: datetime.datetime

    attempt: Optional[int] = None


@dataclasses.dataclass
class UsageStats:
    tokens: int
    requests: int
