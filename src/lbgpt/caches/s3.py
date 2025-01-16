from logging import getLogger
from typing import Optional

from botocore.exceptions import ClientError
from mypy_boto3_s3 import S3Client
from mypy_boto3_s3.type_defs import BlobTypeDef

logger = getLogger(__name__)


class S3Cache:
    def __init__(self, s3_client: S3Client, bucket: str, prefix: str):
        self.s3_client = s3_client
        self.bucket = bucket
        self.prefix = prefix

    def _make_key(self, key) -> str:
        return f"{self.prefix}{key}"

    async def aget(self, key) -> Optional[str]:
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket, Key=self._make_key(key)
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.debug("Key not found in cache")
                return None
            raise e

        return response["Body"].read().decode("utf-8")

    async def aset(self, key, value: BlobTypeDef) -> None:
        self.s3_client.put_object(
            Bucket=self.bucket, Key=self._make_key(key), Body=value
        )

    @property
    def count(self):
        return
