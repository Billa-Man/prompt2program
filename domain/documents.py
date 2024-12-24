from .base import NoSQLBaseDocument
from .types import DataCategory


class Document(NoSQLBaseDocument):
    content: dict
    platform: str

class SiteDocument(Document):
    link: str

    class Settings:
        name = DataCategory.SITES