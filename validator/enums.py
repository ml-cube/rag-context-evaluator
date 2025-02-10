from enum import Enum, EnumMeta, IntFlag
from typing import Any


class ExtendedEnum(Enum, metaclass=EnumMeta):
    """Enhanced enum class"""

    @classmethod
    def _missing_(cls, value: Any):
        for member in cls:
            if (
                isinstance(value, str) and member.name == value.upper()
            ) or member.value == value:
                return member
        return None

    @classmethod
    def names(cls) -> set[str]:
        """Return set of names"""

        def func(input_enum: ExtendedEnum) -> str:
            return input_enum.name.lower()

        return set(map(func, cls))

    @classmethod
    def str2id(cls, name: str) -> Any:
        """Transform name to value"""
        upper_name = name.upper()
        return cls[upper_name].value

    @classmethod
    def id2str(cls, value: Any) -> str:
        """Transform value to name"""
        return cls(value).name.lower()

    @classmethod
    def values_list(cls) -> list[Any]:
        """Returns list of values"""
        return [m.value for m in cls]

    @classmethod
    def values(cls) -> set[Any]:
        """Returns list of values"""

        def func(input_enum: ExtendedEnum) -> Any:
            return input_enum.value

        return set(map(func, cls))

    @classmethod
    def is_valid_name(cls, name: str) -> bool:
        """Check if the name is present in the enum"""
        return name.lower() in cls.names()

    @classmethod
    def is_valid_value(cls, value: Any) -> bool:
        """Check if the value is present in the enum"""
        return value in cls.values()

    @classmethod
    def has(cls, item: Any) -> bool:
        """Check if item is inside enum either as value or name"""
        return item in cls.values() or item in cls.names()


class TextLanguage(ExtendedEnum):
    """Text language of the text in NLP tasks

    Fields
    ------

    ITALIAN
    ENGLISH
    MULTILANGUAGE
    """

    ITALIAN = "italian"
    ENGLISH = "english"
    MULTILANGUAGE = "multilanguage"
