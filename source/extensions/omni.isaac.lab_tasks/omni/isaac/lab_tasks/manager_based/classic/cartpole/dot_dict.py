from typing import Any

class DotDict(dict):
    """
    dot.notation access to dictionary attributes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __setitem__(self, key: Any, item: Any) -> None:
        if isinstance(item, dict):
            item = DotDict(item)
        super().__setitem__(key, item)

    def __getitem__(self, key: Any) -> Any:
        return super().__getitem__(key)

    def __setattr__(self, key: Any, item: Any) -> None:
        return self.__setitem__(key, item)

    def __getattr__(self, key: Any) -> Any:
        return self.__getitem__(key)

    def __missing__(self, key: Any) -> str:
        raise KeyError(f"DotDict object has no item '{key}'")

    def __getstate__(self) -> dict:
        return self

    def __setstate__(self, state: dict) -> None:
        self.update(state)
        self.__dict__ = self

    def __delattr__(self, item: Any) -> None:
        self.__delitem__(item)

    def __delitem__(self, key: Any) -> None:
        super().__delitem__(key)