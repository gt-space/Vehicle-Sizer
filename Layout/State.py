from typing import Any

class State:

    def __init__(self, value:Any = None):
        self.value = value

    @property
    def is_assigned(self) -> bool:
        return self.value is not None

    def __str__(self):
        return f"State: {self.value}"