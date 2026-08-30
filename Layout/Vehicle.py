from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .Network import Network

class Vehicle:

    def __init__(self, network: Network):
        self.network = network