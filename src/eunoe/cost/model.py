"""Usage times a dated price file. A run records which file it used, so a number can be recomputed.

Prices move, so nothing here is hardcoded outside the files in `prices/`, each copied by hand from the
provider's own page on the date in its name, per million tokens.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from eunoe.interfaces import CallUsage

PRICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prices")
DEFAULT_FILE = "2026-11-06-anthropic.json"
DEFAULT_MODEL = "claude-sonnet-5-5"


def list_price_files() -> list:
    return sorted(f for f in os.listdir(PRICE_DIR) if f.endswith(".json"))


def load_prices(name: str = DEFAULT_FILE) -> dict:
    with open(os.path.join(PRICE_DIR, name), encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class ListPrice:
    """Implements `CostModel`: a list-price yardstick, whatever door a call really went through."""

    model: str = DEFAULT_MODEL
    price_file: str = DEFAULT_FILE

    def __post_init__(self) -> None:
        data = load_prices(self.price_file)
        if self.model not in data["models"]:
            raise ValueError(f"{self.model!r} is not in {self.price_file}")
        self.rates = data["models"][self.model]
        self.long_prompt = self.rates.get("over_prompt_tokens")

    def price(self, usage: CallUsage, ttl: str = "5m") -> float:
        rates = self.rates
        if self.long_prompt and usage.context > self.long_prompt:
            rates = {**rates, **rates["over"]}
        return (usage.input * rates["input"] + usage.cache_read * rates["read"]
                + usage.cache_write_5m * rates["write_5m"] + usage.cache_write_1h * rates["write_1h"]
                + usage.output * rates["output"]) / 1e6
