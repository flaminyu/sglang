"""JSON utilities for sglang_simulator."""

import json
from datetime import datetime
from typing import Any


class CustomJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling special types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)
