from typing import Any

def default(val: Any, default_val: Any) -> Any:
    return val if val is not None else default_val