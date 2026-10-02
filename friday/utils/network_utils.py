import logging
import time
from collections.abc import Callable
from typing import Any

import httpx


def safe_llm_call(func: Callable, *args, max_retries: int = 3, **kwargs) -> Any:
    """
    Wrapper for LLM calls to handle transient network errors (like WinError 10054).
    Implements exponential backoff.
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (
            ConnectionResetError,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            httpx.ReadTimeout,
        ) as e:
            last_exception = e
            wait_time = 2**attempt
            logging.warning(
                f"  [NETWORK] Transient error detected: {e}. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(wait_time)
        except Exception as e:
            # For non-network errors, we don't retry but we log them
            logging.error(f"  [NETWORK] Non-transient error: {e}")
            raise e

    logging.error(f"  [NETWORK] Maximum retries reached. Failing: {last_exception}")
    raise last_exception
