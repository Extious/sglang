from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import contextmanager
from typing import TypeVar

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

T = TypeVar("T")


@contextmanager
def tqdm_logging(logger: logging.Logger | None):
    if logger is None:
        yield
        return
    with logging_redirect_tqdm(loggers=[logger]):
        yield


def iter_with_progress(
    iterable: Iterable[T],
    *,
    total: int | None,
    desc: str,
    logger: logging.Logger | None,
) -> Iterable[T]:
    if logger is None:
        yield from iterable
        return
    with tqdm_logging(logger):
        yield from tqdm(iterable, total=total, desc=desc)
