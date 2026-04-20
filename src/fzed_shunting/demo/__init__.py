from __future__ import annotations

from typing import Any

__all__ = ["build_demo_view_model"]


def build_demo_view_model(*args: Any, **kwargs: Any):
    from fzed_shunting.demo.view_model import build_demo_view_model as _build_demo_view_model

    return _build_demo_view_model(*args, **kwargs)
