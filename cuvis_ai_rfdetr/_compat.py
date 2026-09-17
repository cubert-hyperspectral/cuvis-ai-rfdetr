"""Compatibility with cuvis-ai-core before and after 0.15.

cuvis-ai-core 0.15 made execution stages class-level (``Node.EXECUTION_STAGES``) and removed
both ``Node.consume_base_kwargs`` and the ``execution_stages=`` constructor kwarg. The plugin
nodes route their base kwargs through :func:`base_kwargs` so one code path constructs on either
side of that break.
"""

from __future__ import annotations

from typing import Any

from cuvis_ai_core.node.node import Node

# True on cuvis-ai-core < 0.15, where stages are a per-instance constructor kwarg.
CORE_HAS_INSTANCE_STAGES = hasattr(Node, "consume_base_kwargs")


def base_kwargs(kwargs: dict[str, Any], execution_stages: Any = None) -> dict[str, Any]:
    """Pop the base-node kwargs a plugin node forwards to ``Node.__init__``.

    Returns a mapping to splat into ``super().__init__``: ``name`` always, plus
    ``execution_stages`` on cores that still take per-instance stages — either the fixed
    set a node passes (loss nodes) or the yaml override left in ``kwargs``. On core >= 0.15
    stages are the class's ``EXECUTION_STAGES``: a yaml's inert ``execution_stages: null``
    is dropped, any other value is rejected the way core itself rejects it.
    """
    if CORE_HAS_INSTANCE_STAGES:
        name, yaml_stages = Node.consume_base_kwargs(kwargs)
        stages = execution_stages if execution_stages is not None else yaml_stages
        return {"name": name, "execution_stages": stages}
    name = kwargs.pop("name", None)
    leftover = kwargs.pop("execution_stages", None)
    if leftover is not None:
        raise TypeError(
            "execution_stages is class-level on cuvis-ai-core >= 0.15 (a node's "
            "EXECUTION_STAGES); remove it from the yaml hparams."
        )
    return {"name": name}
