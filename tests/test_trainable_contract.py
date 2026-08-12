"""Static contract checks for RFDETRTrainable that need no model build.

Importing the class is cheap: the heavy ``rfdetr`` training stack is imported
lazily inside ``__init__``, not at module import, so these run in plain CI.
"""

from cuvis_ai_rfdetr.node.rfdetr_trainable import RFDETRTrainable


def test_context_port_is_optional():
    """The executor injects Context as a call-site kwarg, never through a port.

    If the ``context`` port is not optional, pipeline graph pre-flight
    (``_validate_graph_inputs``) rejects the node with "missing required inputs:
    ['context']" because Context is in neither the batch nor a connection.
    """
    spec = RFDETRTrainable.INPUT_SPECS["context"]
    assert spec.optional is True


def test_targets_mask_port_is_optional():
    """targets_mask is required only in TRAIN (guarded in forward), optional in the spec."""
    spec = RFDETRTrainable.INPUT_SPECS["targets_mask"]
    assert spec.optional is True
