"""cuvis.ai plugin exposing the official Roboflow RF-DETR (Apache-2.0 tier) as an inference node."""

from cuvis_ai_rfdetr.node.rfdetr_detector import RFDETRDetector

__all__ = ["RFDETRDetector"]
