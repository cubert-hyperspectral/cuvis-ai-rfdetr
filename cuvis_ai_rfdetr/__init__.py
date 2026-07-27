"""cuvis.ai plugin exposing the official Roboflow RF-DETR (Apache-2.0 tier) as
inference nodes (detection + segmentation) and a trainable node + criterion
loss for in-pipeline training."""

from cuvis_ai_rfdetr.node.rfdetr_detector import RFDETRDetector
from cuvis_ai_rfdetr.node.rfdetr_loss import RFDETRCriterionLoss
from cuvis_ai_rfdetr.node.rfdetr_segmenter import RFDETRSegmenter
from cuvis_ai_rfdetr.node.rfdetr_trainable import RFDETRTrainable

__all__ = ["RFDETRCriterionLoss", "RFDETRDetector", "RFDETRSegmenter", "RFDETRTrainable"]
