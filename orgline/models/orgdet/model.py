# orgline ORGDET 🚀, AGPL-3.0 license

from orgline.engine.model import Model
from orgline.models import orgdet  # noqa
from orgline.nn.tasks import ClassificationModel, DetectionModel, PoseModel, SegmentationModel


class ORGDET(Model):
    """ORGDET (You Only Look Once) object detection model."""

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            'classify': {
                'model': ClassificationModel,
                'trainer': orgdet.classify.ClassificationTrainer,
                'validator': orgdet.classify.ClassificationValidator,
                'predictor': orgdet.classify.ClassificationPredictor, },
            'detect': {
                'model': DetectionModel,
                'trainer': orgdet.detect.DetectionTrainer,
                'validator': orgdet.detect.DetectionValidator,
                'predictor': orgdet.detect.DetectionPredictor, },
            'segment': {
                'model': SegmentationModel,
                'trainer': orgdet.segment.SegmentationTrainer,
                'validator': orgdet.segment.SegmentationValidator,
                'predictor': orgdet.segment.SegmentationPredictor, },
            'pose': {
                'model': PoseModel,
                'trainer': orgdet.pose.PoseTrainer,
                'validator': orgdet.pose.PoseValidator,
                'predictor': orgdet.pose.PosePredictor, }, }
