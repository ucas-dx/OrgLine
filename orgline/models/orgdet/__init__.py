# orgline ORGDET 🚀, AGPL-3.0 license

from orgline.models.orgdet import classify, detect, pose, segment

from .model import ORGDET

__all__ = 'classify', 'segment', 'detect', 'pose', 'ORGDET'
