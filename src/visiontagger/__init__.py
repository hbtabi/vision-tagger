"""vision-tagger: classical-CV image tagging and captioning with an optional vision-LLM."""

from .features import ImageFeatures, extract_features
from .tagger import TagResult, tag_image

__all__ = ["ImageFeatures", "TagResult", "extract_features", "tag_image"]
__version__ = "0.1.0"
