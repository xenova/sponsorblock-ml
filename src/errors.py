class SponsorBlockException(Exception):
    """Base class for all sponsor block exceptions"""
    pass


class PredictionException(SponsorBlockException):
    """An exception occurred while predicting sponsor segments"""
    pass


class TranscriptError(SponsorBlockException):
    """An exception occurred while retrieving the video transcript"""
    pass


class ModelError(SponsorBlockException):
    """Base class for model-related errors"""
    pass


class ModelLoadError(ModelError):
    """An exception occurred while loading the model"""
    pass
