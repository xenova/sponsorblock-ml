class SponsorBlockException(Exception):
    """Base class for all sponsor block exceptions"""
    pass


class PredictionException(SponsorBlockException):
    """An exception was occurred while predicting sponsor segments"""
    pass


class TranscriptError(SponsorBlockException):
    """An exception was occurred while retrieving the video transcript"""
    pass
