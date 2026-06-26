"""Project-wide exception hierarchy."""


class LeHomeError(Exception):
    """Base exception for all LeHome errors."""


class HFSyncError(LeHomeError):
    """Error in HuggingFace sync daemon communication."""
