"""
Custom exception classes for the chest X-ray analysis system.
"""


class DICOMLoadError(Exception):
    """Raised when DICOM file cannot be loaded."""
    pass


class AnnotationFormatError(Exception):
    """Raised when annotation format is invalid."""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


class ModelTrainingError(Exception):
    """Raised when training fails."""
    pass


class InsufficientMemoryError(Exception):
    """Raised when GPU memory is insufficient."""
    pass
