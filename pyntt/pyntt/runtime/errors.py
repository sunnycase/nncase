"""PyNTT runtime exceptions."""


class PyNTTError(Exception):
    """Base class for PyNTT runtime failures."""


class PyNTTSpecError(PyNTTError):
    """Raised when a generated PyNTT spec is invalid."""


class PyNTTArgumentError(PyNTTError):
    """Raised when runtime inputs do not match a function spec."""


class PyNTTBackendError(PyNTTError):
    """Raised when backend execution fails."""
