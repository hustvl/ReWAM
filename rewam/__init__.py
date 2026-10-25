__version__ = "0.1.0"


def register():
    """Register ReWAM with the active StarVLA installation."""
    from .integration import register as register_frameworks

    register_frameworks()
