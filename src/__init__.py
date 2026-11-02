"""AnyMo modules. Inference imports only its required components."""

def register_all_modules():
    """Register training datasets, losses, evaluators and model components."""
    from . import models, metrics, datasets, losses
    from .models import evaluator
