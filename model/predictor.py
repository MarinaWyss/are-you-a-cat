"""Human-readable interpretation of raw model probabilities.

The Streamlit app previously inlined a chain of ``if`` statements to turn a
prediction probability into a message. Extracting it here makes the thresholds
testable and keeps presentation logic out of the app.
"""
from dataclasses import dataclass

from steps.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Verdict:
    """A classification verdict.

    Attributes:
        is_cat (bool): Whether the model leans towards "cat".
        confident (bool): Whether the model is confident (outside the unsure
            band).
        message (str): A human-friendly explanation of the result.
    """
    is_cat: bool
    confident: bool
    message: str


def interpret_prediction(probability: float, config: dict) -> Verdict:
    """Maps a raw probability to a :class:`Verdict`.

    Args:
        probability (float): Model output in the ``[0, 1]`` range, where higher
            means more cat-like.
        config (dict): Config providing ``min_unsure`` and ``max_unsure``
            thresholds.

    Returns:
        (Verdict): Structured, human-readable verdict.

    Raises:
        ValueError: If ``probability`` is outside ``[0, 1]``.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"Probability must be in [0, 1], got {probability}."
        )

    min_unsure = config['min_unsure']
    max_unsure = config['max_unsure']

    if probability < min_unsure:
        return Verdict(
            is_cat=False,
            confident=True,
            message="I'm pretty sure you're not a cat.",
        )
    if min_unsure <= probability < max_unsure:
        return Verdict(
            is_cat=False,
            confident=False,
            message="I don't think you're a cat, but it's hard to tell.",
        )
    return Verdict(
        is_cat=True,
        confident=True,
        message="I'm pretty sure you are a cat.",
    )
