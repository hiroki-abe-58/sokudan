"""sokudan — Japanese System One decision model.

Typed answers + calibrated probabilities in a single forward pass. No text generation.

    import sokudan
    agent = sokudan.load("runs/s0/model.pt")
    result = agent.predict({"body": "先月の請求が二重になっています"}, questions)
"""

from sokudan.predict import Agent, load

__version__ = "0.1.1"
__all__ = ["Agent", "load", "__version__"]
