"""Claim-grounded, model-assisted QA generation for EpiSpace episodes.

The package deliberately separates executable spatial truth from language:
the simulator/compiler owns facts and answers, while a structured LLM backend
may only plan wording and realize allow-listed claims.
"""

from episode3d.qa_generation.pipeline import build_qa_generation_pilot

__all__ = ["build_qa_generation_pilot"]

