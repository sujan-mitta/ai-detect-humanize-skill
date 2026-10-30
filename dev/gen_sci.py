#!/usr/bin/env python3
"""Second API batch: dense scientific/technical topics x 3 prompt styles (see gen_texts.py)."""
import sys, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_texts as g
g.OUT = Path(__file__).resolve().parent / "corpus" / "api_sci"
g.TASKS = {
    "colloids": "discontinuous shear thickening in dense colloidal suspensions",
    "bvoc": "how boreal forests produce aerosols through monoterpene oxidation",
    "crispr": "how CRISPR-Cas9 edits a genome and the main off-target risks",
    "tcp": "how TCP congestion control behaves under packet loss",
    "protein": "how protein folding is predicted computationally",
    "superconduct": "why superconductors lose resistance below a critical temperature",
    "monetary": "the transmission channels of monetary policy to the real economy",
    "plate": "how plate tectonics explains earthquake distribution",
    "bayes": "Bayesian versus frequentist inference in clinical trials",
    "battery": "degradation mechanisms in lithium-ion batteries",
}
g.STYLES = {k: v for k, v in g.STYLES.items() if k in ("plain", "casual", "persona")}
use = {"gemini-3.1-flash-lite", "gpt-oss-120b", "qwen3.8-27b", "nemotron-ultra"}
g.MODELS = {k: v for k, v in g.MODELS.items() if k in use}
g.main()
