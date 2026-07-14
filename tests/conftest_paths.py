"""Put backend/ on sys.path so tests can `import wsl`, `import vllm_ctl`, etc.
Imported for its side effect at the top of every test module."""
import os, sys

BACKEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
