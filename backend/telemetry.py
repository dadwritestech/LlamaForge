"""Small thread-safe caches for slow host telemetry probes.

GPU tools such as nvidia-smi take tens of milliseconds to start, especially
inside WSL2.  Dashboard requests should share a recent result instead of
spawning the same process for every browser poll.
"""
import copy, threading, time


class TimedCache:
    """Cache a zero-argument probe for ``ttl`` seconds.

    Values are copied both into and out of the cache so request handlers cannot
    accidentally mutate the snapshot seen by later requests.
    """

    def __init__(self, probe, ttl, clock=time.monotonic):
        self.probe = probe
        self.ttl = ttl
        self.clock = clock
        self._lock = threading.Lock()
        self._value = None
        self._at = None

    def get(self):
        with self._lock:
            now = self.clock()
            if self._at is None or now - self._at >= self.ttl:
                self._value = copy.deepcopy(self.probe())
                self._at = now
            return copy.deepcopy(self._value)
