"""Local Batch API: the on-disk store and the runner.

Store: uploaded and result files under <root>/files/<id>.jsonl (+ .json meta),
batches as <root>/<batch_id>.json. Runner: one batch at a time, oldest first,
one request after another to the router (its parallel slots do the batching).
Every line is written before the batch's next_line moves on, so a restarted
panel carries on where it stopped, re-sending at most the request in flight.

"idle" batches wait to START until the router has been quiet QUIET_SECS (the
stats poller can't tell our own requests from anyone else's, so the gate can't
apply mid-run). Any batch pauses while proving, a speed compare or Auto-Forge
holds the GPUs. No record of who asked: no IPs, no client ids."""
import glob
import json
import os
import threading
import time

import atomicio
import batchapi

QUIET_SECS = 20 * 60
WHEN = ("now", "idle")
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_FILES = 200


class Store:
    def __init__(self, root, max_bytes=MAX_FILE_BYTES, max_files=MAX_FILES):
        self.root, self.max_bytes, self.max_files = root, max_bytes, max_files
        self.fdir = os.path.join(root, "files")
        os.makedirs(self.fdir, exist_ok=True)

    # ---- files
    def _fpath(self, fid, ext):
        return os.path.join(self.fdir, fid + ext)

    def add_file(self, filename, purpose, data):
        if purpose != "batch":
            raise ValueError("purpose must be 'batch'")
        if len(data) > self.max_bytes:
            raise ValueError(f"file too big (limit {self.max_bytes // (1024 * 1024)} MB)")
        if len(self.files()) >= self.max_files:
            raise ValueError(f"{self.max_files} files stored already; remove some first")
        return self._put(filename, purpose, data)

    def _put(self, filename, purpose, data):
        fid = batchapi.new_id("file")
        with open(self._fpath(fid, ".jsonl"), "wb") as f:
            f.write(data)
        return self._meta(fid, filename, purpose, len(data))

    def _meta(self, fid, filename, purpose, size):
        meta = {"id": fid, "bytes": size, "created_at": int(time.time()),
                "filename": filename, "purpose": purpose}
        atomicio.write_json(self._fpath(fid, ".json"), meta)
        return batchapi.file_object(meta)

    def file(self, fid):
        if not batchapi.valid_id(fid, "file"):
            return None
        try:
            with open(self._fpath(fid, ".json"), encoding="utf-8") as f:
                return batchapi.file_object(json.load(f))
        except (OSError, ValueError, KeyError):
            return None

    def files(self):
        out = [self.file(os.path.basename(p)[:-5]) for p in glob.glob(os.path.join(self.fdir, "file-*.json"))]
        return sorted((f for f in out if f), key=lambda f: f["created_at"], reverse=True)

    def content(self, fid):
        if not self.file(fid):
            return None
        with open(self._fpath(fid, ".jsonl"), "rb") as f:
            return f.read()

    def remove_file(self, fid):
        if not self.file(fid):
            return False
        for ext in (".json", ".jsonl"):
            try:
                os.remove(self._fpath(fid, ext))
            except OSError:
                pass
        return True

    def append(self, fid, obj):
        with open(self._fpath(fid, ".jsonl"), "ab") as f:
            f.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))

    def new_result_file(self, name):
        """An empty result file; its meta is written when the batch ends."""
        fid = batchapi.new_id("file")
        open(self._fpath(fid, ".jsonl"), "wb").close()
        return fid

    def seal(self, fid, name):
        """Publish a result file if it has anything in it, else drop it."""
        path = self._fpath(fid, ".jsonl")
        size = os.path.getsize(path) if os.path.exists(path) else 0
        if not size:
            try:
                os.remove(path)
            except OSError:
                pass
            return None
        self._meta(fid, name, "batch_output", size)
        return fid

    # ---- batches
    def _bpath(self, bid):
        return os.path.join(self.root, bid + ".json")

    def save_batch(self, b):
        atomicio.write_json(self._bpath(b["id"]), b)

    def batch(self, bid):
        if not batchapi.valid_id(bid, "batch"):
            return None
        try:
            with open(self._bpath(bid), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def batches(self):
        out = [self.batch(os.path.basename(p)[:-5]) for p in glob.glob(os.path.join(self.root, "batch_*.json"))]
        return sorted((b for b in out if b), key=lambda b: (b["created_at"], b["id"]), reverse=True)


class Runner:
    """send(body) -> (status, json) or raises; ensure(model) loads it or raises;
    busy() is True while another run holds the GPUs; last_activity() is the
    epoch of the router's last generation."""

    def __init__(self, store, send, ensure, busy, last_activity, clock=time.time):
        self.store, self._send, self._ensure = store, send, ensure
        self._busy, self._last, self._clock = busy, last_activity, clock
        self._lock = threading.Lock()
        self._cur = None            # the batch dict being run (shared with cancel)

    def create(self, input_file_id, endpoint, window, metadata=None, when="now"):
        if when not in WHEN:
            raise ValueError("when must be 'now' or 'idle'")
        data = self.store.content(input_file_id)
        if data is None:
            raise LookupError(f"no such file: {input_file_id}")
        reqs, model = batchapi.parse_input(data, endpoint)
        b = batchapi.new_batch(input_file_id, endpoint, window, len(reqs), model,
                               metadata=metadata, when=when, now=self._clock())
        self.store.save_batch(b)
        return b

    def cancel(self, bid):
        with self._lock:
            cur = self._cur if self._cur and self._cur["id"] == bid else None
            b = cur or self.store.batch(bid)
            if not b:
                raise LookupError(f"no such batch: {bid}")
            if b["status"] == "cancelling":
                return batchapi.batch_object(b)
            if batchapi.is_done(b) or b["status"] == "finalizing":
                raise ValueError(f"the batch is already {b['status']}")
            now = self._clock()
            batchapi.move(b, "cancelling", now=now)
            if not cur:                      # not running: nothing to wait for
                self._close(b, "cancelled")
            self.store.save_batch(b)
            return batchapi.batch_object(b)

    # ---- running
    def _next(self):
        live = [b for b in self.store.batches() if not batchapi.is_done(b)]
        return min(live, key=lambda b: (b["created_at"], b["id"]), default=None)

    def _close(self, b, status):
        """End a batch: publish its result files, then the final status."""
        for key, internal, name in (("output_file_id", "_out", "output"), ("error_file_id", "_err", "errors")):
            if b.get(internal):
                b[key] = self.store.seal(b[internal], f"{b['id']}_{name}.jsonl")
        batchapi.move(b, status, now=self._clock())

    def _save(self, b):
        with self._lock:
            self.store.save_batch(b)

    def step(self):
        """Work on the oldest open batch until it ends or has to wait.
        -> "idle" (nothing to do), "waiting", or "done"."""
        b = self._next()
        if not b:
            return "idle"
        with self._lock:
            self._cur = b
        try:
            return self._work(b)
        finally:
            with self._lock:
                self._cur = None

    def _fail(self, b, code, message):
        with self._lock:
            b["errors"] = {"object": "list", "data": [{"code": code, "message": message, "line": None}]}
            self._close(b, "failed")
            self.store.save_batch(b)
        return "done"

    def _expired(self, b):
        if self._clock() <= b["expires_at"]:
            return False
        with self._lock:
            self._close(b, "expired")
            self.store.save_batch(b)
        return True

    def _work(self, b):
        if b["status"] == "cancelling":
            with self._lock:
                self._close(b, "cancelled")
                self.store.save_batch(b)
            return "done"
        if self._expired(b):
            return "done"
        if b["status"] == "validating":
            if b.get("when") == "idle" and self._clock() - (self._last() or 0) < QUIET_SECS:
                return "waiting"
            if self._busy():
                return "waiting"
            with self._lock:
                if b["status"] != "validating":      # cancelled meanwhile
                    return self._work(b)
                b["_out"] = self.store.new_result_file("output")
                b["_err"] = self.store.new_result_file("errors")
                batchapi.move(b, "in_progress", now=self._clock())
                self.store.save_batch(b)
        data = self.store.content(b["input_file_id"])
        if data is None:
            return self._fail(b, "input_file_missing", "the input file was removed")
        reqs, _ = batchapi.parse_input(data, b["endpoint"])
        try:
            self._ensure(b["model"])
        except Exception as e:  # noqa: BLE001 - shown to the user as is
            return self._fail(b, "model_load_failed", str(e) or type(e).__name__)
        while b["next_line"] < len(reqs):
            if b["status"] == "cancelling":
                return self._work(b)
            if self._expired(b):
                return "done"
            if self._busy():
                return "waiting"
            r = reqs[b["next_line"]]
            try:
                status, body = self._send(r["body"])
            except Exception as e:  # noqa: BLE001 - the router is down or went away
                status, body = None, str(e) or type(e).__name__
            with self._lock:
                if status is None:
                    self.store.append(b["_err"], batchapi.error_line(r["custom_id"], "router_unreachable", body))
                    b["request_counts"]["failed"] += 1
                elif batchapi.ok_status(status):
                    self.store.append(b["_out"], batchapi.result_line(r["custom_id"], status, body))
                    b["request_counts"]["completed"] += 1
                else:
                    line = batchapi.result_line(r["custom_id"], status, body)
                    self.store.append(b["_err"], line)
                    b["request_counts"]["failed"] += 1
                b["next_line"] += 1
                self.store.save_batch(b)
        with self._lock:
            if b["status"] == "cancelling":
                self._close(b, "cancelled")
            else:
                batchapi.move(b, "finalizing", now=self._clock())
                self._close(b, "completed")
            self.store.save_batch(b)
        return "done"

    def run_forever(self, every=5):
        while True:
            try:
                state = self.step()
            except Exception:  # noqa: BLE001 - never kill the panel over a batch
                state = "waiting"
            if state != "done":
                time.sleep(every)

    def start(self):
        threading.Thread(target=self.run_forever, daemon=True, name="batches").start()
