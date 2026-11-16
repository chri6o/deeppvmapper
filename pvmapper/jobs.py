"""
Job bookkeeping shared by the CLI and the web UI.

Each job lives in its own folder under `jobs_dir`:
  job.json        parameters + status (rewritten as the job progresses)
  results.sqlite  per-building / per-cell results, used to resume after a crash
  chips/          optional JPEG previews (image | image with PV mask)
  exports         buildings.csv, buildings.geojson, pv_polygons.geojson, …
"""

from __future__ import annotations

import collections
import json
import os
import sqlite3
import threading
import time
import traceback
import uuid


class Cancelled(Exception):
    pass


class Job:
    def __init__(self, jobs_dir: str, kind: str, params: dict, job_id: str | None = None):
        self.id = job_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.kind = kind
        self.params = params
        self.dir = os.path.join(jobs_dir, self.id)
        os.makedirs(self.dir, exist_ok=True)
        self.status = "queued"          # queued | running | done | failed | cancelled
        self.total = 0
        self.done = 0
        self.counters = collections.Counter()
        self.error = None
        self.started = None
        self.finished = None
        self.log_lines = collections.deque(maxlen=400)
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._last_save = 0.0

    # -- state -----------------------------------------------------------------

    def log(self, msg: str):
        line = time.strftime("%H:%M:%S ") + msg
        self.log_lines.append(line)
        print(f"[{self.id}] {msg}", flush=True)

    def cancel(self):
        self._cancel.set()

    def check_cancel(self):
        if self._cancel.is_set():
            raise Cancelled()

    def step(self, n=1, **counts):
        with self._lock:
            self.done += n
            self.counters.update(counts)
        if time.time() - self._last_save > 5:
            self.save()

    def as_dict(self) -> dict:
        elapsed = (self.finished or time.time()) - self.started if self.started else 0
        rate = self.done / elapsed if elapsed > 0 else 0
        eta = (self.total - self.done) / rate if rate > 0 and self.total else None
        return {
            "id": self.id, "kind": self.kind, "status": self.status, "params": self.params,
            "total": self.total, "done": self.done, "counters": dict(self.counters),
            "error": self.error, "started": self.started, "finished": self.finished,
            "elapsed_s": round(elapsed, 1), "rate_per_s": round(rate, 3),
            "eta_s": round(eta) if eta else None, "log": list(self.log_lines)[-60:],
            "files": sorted(f for f in os.listdir(self.dir)
                            if os.path.isfile(os.path.join(self.dir, f))) if os.path.isdir(self.dir) else [],
        }

    def save(self):
        self._last_save = time.time()
        tmp = os.path.join(self.dir, "job.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(self.dir, "job.json"))

    @classmethod
    def load(cls, jobs_dir: str, job_id: str) -> "Job":
        with open(os.path.join(jobs_dir, job_id, "job.json"), encoding="utf-8") as f:
            d = json.load(f)
        job = cls(jobs_dir, d["kind"], d["params"], job_id)
        job.status, job.total, job.done = d["status"], d["total"], d["done"]
        job.counters.update(d.get("counters") or {})
        job.error, job.started, job.finished = d.get("error"), d.get("started"), d.get("finished")
        for line in d.get("log") or []:
            job.log_lines.append(line)
        return job

    # -- execution -------------------------------------------------------------

    def run(self, fn):
        """Runs fn(job) with status handling. fn does the actual work."""
        self.status = "running"
        self.started = self.started or time.time()
        self.finished = None
        self.error = None
        self.save()
        try:
            fn(self)
            self.status = "done"
        except Cancelled:
            self.status = "cancelled"
            self.log("Abgebrochen. Mit 'Fortsetzen' geht es an derselben Stelle weiter.")
        except Exception as e:                                   # noqa: BLE001
            self.status = "failed"
            self.error = f"{type(e).__name__}: {e}"
            self.log("FEHLER: " + self.error)
            self.log(traceback.format_exc()[-1500:])
        finally:
            self.finished = time.time()
            self.save()


def open_db(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def runner(kind: str):
    """job kind → function(job, cfg)"""
    if kind == "buildings":
        from .building_mode import run
    elif kind == "area":
        from .area_mode import run
    elif kind == "train":
        from .training import run
    elif kind == "ai_masks":
        from .ai_label import run
    elif kind == "sync":
        from .sync import run
    else:
        raise ValueError(f"unknown job kind {kind}")
    return run
