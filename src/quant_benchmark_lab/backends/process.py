"""Only processes started by this harness are terminated."""

import re
import socket
import subprocess
from pathlib import Path

from ..environment import engine_environment
from .base import BackendError


def offload_evidence(log: str) -> dict:
    matches = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers", log)
    if matches:
        actual, total = map(int, matches[-1])
        return {"full_offload": actual == total and total > 0,
                "offloaded_layers": actual, "total_layers": total, "source": "engine log"}
    return {"full_offload": None, "reason": "no recognized complete layer-offload log"}


class OwnedProcess:
    def __init__(self, log: Path, port: int):
        self.log, self.port, self.process, self.handle = log, port, None, None
        self.offset = 0

    def start(self, args: list[str], env: dict | None = None) -> None:
        if self.process is not None:
            raise BackendError("owned process is already running")
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                raise BackendError(f"port {self.port} is occupied; refusing to attach to another server")
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.offset = self.log.stat().st_size if self.log.exists() else 0
        self.handle = self.log.open("ab", buffering=0)
        try:
            self.process = subprocess.Popen(args, stdout=self.handle, stderr=self.handle, env=engine_environment() if env is None else env)
        except BaseException:
            self.handle.close()
            self.handle = None
            raise

    def log_text(self) -> str:
        if not self.log.exists():
            return ""
        with self.log.open("rb") as f:
            f.seek(self.offset)
            return f.read().decode("utf-8", errors="replace")

    def assert_alive(self) -> None:
        if self.process is None or self.process.poll() is not None:
            raise BackendError(f"server exited: {self.log_text()[-2000:]}")

    def stop(self, timeout: float) -> None:
        children = []
        if self.process is not None and self.process.poll() is None:
            import psutil

            try:
                children = psutil.Process(self.process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                pass
        try:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            if children:
                import psutil

                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(children, timeout=5)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                _, remaining = psutil.wait_procs(alive, timeout=5)
                if remaining:
                    raise BackendError("owned child processes did not exit")
        finally:
            if self.handle is not None:
                self.handle.close()
            self.process = self.handle = None
