#!/usr/bin/env python3
"""llama-server on the Tesseract backend against the CPU backend (ctest ggml_tesseract_server_tiny).

Two llama-server processes run the same model with the M7 settings (-ngl 99 -fa on -ctk q8_0 -ctv q8_0,
LLAMA_ATTN_ROT_DISABLE=1) and the same slots (default -np 2 --kv-unified): one on the device
(TSR_DEVICE=sim unless --device says otherwise, TSR_STATS=1), one on the CPU (TSR_DEVICE=none). Each
gets a free localhost port. The script then:

1. waits for /health (200) on both and reads /v1/models on the device server;
2. sends the same requests to both, temperature 0: a non-streaming chat completion, the same chat
   with stream=true (SSE: every event is JSON, the role comes first, a finish_reason and
   "data: [DONE]" end the stream), a /v1/completions request, --np different chat requests at
   once (several sequences in the unified KV cache, decoded together), and the first chat prompt
   again. By then every slot has been reused, so the server saves each slot's KV state into its RAM
   prompt cache (the device reads it back) and restores this prompt from it (the device writes it
   back); the response reports how many prompt tokens came from the cache;
3. checks that every device text equals the CPU's, that the streamed text equals the non-streamed
   one and that the last request was served from the prompt cache;
4. stops both with SIGINT and requires exit status 0, then reads the device's TSR_STATS summary
   (printed when the backend is freed): at least one launch per generated token, exactly one launch
   in every graph_compute (the M7 one-launch decode), and no "tesseract:" error line in the log.

The time limits add up to less than the ctest TIMEOUT: --timeout for both servers to start, 5 s for
/v1/models, --request-timeout for each of at most 5 sequential requests, 2 x STOP_TIMEOUT to stop the
servers. A hang therefore ends in a FAIL line and report.json, not in a ctest kill.

Python standard library only. Exit status: 0 pass, 1 fail, 2 usage or tool error.

By hand (from the repository root):
  python3 sw/ggml-tesseract/tests/server_smoke.py --server build/dev/bin/llama-server \\
      --model build/dev/models/tiny/qwen2-tiny-q8_0.gguf --out-dir /tmp/server_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

CHAT_PROMPTS = [
    "once upon a time there was a small robot",
    "the cat sat on the mat",
    "the river starts high in the hills",
    "the dog ran in the park",
]
COMPLETION_PROMPT = "the river starts high in the hills"
STOP_TIMEOUT = 10.0     # seconds a server gets to exit after SIGINT


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """One llama-server process; its stdout and stderr go to <out_dir>/<name>.log."""

    def __init__(self, name: str, a: argparse.Namespace, device: str, out_dir: Path):
        self.name = name
        self.a = a
        self.device = device
        self.log = out_dir / f"{name}.log"
        self.proc: subprocess.Popen | None = None
        self.port = 0
        self.rc: int | None = None

    def start(self) -> None:
        env = dict(os.environ)
        env["TSR_DEVICE"] = self.device
        env["LLAMA_ATTN_ROT_DISABLE"] = "1"
        env["TSR_STATS"] = "1"
        self.port = free_port()
        cmd = [self.a.server, "-m", self.a.model, "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
               "-np", str(self.a.np), "-c", str(self.a.ctx), "-t", str(self.a.threads),
               "--host", "127.0.0.1", "--port", str(self.port)]
        if self.a.np > 1:
            cmd.append("--kv-unified")     # M7 keeps one KV stream (SW §8 R10)
        print(f"server_smoke: [{self.name}] TSR_DEVICE={self.device} $ {' '.join(cmd)}", flush=True)
        self.fh = open(self.log, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.fh, stderr=subprocess.STDOUT, env=env)

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def wait_healthy(self, deadline: float) -> bool:
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(self.url("/health"), timeout=2) as r:
                    if r.status == 200:
                        return True
            except (urllib.error.URLError, ConnectionError, OSError):
                pass        # 503 while loading, or not listening yet
            time.sleep(0.1)
        return False

    def bind_failed(self) -> bool:
        return self.proc.poll() is not None and "couldn't bind" in self.log.read_text(errors="replace")

    def stop(self, timeout: float) -> int | None:
        if self.proc is None:
            return None
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
                self.rc = None
                self.fh.close()
                return None
        self.rc = self.proc.returncode
        self.fh.close()
        return self.rc

    # ---- requests ----------------------------------------------------------------------------
    def post(self, path: str, body: dict):
        req = urllib.request.Request(self.url(path), data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=self.a.request_timeout)

    def chat(self, prompt: str) -> dict:
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": self.a.max_tokens, "temperature": 0}
        with self.post("/v1/chat/completions", body) as r:
            j = json.load(r)
        c = j["choices"][0]
        return {"text": c["message"]["content"], "finish": c["finish_reason"], "tokens": j["usage"]["completion_tokens"],
                "cached": j.get("timings", {}).get("cache_n", 0)}

    def chat_stream(self, prompt: str) -> dict:
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": self.a.max_tokens, "temperature": 0,
                "stream": True}
        parts, problems, events, finish, done, first_role = [], [], 0, None, False, None
        with self.post("/v1/chat/completions", body) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if not line.startswith("data: "):
                    problems.append(f"not an SSE data line: {line[:80]!r}")
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    done = True
                    break
                try:
                    j = json.loads(data)
                except json.JSONDecodeError:
                    problems.append(f"event is not JSON: {data[:80]!r}")
                    continue
                events += 1
                for c in j.get("choices", []):
                    delta = c.get("delta", {})
                    if first_role is None and "role" in delta:
                        first_role = (events, delta["role"])
                    if delta.get("content"):
                        parts.append(delta["content"])
                    if c.get("finish_reason"):
                        finish = c["finish_reason"]
        if not done:
            problems.append("no 'data: [DONE]' at the end")
        if first_role != (1, "assistant"):
            problems.append(f"the first event does not carry role 'assistant' ({first_role})")
        if finish is None:
            problems.append("no finish_reason in the stream")
        return {"text": "".join(parts), "finish": finish, "events": events, "problems": problems}

    def completion(self, prompt: str) -> dict:
        body = {"prompt": prompt, "max_tokens": self.a.max_tokens, "temperature": 0}
        with self.post("/v1/completions", body) as r:
            j = json.load(r)
        c = j["choices"][0]
        return {"text": c["text"], "finish": c["finish_reason"], "tokens": j["usage"]["completion_tokens"]}

    def concurrent_chats(self, prompts: list[str]) -> list[dict]:
        out: list = [None] * len(prompts)

        def run(i: int) -> None:
            try:
                out[i] = self.chat(prompts[i])
            except Exception as e:          # reported as a failed request
                out[i] = {"error": f"{type(e).__name__}: {e}"}

        threads = [threading.Thread(target=run, args=(i,)) for i in range(len(prompts))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return out

    def run_requests(self) -> dict:
        res = {
            "chat": self.chat(CHAT_PROMPTS[0]),
            "chat_stream": self.chat_stream(CHAT_PROMPTS[0]),
            "completion": self.completion(COMPLETION_PROMPT),
        }
        if self.a.np > 1:
            prompts = [CHAT_PROMPTS[(1 + i) % len(CHAT_PROMPTS)] for i in range(self.a.np)]
            res["concurrent"] = self.concurrent_chats(prompts)
        # the first prompt again, after the slots held other prompts: restored from the prompt cache
        res["restore"] = self.chat(CHAT_PROMPTS[0])
        return res


def start_servers(servers: list[Server], timeout: float) -> list[str]:
    """Start every server (a port taken between free_port and bind is retried); the failures."""
    deadline = time.monotonic() + timeout
    pending = list(servers)
    for attempt in range(3):
        for s in pending:
            s.start()
        failed = [s for s in pending if not s.wait_healthy(deadline)]
        retry = [s for s in failed if s.bind_failed()]
        if not retry or attempt == 2:
            return [f"[{s.name}] not healthy within {timeout:.0f} s (log: {s.log})" for s in failed]
        for s in retry:
            s.stop(5)
            print(f"server_smoke: [{s.name}] port {s.port} was taken, retrying", flush=True)
        pending = retry
    return []


STATS_RE = re.compile(r"stats: graph_compute (\d+) calls.*?; launches (\d+) .*?([\d.]+) per scheduler pass")
LAUNCH_HIST_RE = re.compile(r"graph_compute calls with 0 / 1 / 2 / >= 3 launches: (\d+) / (\d+) / (\d+) / (\d+)")
# the backend's errors (GGML_LOG_ERROR "tesseract: ..."; level E in llama.cpp's log format)
ERROR_RE = re.compile(r"(^|\s)E tesseract:|^tesseract:")


def device_stats(log: Path) -> dict | None:
    text = log.read_text(errors="replace")
    m = None
    for m in STATS_RE.finditer(text):
        pass
    h = None
    for h in LAUNCH_HIST_RE.finditer(text):
        pass
    if m is None or h is None:
        return None
    return {"graph_compute": int(m.group(1)), "launches": int(m.group(2)), "per_pass": float(m.group(3)),
            "hist": [int(x) for x in h.groups()],
            "errors": [l for l in text.splitlines() if ERROR_RE.search(l)]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True, help="llama-server binary")
    ap.add_argument("--model", required=True, help="GGUF (the tiny Qwen2 model in the ctest)")
    ap.add_argument("--out-dir", default=None, help="logs and report.json (default: the current directory)")
    ap.add_argument("--device", default="sim", help="TSR_DEVICE of the device server (default sim)")
    ap.add_argument("--np", type=int, default=2, help="server slots; > 1 adds --kv-unified and a concurrent round")
    ap.add_argument("-c", "--ctx", type=int, default=1024, help="context size")
    ap.add_argument("-t", "--threads", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=20.0, help="seconds until both servers are healthy")
    ap.add_argument("--request-timeout", type=float, default=12.0, help="seconds without data on one request")
    a = ap.parse_args()

    for p in (a.server, a.model):
        if not Path(p).exists():
            print(f"server_smoke: error: {p} does not exist")
            return 2
    out_dir = Path(a.out_dir) if a.out_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    dev = Server("device", a, a.device, out_dir)
    cpu = Server("cpu", a, "none", out_dir)
    fails: list[str] = []
    results: dict = {}
    try:
        fails += start_servers([dev, cpu], a.timeout)
        if not fails:
            with urllib.request.urlopen(dev.url("/v1/models"), timeout=5) as r:
                models = json.load(r)
            ids = [m.get("id", "") for m in models.get("data", [])]
            if not any(Path(i).name == Path(a.model).name for i in ids):
                fails.append(f"/v1/models does not list {Path(a.model).name}: {ids}")
            # both servers at once (the sim is the slow one)
            th = {}
            for s in (dev, cpu):
                def run(s=s):
                    try:
                        results[s.name] = s.run_requests()
                    except Exception as e:
                        results[s.name] = {"error": f"{type(e).__name__}: {e}"}
                th[s.name] = threading.Thread(target=run)
                th[s.name].start()
            for t in th.values():
                t.join()
    finally:
        rc_dev = dev.stop(STOP_TIMEOUT)
        rc_cpu = cpu.stop(STOP_TIMEOUT)

    for name, rc in (("device", rc_dev), ("cpu", rc_cpu)):
        if rc != 0:
            fails.append(f"[{name}] exit status {rc} after SIGINT (None: killed after {STOP_TIMEOUT:.0f} s; log: {out_dir / (name + '.log')})")
    d, c = results.get("device", {}), results.get("cpu", {})
    for name, r in (("device", d), ("cpu", c)):
        if "error" in r:
            fails.append(f"[{name}] request failed: {r['error']}")
    if "error" not in d and "error" not in c and d and c:
        for key in ("chat", "chat_stream", "completion", "restore"):
            print(f"server_smoke: {key:12s} device {d[key]['text']!r} ({d[key]['finish']})")
            if d[key]["text"] != c[key]["text"]:
                fails.append(f"{key}: device {d[key]['text']!r} != cpu {c[key]['text']!r}")
        for name, r in (("device", d), ("cpu", c)):
            for p in r["chat_stream"]["problems"]:
                fails.append(f"[{name}] stream: {p}")
            if r["chat_stream"]["text"] != r["chat"]["text"]:
                fails.append(f"[{name}] streamed text {r['chat_stream']['text']!r} != non-streamed {r['chat']['text']!r}")
        for name, r in (("device", d), ("cpu", c)):
            if r["restore"]["cached"] <= 0:
                fails.append(f"[{name}] the repeated prompt was not served from the prompt cache (cached tokens "
                             f"{r['restore']['cached']}): the KV state save and restore were not exercised")
        for i, (x, y) in enumerate(zip(d.get("concurrent", []), c.get("concurrent", []))):
            if "error" in x or "error" in y:
                fails.append(f"concurrent request {i}: device {x.get('error')}, cpu {y.get('error')}")
                continue
            print(f"server_smoke: concurrent {i}  device {x['text']!r} ({x['finish']})")
            if x["text"] != y["text"]:
                fails.append(f"concurrent request {i}: device {x['text']!r} != cpu {y['text']!r}")

    st = device_stats(dev.log) if dev.log.exists() else None
    if st is None:
        fails.append(f"no TSR_STATS summary in the device log ({dev.log}): the backend did not run or was not freed")
    else:
        tokens = 0
        if "error" not in d and d:
            tokens = (d["chat"]["tokens"] + d["completion"]["tokens"] + d["restore"]["tokens"] +
                      sum(x.get("tokens", 0) for x in d.get("concurrent", [])))
        print(f"server_smoke: device: {st['graph_compute']} graph_compute calls, {st['launches']} launches "
              f"({st['per_pass']:.2f} per scheduler pass), graph_compute calls with 0 / 1 / 2 / >= 3 launches: "
              f"{' / '.join(map(str, st['hist']))}; {tokens} tokens generated in the non-streamed requests")
        if st["launches"] < max(tokens, 1):
            fails.append(f"device launches {st['launches']} < {max(tokens, 1)} generated tokens: the backend did not run them")
        if st["hist"][0] or st["hist"][2] or st["hist"][3]:
            fails.append(f"a graph_compute did not make exactly one launch (0 / 1 / 2 / >= 3: {st['hist']})")
        for e in st["errors"][:5]:
            fails.append(f"device log: {e.strip()}")

    elapsed = time.monotonic() - t0
    report = {"model": a.model, "np": a.np, "max_tokens": a.max_tokens, "device": d, "cpu": c, "stats": st,
              "exit": {"device": rc_dev, "cpu": rc_cpu}, "seconds": round(elapsed, 2), "failures": fails}
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    for f in fails:
        print(f"server_smoke: FAIL: {f}")
    print(f"server_smoke: {'PASS' if not fails else 'FAIL'} in {elapsed:.1f} s (logs and report.json: {out_dir})")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
