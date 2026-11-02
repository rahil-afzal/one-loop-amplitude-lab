"""Loopback-only SSE interface for observing autonomous derivations."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .autonomous import run_autonomous_derivation
from .llm import DEFAULT_AUDIT_MODEL, DEFAULT_DERIVATION_MODEL, role_llm_from_env
from .task import Task

_TASK_NAME = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_MAX_REQUEST_BYTES = 4096
_TERMINAL_STATES = {"completed_verified", "completed_unverified", "failed", "retry_exhausted"}


class EventChannel:
    def __init__(self):
        self.condition = threading.Condition()
        self.events: list[dict] = []
        self.done = False

    def publish(self, event: dict) -> None:
        with self.condition:
            row = {**event, "id": len(self.events) + 1}
            self.events.append(row)
            if row.get("state") in _TERMINAL_STATES:
                self.done = True
            self.condition.notify_all()


_channels: dict[str, EventChannel] = {}
_channels_lock = threading.Lock()


def _start_derivation(run_id: str, task_name: str, max_retries: int) -> None:
    channel = _channels[run_id]
    try:
        task = Task.load(task_name)
        derivation_llm = role_llm_from_env("derivation", DEFAULT_DERIVATION_MODEL)
        audit_llm = role_llm_from_env("audit", DEFAULT_AUDIT_MODEL)
        result = run_autonomous_derivation(
            task,
            derivation_llm,
            audit_llm,
            max_step_retries=max_retries,
            on_event=channel.publish,
        )
        if result.status not in _TERMINAL_STATES:
            channel.publish(
                {"type": "server.finished", "state": result.status, "result": result.to_dict()}
            )
    except Exception as exc:
        channel.publish(
            {
                "type": "server.failed",
                "state": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


class _Handler(BaseHTTPRequestHandler):
    server_version = "AmplitudeLab/1"

    def log_message(self, _format, *_args):
        if _format or _args:
            return

    def _is_loopback_request(self) -> bool:
        host = urlparse(f"//{self.headers.get('Host', '')}").hostname
        if host not in ("127.0.0.1", "localhost"):
            return False
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        parsed_origin = urlparse(origin)
        return (
            parsed_origin.scheme == "http"
            and parsed_origin.hostname in ("127.0.0.1", "localhost")
            and parsed_origin.netloc == self.headers.get("Host")
        )

    def do_GET(self):
        if not self._is_loopback_request():
            self.send_error(403)
            return
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = _PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        prefix = "/api/events/"
        if parsed.path.startswith(prefix):
            self._stream_events(parsed.path[len(prefix) :])
            return
        self.send_error(404)

    def do_POST(self):
        if not self._is_loopback_request():
            self.send_error(403)
            return
        if urlparse(self.path).path != "/api/derive":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= _MAX_REQUEST_BYTES:
                raise ValueError("request body must be between 1 and 4096 bytes")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            task_name = payload.get("task", "phi4_dim4")
            max_retries = payload.get(
                "max_step_retries", int(os.environ.get("MAX_STEP_RETRIES", "3"))
            )
            if not isinstance(task_name, str) or not _TASK_NAME.fullmatch(task_name):
                raise ValueError("task must be a task name, not a path")
            if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
                raise ValueError("max_step_retries must be a non-negative integer")
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        run_id = uuid.uuid4().hex
        channel = EventChannel()
        with _channels_lock:
            _channels[run_id] = channel
        threading.Thread(
            target=_start_derivation,
            args=(run_id, task_name, max_retries),
            daemon=True,
        ).start()
        body = json.dumps({"run_id": run_id, "events": f"/api/events/{run_id}"}).encode("utf-8")
        self.send_response(202)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream_events(self, run_id: str) -> None:
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            self.send_error(404)
            return
        with _channels_lock:
            channel = _channels.get(run_id)
        if channel is None:
            self.send_error(404)
            return
        try:
            cursor = int(self.headers.get("Last-Event-ID", "0"))
            if cursor < 0:
                raise ValueError
        except ValueError:
            self.send_error(400, "invalid Last-Event-ID")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            while True:
                with channel.condition:
                    if cursor >= len(channel.events) and not channel.done:
                        channel.condition.wait(timeout=15)
                    available = channel.events[cursor:]
                    done = channel.done
                if not available:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                for event in available:
                    cursor = event["id"]
                    data = json.dumps(event, ensure_ascii=False)
                    encoded = f"id: {cursor}\nevent: update\ndata: {data}\n\n".encode("utf-8")
                    self.wfile.write(encoded)
                    self.wfile.flush()
                if done and cursor >= len(channel.events):
                    return
        except (BrokenPipeError, ConnectionResetError):
            return


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("the SSE server only binds to loopback addresses")
    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"Amplitude Lab SSE UI: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Amplitude Lab SSE server.")
    finally:
        server.server_close()


_PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark light">
<title>Amplitude Lab | Derivation workspace</title>
<style>
*{box-sizing:border-box}
:root{font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;color:#e6edf7;
  background:#10151d;color-scheme:dark;--panel:#19212c;--line:#303d4d;--muted:#a9b5c5;
  --accent:#8cc8ff;--green:#76d7a5;--red:#ff9e9e;--yellow:#f6d477}
body{max-width:1180px;margin:0 auto;padding:2rem 1.25rem 4rem}
h1,h2,h3,p{margin-top:0}h1{font-size:clamp(1.7rem,4vw,2.4rem);line-height:1.15}
h2{font-size:1.25rem;margin-bottom:.75rem}h3{font-size:1rem;margin-bottom:.4rem}
a{color:var(--accent)}button,input{font:inherit}
button{background:#256eaa;color:white;border:0;border-radius:.45rem;padding:.65rem 1rem;
  cursor:pointer;font-weight:650}button:hover{background:#3785c3}
button:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid var(--accent);
  outline-offset:2px}
input{background:#0e141b;color:inherit;border:1px solid var(--line);border-radius:.4rem;
  padding:.55rem .65rem}
pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:.55rem 0 0;
  font: .9rem/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}
code{font: .9em ui-monospace,SFMono-Regular,Consolas,monospace}
.markdown{line-height:1.7;overflow-wrap:anywhere}
.markdown p{margin:.5rem 0}.markdown h1,.markdown h2,.markdown h3,.markdown h4{
  margin:.75rem 0 .4rem;line-height:1.3}
.markdown ul,.markdown ol{padding-left:1.4rem;margin:.4rem 0}
.markdown blockquote{margin:.5rem 0;border-left:3px solid var(--line);padding-left:.8rem;
  color:var(--muted)}
.markdown pre{background:#0e141b;border:1px solid var(--line);border-radius:.5rem;padding:.8rem}
.math-inline{white-space:nowrap}.math-display{overflow-x:auto;margin:.8rem 0;text-align:center}
.math-inline math,.math-display math{font-family:"Cambria Math",serif;font-size:1.08em}
.math-display math{font-size:1.2em}
.math-fallback{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
.muted,.hint{color:var(--muted)}.hint{font-size:.9rem}
.topline,.section-heading,.step-heading,.row{display:flex;align-items:center;gap:.75rem}
.topline,.section-heading,.step-heading{justify-content:space-between}
.topline{align-items:flex-start;margin-bottom:1.5rem}
.panel,.step-card{background:var(--panel);border:1px solid var(--line);border-radius:.75rem;
  padding:1.1rem;margin:1rem 0}
.panel p:last-child{margin-bottom:0}
.controls{display:flex;align-items:end;gap:.8rem;flex-wrap:wrap}
.controls label{display:grid;gap:.25rem;font-size:.88rem;color:var(--muted)}
.status{display:inline-flex;align-items:center;gap:.45rem;border:1px solid var(--line);
  border-radius:999px;padding:.3rem .7rem;font-size:.88rem;font-weight:700;white-space:nowrap}
.status:before{content:"";width:.55rem;height:.55rem;border-radius:50%;background:#a9b5c5}
.status[data-state="accepted"],.status[data-state="completed_verified"]{color:var(--green)}
.status[data-state="accepted"]:before,.status[data-state="completed_verified"]:before{background:var(--green)}
.status[data-state="rejected"],.status[data-state="failed"],.status[data-state="retry_exhausted"]{color:var(--red)}
.status[data-state="rejected"]:before,.status[data-state="failed"]:before,
.status[data-state="retry_exhausted"]:before{background:var(--red)}
.status[data-state="awaiting_audit"],.status[data-state="verifying"],
.status[data-state="revising"]{color:var(--yellow)}
.status[data-state="awaiting_audit"]:before,.status[data-state="verifying"]:before,
.status[data-state="revising"]:before{background:var(--yellow)}
.summary-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.75rem}
.metric{background:#111821;border:1px solid var(--line);border-radius:.55rem;padding:.75rem}
.metric strong{display:block;font-size:1.35rem}
.grid{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(260px,.7fr);gap:1rem}
.step-list,.activity-list,.accepted-list,.check-list{list-style:none;padding:0;margin:0}
.step-card{margin:.75rem 0;padding:.9rem}
.step-card[data-state="accepted"]{border-left:4px solid var(--green)}
.step-card[data-state="rejected"],
.step-card[data-state="invalidated"]{border-left:4px solid var(--red)}
.step-card[data-state="generating"],.step-card[data-state="awaiting_audit"],
.step-card[data-state="revising"],.step-card[data-state="verifying"]{
  border-left:4px solid var(--yellow)}
.step-title{font-weight:700}.step-body{margin-top:.65rem}
.dependency{display:inline-block;background:#111821;border-radius:99px;padding:.15rem .5rem;
  margin:.15rem .25rem .15rem 0;font-size:.8rem;color:var(--muted)}
.attempts{margin-top:.75rem;border-top:1px solid var(--line);padding-top:.6rem}
details>summary{cursor:pointer;color:var(--accent)}
.attempt{border-top:1px solid var(--line);margin-top:.7rem;padding-top:.7rem}
.attempt-label{font-weight:650}.diagnostic{border-left:3px solid var(--yellow);padding-left:.65rem}
.accepted-item{padding:.75rem 0;border-top:1px solid var(--line)}
.accepted-item:first-child{border-top:0}
.check{padding:.55rem 0;border-bottom:1px solid var(--line)}
.check:last-child{border-bottom:0}.pass{color:var(--green)}.fail{color:var(--red)}
.activity-list{max-height:24rem;overflow:auto}
.activity-list li{padding:.5rem 0;border-bottom:1px solid var(--line)}
.activity-time{color:var(--muted);font-size:.8rem;margin-right:.45rem}
.final-box{background:#0e141b;border:1px solid var(--line);border-radius:.5rem;padding:1rem}
.empty{color:var(--muted);font-style:italic}
#technical-log{max-height:28rem;overflow:auto;background:#0e141b;border-radius:.5rem;
  padding:.8rem}
.connection{color:var(--muted);font-size:.88rem}
@media(max-width:760px){body{padding:1.2rem .8rem 3rem}.grid{grid-template-columns:1fr}
  .summary-grid{grid-template-columns:1fr 1fr}.topline{display:block}
  .topline .status{margin-top:.5rem}}
</style>
<header class="topline">
  <div><p class="hint">AMPLITUDE LAB / RESEARCH WORKSPACE</p>
    <h1>Autonomous QFT derivation</h1>
    <p class="muted">Follow the plan, derivation, independent AI audit, and
      deterministic checks.</p>
  </div>
  <div><span id="run-status" class="status" data-state="idle" role="status" aria-live="polite">
    Idle</span><p id="connection" class="connection">No active run</p></div>
</header>
<section class="panel" aria-labelledby="run-heading">
  <h2 id="run-heading">Start a derivation</h2>
  <form id="start" class="controls">
    <label>Task name <input name="task" value="phi4_dim4" required maxlength="80"></label>
    <label>Retries per step <input name="max_step_retries" type="number" min="0"
      placeholder="Server default"></label>
    <button type="submit">Start derivation</button>
  </form>
  <p class="hint" style="margin:.75rem 0 0">Models and credentials stay on the server. Select a task
    already defined in the project.</p>
</section>
<section class="panel" aria-label="Run progress">
  <div class="summary-grid">
    <div class="metric"><span class="muted">Accepted steps</span>
      <strong id="accepted-count">0 / 0</strong></div>
    <div class="metric"><span class="muted">Attempts</span>
      <strong id="attempt-count">0</strong></div>
    <div class="metric"><span class="muted">Run ID</span><strong id="run-id">—</strong></div>
  </div>
</section>
<div class="grid">
  <main>
    <section class="panel" aria-labelledby="overview-heading">
      <div class="section-heading">
        <h2 id="overview-heading">Problem interpretation &amp; approach</h2>
        <span id="planning-state" class="hint">Waiting for a run</span></div>
      <div id="planning-preview" class="empty">The derivation agent will summarize
        its interpretation, assumptions, and selected approach here.</div>
      <div id="overview" hidden>
        <h3>Interpretation</h3><div id="interpretation" class="markdown"></div>
        <h3>Selected approach</h3><div id="strategy" class="markdown"></div>
        <div id="assumptions-wrap"><h3>Assumptions</h3><ul id="assumptions"></ul></div>
      </div>
    </section>
    <section class="panel" aria-labelledby="plan-heading">
      <div class="section-heading"><h2 id="plan-heading">Derivation plan</h2>
        <span id="plan-progress" class="hint">No plan yet</span></div>
      <ol id="step-list" class="step-list"><li class="empty">Plan steps will appear here.</li></ol>
    </section>
    <section class="panel" aria-labelledby="accepted-heading">
      <h2 id="accepted-heading">Accepted derivation</h2>
      <p class="hint">Only steps that passed their AI audit appear here. Deterministic checks are
        reported separately below.</p>
      <div id="accepted-list" class="empty">No steps accepted yet.</div>
    </section>
    <section class="panel" aria-labelledby="final-heading">
      <div class="section-heading"><h2 id="final-heading">Final amplitude</h2>
        <button id="copy-final" type="button" disabled>Copy</button></div>
      <div id="final-amplitude" class="final-box empty">The final result will appear
        when the run finishes.</div>
    </section>
    <section class="panel" aria-labelledby="verification-heading">
      <h2 id="verification-heading">Deterministic verification</h2>
      <p class="hint">This is independent of the AI auditor. Unsupported tasks are
        explicitly shown as not verified.</p>
      <div id="verification" class="empty">No deterministic verification has run yet.</div>
    </section>
  </main>
  <aside>
    <section class="panel" aria-labelledby="history-heading">
      <h2 id="history-heading">Audit &amp; backtracking history</h2>
      <ul id="activity" class="activity-list">
        <li class="empty">Run events will appear here.</li></ul>
    </section>
  </aside>
</div>
<details class="panel">
  <summary>Technical event log (raw SSE)</summary>
  <p class="hint">For debugging; the readable run view above is usually easier to follow.</p>
  <pre id="technical-log"></pre>
</details>
<script>
const form=document.querySelector("#start");
const statusEl=document.querySelector("#run-status");
const connection=document.querySelector("#connection");
const stepList=document.querySelector("#step-list");
const activity=document.querySelector("#activity");
const technicalLog=document.querySelector("#technical-log");
const terminalStates=new Set([
  "completed_verified","completed_unverified","failed","retry_exhausted"
]);
let source=null, plan=[], steps=new Map(), accepted=new Map();
let planTokens="", runStatus="idle";
const text=(tag,value,className="")=>{
  const node=document.createElement(tag); if(className) node.className=className;
  node.textContent=value||""; return node;
};
const MATH_NS="http://www.w3.org/1998/Math/MathML";
const greek={alpha:"α",beta:"β",gamma:"γ",delta:"δ",epsilon:"ϵ",varepsilon:"ε",
  theta:"θ",lambda:"λ",mu:"μ",nu:"ν",pi:"π",rho:"ρ",sigma:"σ",tau:"τ",phi:"ϕ",
  varphi:"φ",chi:"χ",psi:"ψ",omega:"ω",Gamma:"Γ",Delta:"Δ",Theta:"Θ",Lambda:"Λ",
  Pi:"Π",Sigma:"Σ",Phi:"Φ",Psi:"Ψ",Omega:"Ω"};
const mathSymbols={to:"→",rightarrow:"→",leftarrow:"←",Rightarrow:"⇒",leftrightarrow:"↔",
  cdot:"·",times:"×",pm:"±",infty:"∞",partial:"∂",nabla:"∇",leq:"≤",geq:"≥",
  neq:"≠",approx:"≈",sum:"∑",int:"∫",prod:"∏",ell:"ℓ",hbar:"ℏ",quad:" "};
function mathNode(name,children=[]){
  const node=document.createElementNS(MATH_NS,name);
  for(const child of children)node.append(child);
  return node;
}
function parseMath(tex,display=false){
  const tokens=tex.match(/\\\\[a-zA-Z]+|\\\\.|[a-zA-Z]+|[0-9]+|[_^{}]|./gs)||[];
  let cursor=0;
  const atom=(name,value)=>{const node=mathNode(name);node.textContent=value;return node;};
  function argument(){
    if(tokens[cursor]==="{"){cursor++;return mathNode("mrow",sequence(true));}
    return nextAtom();
  }
  function sequence(stop=false){
    const nodes=[];
    while(cursor<tokens.length){
      const token=tokens[cursor++];
      if(token==="}"&&stop)break;
      if(token==="{" ){nodes.push(mathNode("mrow",sequence(true)));continue;}
      if(token==="_"||token==="^"){
        const base=nodes.pop();
        if(!base){nodes.push(atom("mo",token));continue;}
        const arg=argument();
        const sub=token==="_"?arg:(base.localName==="msub"?base.children[1]:null);
        const sup=token==="^"?arg:(base.localName==="msup"?base.children[1]:null);
        const original=(base.localName==="msub"||base.localName==="msup")?
          base.children[0]:base;
        if(sub&&sup)nodes.push(mathNode("msubsup",[original,sub,sup]));
        else nodes.push(mathNode(sub?"msub":"msup",[original,sub||sup]));
        continue;
      }
      if(token.startsWith("\\\\")){
        const command=token.slice(1);
        if(["left","right","displaystyle","textstyle","limits","!"].includes(command))continue;
        if(command==="frac"||command==="dfrac"||command==="tfrac"){
          nodes.push(mathNode("mfrac",[argument(),argument()]));continue;
        }
        if(command==="sqrt"){nodes.push(mathNode("msqrt",[argument()]));continue;}
        if(command==="overline"||command==="bar"){
          nodes.push(mathNode("mover",[argument(),atom("mo","¯")]));
          continue;
        }
        if(command==="underline"){
          nodes.push(mathNode("munder",[argument(),atom("mo","_")]));continue;
        }
        if(["mathcal","mathbb","mathrm","mathbf","mathit","operatorname"].includes(command)){
          const value=argument();
          const variant={mathcal:"script",mathbb:"double-struck",mathrm:"normal",
            mathit:"italic",mathbf:"bold",operatorname:"normal"}[command];
          const style=mathNode("mstyle",[value]);style.setAttribute("mathvariant",variant);
          nodes.push(style);continue;
        }
        if(command==="text"){
          const value=argument();const label=atom("mtext",value.textContent);
          nodes.push(label);continue;
        }
        if(greek[command]){nodes.push(atom("mi",greek[command]));continue;}
        if(mathSymbols[command]){nodes.push(atom("mo",mathSymbols[command]));continue;}
        if(command===","||command===";"||command===" "){
          const space=mathNode("mspace");space.setAttribute("width","0.25em");
          nodes.push(space);continue;
        }
        nodes.push(atom("mtext",command));continue;
      }
      if(/^[0-9]+$/.test(token)){nodes.push(atom("mn",token));continue;}
      if(/^[a-zA-Z]+$/.test(token)){nodes.push(atom("mi",token));continue;}
      if(token===" "){const space=mathNode("mspace");space.setAttribute("width","0.25em");
        nodes.push(space);continue;}
      nodes.push(atom("mo",token));
    }
    return nodes;
  }
  function nextAtom(){
    if(cursor>=tokens.length)return atom("mrow","");
    const token=tokens[cursor++];
    if(token==="{")return mathNode("mrow",sequence(true));
    if(token.startsWith("\\\\")){
      const command=token.slice(1);
      if(command==="frac"||command==="dfrac"||command==="tfrac")
        return mathNode("mfrac",[argument(),argument()]);
      if(command==="sqrt")return mathNode("msqrt",[argument()]);
      if(command==="overline"||command==="bar")
        return mathNode("mover",[argument(),atom("mo","¯")]);
      if(command==="text")return atom("mtext",argument().textContent);
      if(greek[command])return atom("mi",greek[command]);
      if(mathSymbols[command])return atom("mo",mathSymbols[command]);
      return atom("mtext",command);
    }
    if(/^[0-9]+$/.test(token))return atom("mn",token);
    if(/^[a-zA-Z]+$/.test(token))return atom("mi",token);
    return atom("mo",token);
  }
  try{
    const root=mathNode("math",sequence());
    root.setAttribute("display",display?"block":"inline");
    root.setAttribute("aria-label",tex);
    return root;
  }catch(_error){
    const fallback=text("span",tex,"math-fallback");
    fallback.setAttribute("aria-label","Math expression: "+tex);
    return fallback;
  }
}
function appendMath(target,tex,display=false){
  const wrapper=text("span","","");
  wrapper.className=display?"math-display":"math-inline";
  wrapper.append(parseMath(tex,display));target.append(wrapper);
}
function appendInline(target,value){
  let index=0,plain="";
  const flush=()=>{if(plain){target.append(document.createTextNode(plain));plain="";}};
  while(index<value.length){
    if(value.startsWith("$$",index)||value.startsWith("\\\\[",index)||
      value.startsWith("\\\\(",index)){
      const delimiter=value.startsWith("$$",index)?"$$":
        value.startsWith("\\\\[",index)?"\\\\[":"\\\\(";
      const closing=delimiter==="$$"?"$$":delimiter==="\\\\["?"\\\\]":"\\\\)";
      const end=value.indexOf(closing,index+delimiter.length);
      if(end>=0){flush();appendMath(target,value.slice(index+delimiter.length,end));
        index=end+closing.length;continue;}
    }
    if(value[index]==="$"&&value[index+1]!=="$"){
      const end=value.indexOf("$",index+1);
      if(end>index+1){flush();appendMath(target,value.slice(index+1,end));
        index=end+1;continue;}
    }
    if(value.startsWith("**",index)||value[index]==="*"||value[index]==="`"){
      const marker=value.startsWith("**",index)?"**":value[index];
      const end=value.indexOf(marker,index+marker.length);
      if(end>index+marker.length){
        flush();
        const node=text(marker==="`"?"code":marker==="**"?"strong":"em","");
        if(marker==="`")node.textContent=value.slice(index+marker.length,end);
        else appendInline(node,value.slice(index+marker.length,end));
        target.append(node);index=end+marker.length;continue;
      }
    }
    if(value[index]==="\\\\"&&/[A-Za-z]/.test(value[index+1]||"")){
      flush();let end=index+1;
      while(end<value.length&&!/[\\s,.;:!?]/.test(value[end]))end++;
      let expression=value.slice(index,end),punctuation="";
      while(/[.,;:!?]$/.test(expression)){
        punctuation=expression.slice(-1)+punctuation;expression=expression.slice(0,-1);
      }
      appendMath(target,expression);if(punctuation)plain+=punctuation;
      index=end;continue;
    }
    if(value[index]==="\\n"){flush();target.append(document.createElement("br"));index++;continue;}
    plain+=value[index++];
  }
  flush();
}
function renderMarkdown(target,source){
  target.replaceChildren();target.classList.add("markdown");
  const lines=String(source||"").replace(/\\r/g,"").split("\\n");
  let index=0;
  while(index<lines.length){
    const line=lines[index];
    if(!line.trim()){index++;continue;}
    const fence=line.match(/^\\s*```([\\w-]*)\\s*$/);
    if(fence){
      const code=[];index++;
      while(index<lines.length&&!/^\\s*```\\s*$/.test(lines[index]))code.push(lines[index++]);
      if(index<lines.length)index++;
      if(fence[1]==="final"){
        const final=text("div","");final.className="final-markdown";renderMarkdown(final,code.join("\\n"));
        target.append(final);
      }else{
        const pre=text("pre","");pre.append(text("code",code.join("\\n")));target.append(pre);
      }
      continue;
    }
    if(line.startsWith("$$")||line.startsWith("\\\\[")){
      const delimiter=line.startsWith("$$")?"$$":"\\\\[";
      const closing=delimiter==="$$"?"$$":"\\\\]";
      let expression=line.slice(delimiter.length),end=expression.indexOf(closing);
      if(end>=0){expression=expression.slice(0,end);index++;}
      else{
        index++;
        while(index<lines.length&&!lines[index].includes(closing))expression+="\\n"+lines[index++];
        if(index<lines.length){const tail=lines[index].indexOf(closing);
          expression+="\\n"+lines[index].slice(0,tail);index++;}
      }
      appendMath(target,expression,true);continue;
    }
    const heading=line.match(/^(#{1,6})\\s+(.+)$/);
    if(heading){
      const node=text("h"+heading[1].length,"");appendInline(node,heading[2]);
      target.append(node);index++;continue;
    }
    const listMatch=line.match(/^\\s*([-*]|\\d+\\.)\\s+(.+)$/);
    if(listMatch){
      const ordered=/\\d/.test(listMatch[1]),list=document.createElement(ordered?"ol":"ul");
      while(index<lines.length){
        const item=lines[index].match(/^\\s*([-*]|\\d+\\.)\\s+(.+)$/);
        if(!item||/\\d/.test(item[1])!==ordered)break;
        const row=document.createElement("li");appendInline(row,item[2]);list.append(row);index++;
      }
      target.append(list);continue;
    }
    if(line.startsWith(">")){
      const quote=text("blockquote","");
      while(index<lines.length&&lines[index].startsWith(">")){
        const para=text("p","");appendInline(para,lines[index++].replace(/^>\\s?/,""));
        quote.append(para);
      }
      target.append(quote);continue;
    }
    const paragraph=[line];index++;
    while(index<lines.length&&lines[index].trim()&&
      !/^\\s*```/.test(lines[index])&&!/^#{1,6}\\s/.test(lines[index])&&
      !/^\\s*([-*]|\\d+\\.)\\s+/.test(lines[index])&&!lines[index].startsWith("$$")&&
      !lines[index].startsWith("\\\\["))paragraph.push(lines[index++]);
    const node=text("p","");appendInline(node,paragraph.join("\\n"));target.append(node);
  }
}
const stateLabels={idle:"Idle",planning:"Interpreting problem",generating:"Deriving step",
  awaiting_audit:"Awaiting AI audit",accepted:"Accepted by AI audit",rejected:"Rejected",
  revising:"Revising / backtracking",verifying:"Running deterministic checks",
  completed_verified:"Completed — deterministically verified",
  completed_unverified:"Completed — not deterministically verified",
  retry_exhausted:"Stopped — retry limit reached",failed:"Run failed"};
function setStatus(state){
  runStatus=state||"generating"; statusEl.dataset.state=runStatus;
  statusEl.textContent=stateLabels[runStatus]||runStatus.replaceAll("_"," ");
}
function ensureStep(id){
  if(!steps.has(id)) steps.set(id,{id,state:"pending",goal:id,depends_on:[],attempts:[]});
  return steps.get(id);
}
function currentAttempt(step){
  if(!step.attempts.length) step.attempts.push({number:step.attempts.length+1,proposal:"",
    auditText:"",verdict:"",diagnostic:"",state:"generating"});
  return step.attempts[step.attempts.length-1];
}
function renderPlan(){
  stepList.replaceChildren();
  const ordered=plan.length?plan:Array.from(steps.values());
  if(!ordered.length){stepList.append(text("li","Plan steps will appear here.","empty"));return;}
  let acceptedCount=0,attemptCount=0;
  for(const item of ordered){
    const step=steps.get(item.id)||ensureStep(item.id);
    if(step.state==="accepted") acceptedCount++;
    attemptCount+=step.attempts.length;
    const card=text("li",""); card.className="step-card"; card.dataset.state=step.state;
    const heading=text("div",""); heading.className="step-heading";
    heading.append(text("span",item.goal||step.goal||item.id,"step-title"));
    const badge=text("span",stateLabels[step.state]||step.state.replaceAll("_"," "));
    badge.className="status"; badge.dataset.state=step.state;
    heading.append(badge); card.append(heading);
    const id=text("div","Step "+item.id+" · "+(step.attempts.length||0)+" attempt(s)","hint");
    card.append(id);
    if((item.depends_on||[]).length){
      const deps=text("div","Depends on: ","hint");
      for(const dep of item.depends_on) deps.append(text("span",dep,"dependency"));
      card.append(deps);
    }
    const latest=step.attempts[step.attempts.length-1];
    if(latest&&latest.proposal&&step.state!=="accepted"){
      const live=text("div",""); live.className="step-body";
      live.append(text("strong",latest.state==="generating"?
        "Live derivation":"Proposed derivation"));
      live.append(text("pre",latest.proposal)); card.append(live);
    }
    if(latest&&latest.auditText&&latest.state==="awaiting_audit"){
      const audit=text("div",""); audit.className="step-body";
      audit.append(text("strong","Audit response in progress"));
      audit.append(text("pre",latest.auditText));
      card.append(audit);
    }
    if(latest&&latest.diagnostic){
      const diagnostic=text("p","Audit diagnostic: "+latest.diagnostic,"diagnostic");
      card.append(diagnostic);
    }
    if(step.attempts.length){
      const details=document.createElement("details"); details.className="attempts";
      details.append(text("summary","Show attempt and audit history ("+step.attempts.length+")"));
      for(const attempt of step.attempts){
        const row=text("div",""); row.className="attempt";
        row.append(text("div","Attempt "+attempt.number+": "+
          (stateLabels[attempt.state]||attempt.state.replaceAll("_"," ")),"attempt-label"));
        if(attempt.verdict) row.append(text("p","AI auditor: "+attempt.verdict));
        if(attempt.diagnostic) row.append(text("p",attempt.diagnostic,"diagnostic"));
        if(attempt.proposal){row.append(text("strong","Proposal"));row.append(text("pre",attempt.proposal));}
        if(attempt.auditText){
          row.append(text("strong","Auditor response"));
          row.append(text("pre",attempt.auditText));
        }
        details.append(row);
      }
      card.append(details);
    }
    stepList.append(card);
  }
  document.querySelector("#accepted-count").textContent=acceptedCount+" / "+plan.length;
  document.querySelector("#attempt-count").textContent=String(attemptCount);
  document.querySelector("#plan-progress").textContent=plan.length?
    acceptedCount+" of "+plan.length+" steps accepted":"No plan yet";
}
function renderAccepted(){
  const target=document.querySelector("#accepted-list"); target.replaceChildren();
  if(!accepted.size){target.className="empty";target.textContent="No steps accepted yet.";return;}
  target.className="accepted-list";
  for(const item of (plan.length?plan:Array.from(accepted.values()))){
    const row=accepted.get(item.id); if(!row) continue;
    const section=text("section",""); section.className="accepted-item";
    section.append(text("h3",item.id+": "+(item.goal||row.goal)));
    section.append(text("pre",row.content));
    const details=document.createElement("details");
    details.append(text("summary","AI audit record"));
    details.append(text("pre",row.audit||"No complete audit response recorded."));
    section.append(details); target.append(section);
  }
}
function renderVerification(report){
  const target=document.querySelector("#verification"); target.replaceChildren();
  target.className="";
  if(!report||report.available===false){
    target.className="empty";
    target.textContent=(report&&report.message)||"No deterministic verification has run yet.";
    return;
  }
  const heading=text("p",report.passed===true?"All configured deterministic checks passed.":
    report.passed===false?"One or more deterministic checks failed.":
    "Verification result unavailable.",
    report.passed===true?"pass":report.passed===false?"fail":"");
  target.append(heading);
  const checks=report.checks&&report.checks.checks;
  if(Array.isArray(checks)){
    const list=text("ul","");list.className="check-list";
    for(const check of checks){
      const row=text("li",""); row.className="check";
      row.append(text("strong",(check.passed?"PASS · ":"FAIL · ")+check.name,
        check.passed?"pass":"fail"));
      row.append(text("div",check.message,"hint"));list.append(row);
    }
    target.append(list);
  }else target.append(text("pre",JSON.stringify(report,null,2)));
}
function addActivity(item,message){
  if(activity.children.length===1&&activity.firstElementChild.classList.contains("empty"))
    activity.replaceChildren();
  const row=text("li",""); row.append(text("span",item.time||"live","activity-time"));
  row.append(document.createTextNode(message)); activity.prepend(row);
}
function describe(item){
  const name=item.step_id||item.target_step_id||"";
  switch(item.type){
    case "plan.started": return "Planner is interpreting the problem.";
    case "plan.ready": return "Plan ready: "+item.steps.length+" dependency-ordered steps.";
    case "step.generating": return "Derivation agent started "+name+" (attempt "+item.attempt+").";
    case "step.awaiting_audit": return "Step "+name+" is awaiting the independent AI audit.";
    case "step.audit": return "AI audit "+item.verdict.toLowerCase()+" for "+name+
      (item.diagnostic?": "+item.diagnostic:"");
    case "step.accepted": return "Step "+name+" passed its AI audit and was checkpointed.";
    case "step.rejected":
      return (item.deterministic_failure?"Deterministic check rejected ":"AI audit rejected ")+
        name+": "+(item.diagnostic||"");
    case "step.revising": return "Derivation agent is revising "+name+".";
    case "backtracking.started": return "Backtracking to "+item.target_step_id+
      "; invalidated: "+(item.invalidated_steps||[]).join(", ")+". "+item.diagnostic;
    case "verification.started": return "Running deterministic verification.";
    case "verification.finished": return "Deterministic checks "+(item.passed?"passed.":"failed.");
    case "derivation.completed": return "Run completed: "+
      (stateLabels[item.state]||item.state)+"."+(item.error?" "+item.error:"");
    case "derivation.failed": case "server.failed": return "Run failed: "+item.error;
    default: return item.message||item.type.replaceAll("."," ");
  }
}
function handleEvent(item){
  if(item.state) setStatus(item.state);
  technicalLog.textContent+=JSON.stringify(item)+"\\n";
  const finished=terminalStates.has(item.state);
  if(item.type==="token"){
    if(item.state==="planning"){
      planTokens+=item.text||"";
      document.querySelector("#planning-preview").textContent=planTokens;
    }else if(item.step_id){
      const step=ensureStep(item.step_id), attempt=currentAttempt(step);
      if(item.agent==="audit") attempt.auditText+=item.text||"";
      else attempt.proposal+=item.text||"";
      renderPlan();
    }
  }else{
    if(item.type==="plan.ready"){
      plan=item.steps||[];
      steps=new Map(plan.map(row=>[row.id,{...row,state:"pending",attempts:[]}]));
      document.querySelector("#interpretation").textContent=item.interpretation||"Not supplied.";
      document.querySelector("#strategy").textContent=item.strategy||"Not supplied.";
      const assumptions=document.querySelector("#assumptions");
      assumptions.replaceChildren();
      for(const value of (item.assumptions||[])) assumptions.append(text("li",value));
      document.querySelector("#assumptions-wrap").hidden=!(item.assumptions||[]).length;
      document.querySelector("#overview").hidden=false;
      document.querySelector("#planning-preview").hidden=true;
      document.querySelector("#planning-state").textContent="Plan ready";
    }else if(item.type==="plan.started"){
      document.querySelector("#planning-state").textContent="Interpreting problem";
    }else if(item.step_id&&item.type==="step.generating"){
      const step=ensureStep(item.step_id); step.state="generating";
      step.goal=item.goal||step.goal;
      step.attempts.push({number:item.attempt,proposal:"",auditText:"",verdict:"",
        diagnostic:"",state:"generating"});
    }else if(item.step_id&&item.type==="step.awaiting_audit"){
      const step=ensureStep(item.step_id);step.state="awaiting_audit";
      currentAttempt(step).state="awaiting_audit";
    }else if(item.type==="step.audit"){
      const step=ensureStep(item.step_id), attempt=currentAttempt(step);
      attempt.verdict=item.verdict;attempt.diagnostic=item.diagnostic;attempt.audit=item.response;
      attempt.state=item.verdict==="ACCEPT"?"accepted":"rejected";
      step.state=item.verdict==="ACCEPT"?"accepted":"rejected";
    }else if(item.type==="step.accepted"){
      const step=ensureStep(item.step_id);step.state="accepted";
      const attempt=currentAttempt(step);attempt.state="accepted";
      accepted.set(item.step_id,{id:item.step_id,goal:step.goal,content:item.content,audit:item.audit});
      renderAccepted();
    }else if(item.type==="step.rejected"){
      const step=ensureStep(item.step_id);step.state="rejected";
      const attempt=currentAttempt(step);attempt.state="rejected";
      attempt.diagnostic=item.diagnostic||attempt.diagnostic;
      accepted.delete(item.step_id);renderAccepted();
    }else if(item.type==="step.revising"){
      ensureStep(item.step_id).state="revising";
    }else if(item.type==="backtracking.started"){
      for(const id of (item.invalidated_steps||[])){
        const step=ensureStep(id);step.state="invalidated";accepted.delete(id);
      }
      ensureStep(item.target_step_id).state="revising";renderAccepted();
    }else if(item.type==="verification.finished"){
      renderVerification({available:true,passed:item.passed,checks:item.checks});
    }else if(item.type==="derivation.completed"||item.type==="run.done"){
      const result=item.result||item;
      if(result.final_amplitude){
        const output=document.querySelector("#final-amplitude");
        output.className="final-box";output.textContent=result.final_amplitude;
        document.querySelector("#copy-final").disabled=false;
      }else if(result.error){
        document.querySelector("#final-amplitude").textContent=
          "No final amplitude was accepted. "+result.error;
        document.querySelector("#final-amplitude").className="final-box";
      }
      if(result.verification) renderVerification(result.verification);
      if(Array.isArray(result.accepted_steps)){
        accepted=new Map(result.accepted_steps.map(row=>[row.step_id,{id:row.step_id,...row}]));
        renderAccepted();
      }
    }else if(item.type==="derivation.failed"||item.type==="server.failed"){
      document.querySelector("#final-amplitude").textContent=item.error||"Run failed.";
    }
    if(item.type!=="plan.ready"&&
      !(item.type==="step.rejected"&&!item.deterministic_failure))
      addActivity(item,describe(item));
    renderPlan();
  }
  if(finished){connection.textContent="Run finished";if(source)source.close();}
}
document.querySelector("#copy-final").addEventListener("click",async()=>{
  try{
    await navigator.clipboard.writeText(document.querySelector("#final-amplitude").textContent);
    connection.textContent="Final amplitude copied";
  }catch(error){connection.textContent="Copy failed: "+error.message;}
});
form.addEventListener("submit",async event=>{
  event.preventDefault();if(source)source.close();
  plan=[];steps=new Map();accepted=new Map();planTokens="";
  activity.replaceChildren(text("li","Run events will appear here.","empty"));
  technicalLog.textContent="";document.querySelector("#overview").hidden=true;
  document.querySelector("#planning-preview").hidden=false;
  document.querySelector("#planning-preview").textContent="Waiting for planner output…";
  document.querySelector("#planning-state").textContent="Starting";
  document.querySelector("#accepted-list").textContent="No steps accepted yet.";
  document.querySelector("#accepted-list").className="empty";
  document.querySelector("#final-amplitude").textContent=
    "The final result will appear when the run finishes.";
  document.querySelector("#final-amplitude").className="final-box empty";
  document.querySelector("#copy-final").disabled=true;
  renderVerification(null);renderPlan();setStatus("planning");
  connection.textContent="Starting run…";
  const body=Object.fromEntries(new FormData(form));
  if(body.max_step_retries==="")delete body.max_step_retries;
  else body.max_step_retries=Number(body.max_step_retries);
  try{
    const response=await fetch("/api/derive",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const result=await response.json();
    if(!response.ok)throw new Error(result.error||"Could not start derivation.");
    document.querySelector("#run-id").textContent=result.run_id;
    connection.textContent="Connecting to live event stream";
    document.querySelector("#planning-state").textContent="Planner starting";
    source=new EventSource(result.events);
    source.addEventListener("open",()=>{connection.textContent="Live updates connected";});
    source.addEventListener("update",event=>handleEvent(JSON.parse(event.data)));
    source.onerror=()=>{
      if(source.readyState===EventSource.CLOSED)connection.textContent="Event stream closed";
      else connection.textContent="Connection interrupted — attempting to reconnect…";
    };
  }catch(error){
    setStatus("failed");connection.textContent="Unable to start run";
    addActivity({},"Could not start derivation: "+error.message);
  }
});
</script>
"""
