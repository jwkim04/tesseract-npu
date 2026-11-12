/* report.js: the profile report window. Reads the data block, draws the timeline on a canvas, fills the tree and the tabs. Selection
 * (one record, optionally one op) is shared by the tree, the timeline, the tables and the Details tab. No network access. */
(function () {
  "use strict";
  var D = JSON.parse(document.getElementById("data").textContent);
  var M = D.meta, R = D.records, S = D.summary, F = M.f;
  var RUNS = D.tl.runs, SPAN = M.span;
  var GL = {};
  D.groups.forEach(function (g) { GL[g.id] = g.label; });
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  // ---------------------------------------------------------------- formatting
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function grp3(s) { return s.replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  function fint(n) { if (n == null || !isFinite(n)) return "-"; var v = Math.round(n); return (v < 0 ? "-" : "") + grp3(String(Math.abs(v))); }
  function fx(n, d) { if (n == null || !isFinite(n)) return "-"; var p = Math.abs(n).toFixed(d).split("."); p[0] = grp3(p[0]); return (n < 0 ? "-" : "") + p.join("."); }
  function fpct(x, d) { return x == null || !isFinite(x) ? "-" : fx(100 * x, d == null ? 1 : d); }
  function fus(c) { return c / F * 1e6; }
  function ftime(us, step) {
    if (Math.abs(us) >= 1000) return fx(us / 1000, step >= 100 ? 1 : step >= 10 ? 2 : 3) + " ms";
    return fx(us, step < 0.1 ? 2 : step < 1 ? 1 : 0) + " µs";
  }
  function div(a, b) { return b ? a / b : null; }

  // ---------------------------------------------------------------- theme
  var root = document.documentElement;
  function themeNow() { var t = root.getAttribute("data-theme"); return t || (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"); }
  try { var saved = localStorage.getItem("tsr-prof-theme"); if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved); } catch (e) { /* no storage */ }
  function themeLabel() { $("#theme").textContent = "Theme: " + (themeNow() === "dark" ? "Dark" : "Light"); }
  $("#theme").addEventListener("click", function () {
    var n = themeNow() === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", n);
    try { localStorage.setItem("tsr-prof-theme", n); } catch (e) { /* no storage */ }
    themeLabel(); readColors(); drawAll();
  });
  if (window.matchMedia) { try { matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () { themeLabel(); readColors(); drawAll(); }); } catch (e) { /* old browser */ } }

  var C = {};
  function readColors() {
    var cs = getComputedStyle(root);
    ["bg", "bg-head", "rule", "rule-strong", "fg", "fg2", "fg3", "accent", "selected", "g-none"].forEach(function (k) { C[k] = cs.getPropertyValue("--" + k).trim(); });
    C.g = {};
    D.groups.forEach(function (g) { C.g[g.id] = cs.getPropertyValue("--g-" + g.id).trim() || C["g-none"]; });
    C.font = cs.getPropertyValue("--font-ui").trim();
    C.mono = cs.getPropertyValue("--font-num").trim();
  }

  // ---------------------------------------------------------------- state
  var st = { sel: -1, op: null, tab: "summary", recView: "kind", sort: { kind: ["cycles", -1], record: ["cycles", -1], node: ["cost", -1] }, recFilter: "", recGroup: "",
             graphView: "splits", nodeFilter: "", nodePlace: "", expanded: {}, treeFilter: "", collapsed: {}, treeSel: "" };
  var recKey = ["i"].concat([]);
  // per record: the card's and the simulator's ops and the index of its op in each timeline row
  var recIdx = RUNS.map(function () { return {}; });
  RUNS.forEach(function (run, ri) {
    run.rows.forEach(function (row, rk) {
      row.ops.forEach(function (o, oi) { var r = o[2]; (recIdx[ri][r] = recIdx[ri][r] || []).push([rk, oi]); });
    });
  });
  var byLayer = {}, layerCost = {};
  R.forEach(function (r) { (byLayer[r.layer] = byLayer[r.layer] || []).push(r.i); layerCost[r.layer] = (layerCost[r.layer] || 0) + r.cost; });
  var kindOf = {};
  D.kinds.forEach(function (k) { kindOf[k.kind] = k; });

  function layerName(l) { return l === "in" ? "In" : l === "out" ? "Out" : "Layer " + l; }
  function recLabel(r) { return r.kn + " " + r.name; }
  function recColor(i) { return i >= 0 && R[i] ? C.g[R[i].group] || C["g-none"] : C["g-none"]; }
  function gcls(g) { return "sw g-" + g; }
  function opName(opc) { return D.opnames[String(opc)] || ("op" + opc.toString(16)); }
  function opId(tag, rec, uname, ord) { return tag + ":" + (rec >= 0 ? "r" + rec : "setup") + "." + uname + ord; }

  // ---------------------------------------------------------------- tables
  function table(heads, rows, o) {
    o = o || {};
    var h = "<thead><tr>" + heads.map(function (c) { return "<th" + (c[1] ? ' class="' + c[1] + '"' : "") + (c[2] ? ' data-k="' + c[2] + '"' : "") + ">" + c[0] + "</th>"; }).join("") + "</tr></thead>";
    var b = rows.map(function (r) {
      var cells = r.cells || r, cls = r.cls ? ' class="' + r.cls + '"' : "", at = r.attr || "";
      return "<tr" + cls + at + ">" + cells.map(function (c, k) { var hc = heads[k] ? heads[k][1] : ""; return "<td" + (hc ? ' class="' + hc.replace("s", "").trim() + '"' : "") + ">" + c + "</td>"; }).join("") + "</tr>";
    }).join("");
    return "<table>" + h + "<tbody>" + b + "</tbody></table>";
  }
  function section(title, body) { return '<div class="sec"><h4>' + esc(title) + "</h4>" + body + "</div>"; }
  var H3 = [["Metric", ""], ["Unit", "u"], ["Value", "n"]];
  function mrow(name, unit, val) { return [esc(name), esc(unit || ""), val]; }

  // ---------------------------------------------------------------- the frame
  var app = $("#app");
  var tlscroll = $("#tlscroll"), cvR = $("#rows"), cvT = $("#ruler"), labels = $("#labels"), tip = $("#tip");
  var geo = [], totalH = 0, view = { a: 0, b: SPAN }, W = 0;
  var ROW = 20, HEAD = 20, RULER = 34;

  function buildGeo() {
    geo = []; var y = 0;
    RUNS.forEach(function (run, ri) {
      geo.push({ type: "head", run: ri, y: y, h: HEAD }); y += HEAD;
      if (!st.collapsed[run.tag]) run.rows.forEach(function (row, rk) { geo.push({ type: "row", run: ri, row: rk, y: y, h: ROW }); y += ROW; });
    });
    totalH = y;
    var h = "";
    geo.forEach(function (g) {
      if (g.type === "head") h += '<div class="lab head" data-run="' + g.run + '" style="top:' + g.y + 'px"><span class="ch">' + (st.collapsed[RUNS[g.run].tag] ? "+" : "-") + "</span>" + esc(RUNS[g.run].label) + "</div>";
      else h += '<div class="lab row" style="top:' + g.y + 'px" title="' + esc(RUNS[g.run].rows[g.row].id) + '">' + esc(RUNS[g.run].rows[g.row].label) + "</div>";
    });
    labels.innerHTML = h;
    labels.style.height = totalH + "px";
    sizeCanvas();
  }
  labels.addEventListener("click", function (e) {
    var t = e.target.closest(".lab.head"); if (!t) return;
    var tag = RUNS[+t.getAttribute("data-run")].tag; st.collapsed[tag] = !st.collapsed[tag]; buildGeo(); draw();
  });
  $("#corner").innerHTML = '<div style="height:17px;padding:0 8px;line-height:17px;font-size:11px;color:var(--fg3)">Cycle</div><div style="height:17px;padding:0 8px;line-height:17px;font-size:11px;color:var(--fg3)">Time</div>';

  var dpr = 1;
  function sizeCanvas() {
    dpr = window.devicePixelRatio || 1;
    W = Math.max($("#rowsw").clientWidth, 100);
    $("#rowsw").style.height = totalH + "px";
    cvR.style.height = totalH + "px";
    cvR.width = Math.round(W * dpr); cvR.height = Math.max(Math.round(totalH * dpr), 1);
    cvT.width = Math.round(W * dpr); cvT.height = Math.round(RULER * dpr);
  }
  function clampView() {
    var w = view.b - view.a;
    if (w > SPAN) w = SPAN;
    if (w < 40) w = 40;
    if (view.a < 0) view.a = 0;
    if (view.a + w > SPAN) view.a = SPAN - w;
    view.b = view.a + w;
  }
  function xOf(t) { return (t - view.a) * W / (view.b - view.a); }
  function tOf(x) { return view.a + x * (view.b - view.a) / W; }
  function niceStep(raw) { var p = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10)); var m = [1, 2, 5, 10]; for (var i = 0; i < m.length; i++) if (m[i] * p >= raw) return m[i] * p; return 10 * p; }

  var cursor = { x: -1, y: -1 }, raf = 0;
  function draw() { if (!raf) raf = requestAnimationFrame(function () { raf = 0; drawNow(); }); }
  function drawAll() { sizeCanvas(); drawNow(); }
  function drawNow() {
    if (!C.bg) readColors();
    clampView();
    var vw = view.b - view.a, step = niceStep(vw * 90 / W);
    // ruler
    var t = cvT.getContext("2d"); t.setTransform(dpr, 0, 0, dpr, 0, 0);
    t.fillStyle = C["bg-head"]; t.fillRect(0, 0, W, RULER);
    t.font = "11px " + C.mono; t.textBaseline = "middle";
    var sel = st.sel >= 0 ? R[st.sel] : null;
    if (sel && sel.first != null) { var sx = xOf(sel.first), ex = xOf(sel.last); t.fillStyle = C.selected; t.fillRect(sx, 0, Math.max(ex - sx, 2), RULER); }
    t.strokeStyle = C["rule-strong"]; t.fillStyle = C.fg2; t.lineWidth = 1;
    var us0 = fus(step);
    var first = Math.ceil(view.a / step) * step, ticks = [];
    for (var v = first; v <= view.b + 1; v += step) ticks.push(v);
    t.beginPath();
    ticks.forEach(function (v) { var x = Math.round(xOf(v)) + 0.5; t.moveTo(x, RULER - 6); t.lineTo(x, RULER); });
    t.stroke();
    ticks.forEach(function (v) { var x = xOf(v); t.fillStyle = C.fg2; t.fillText(fint(v), x + 3, 8); t.fillStyle = C.fg3; t.fillText(ftime(fus(v), us0), x + 3, 25); });
    // rows
    var c = cvR.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.fillStyle = C.bg; c.fillRect(0, 0, W, totalH);
    c.strokeStyle = C.rule; c.globalAlpha = 0.55; c.beginPath();
    ticks.forEach(function (v) { var x = Math.round(xOf(v)) + 0.5; c.moveTo(x, 0); c.lineTo(x, totalH); });
    c.stroke(); c.globalAlpha = 1;
    if (sel && sel.first != null) {
      var bx = xOf(sel.first), bw = Math.max(xOf(sel.last) - bx, 2);
      c.globalAlpha = 0.5; c.fillStyle = C.selected; c.fillRect(bx, 0, bw, totalH); c.globalAlpha = 1;
    }
    c.font = "11px " + C.font; c.textBaseline = "middle";
    geo.forEach(function (g) {
      if (g.type === "head") { c.fillStyle = C["bg-head"]; c.fillRect(0, g.y, W, g.h); c.fillStyle = C.rule; c.fillRect(0, g.y + g.h - 1, W, 1); return; }
      var row = RUNS[g.run].rows[g.row], ops = row.ops, issue = row.unit < 0;
      c.fillStyle = C.rule; c.fillRect(0, g.y + g.h - 1, W, 1);
      // first op whose end is at or after the view start (ops of a row do not overlap)
      var lo = 0, hi = ops.length;
      while (lo < hi) { var mid = (lo + hi) >> 1; if (ops[mid][0] + ops[mid][1] < view.a) lo = mid + 1; else hi = mid; }
      var ytop = g.y + 3, hh = g.h - 7, lastX = -1, lastEnd = -1;
      for (var k = lo; k < ops.length; k++) {
        var o = ops[k]; if (o[0] > view.b) break;
        var x0 = Math.floor(xOf(o[0])), x1 = Math.max(Math.round(xOf(o[0] + o[1])), x0 + 1);
        var w = x1 - x0;
        if (!issue && x1 <= lastEnd && w <= 1) continue;
        c.fillStyle = recColor(o[2]);
        c.fillRect(x0, ytop, issue && w > 3 ? w - 1 : w, hh);
        lastEnd = x1; lastX = x0;
        if (w > 64) {
          var lab = issue ? (R[o[2]] ? R[o[2]].kn : "") : opName(o[3]);
          c.save(); c.beginPath(); c.rect(x0, ytop, w, hh); c.clip(); c.fillStyle = "#ffffff"; c.fillText(lab, x0 + 3, ytop + hh / 2 + 0.5); c.restore();
        }
      }
    });
    // the selected record's ops, outlined
    if (st.sel >= 0) {
      c.strokeStyle = C.accent; c.lineWidth = 1.5;
      RUNS.forEach(function (run, ri) {
        (recIdx[ri][st.sel] || []).forEach(function (ref) {
          var g = geoOf(ri, ref[0]); if (!g) return;
          var o = run.rows[ref[0]].ops[ref[1]];
          var x0 = xOf(o[0]), x1 = xOf(o[0] + o[1]); if (x1 < 0 || x0 > W) return;
          c.strokeRect(Math.floor(x0) + 0.5, g.y + 2.5, Math.max(Math.round(x1 - x0), 2), g.h - 6);
        });
      });
    }
    if (st.op) {
      var g2 = geoOf(st.op.run, st.op.row);
      if (g2) {
        var oo = RUNS[st.op.run].rows[st.op.row].ops[st.op.idx], xa = xOf(oo[0]), xb = xOf(oo[0] + oo[1]);
        c.strokeStyle = C.fg; c.lineWidth = 1.5; c.strokeRect(Math.floor(xa) + 0.5, g2.y + 1.5, Math.max(Math.round(xb - xa), 2), g2.h - 4);
      }
    }
    if (cursor.x >= 0) { c.strokeStyle = C.fg3; c.lineWidth = 1; c.beginPath(); var cx = Math.round(cursor.x) + 0.5; c.moveTo(cx, 0); c.lineTo(cx, totalH); c.stroke(); }
    $("#tlrange").textContent = fint(view.a) + " to " + fint(view.b) + " cycles, " + ftime(fus(view.b - view.a), 1000) + " shown";
  }
  var geoCache = {};
  function geoOf(ri, rk) { for (var i = 0; i < geo.length; i++) if (geo[i].type === "row" && geo[i].run === ri && geo[i].row === rk) return geo[i]; return null; }

  function hitTest(px, py) {
    var g = null;
    for (var i = 0; i < geo.length; i++) if (py >= geo[i].y && py < geo[i].y + geo[i].h) { g = geo[i]; break; }
    if (!g || g.type !== "row") return null;
    var row = RUNS[g.run].rows[g.row], ops = row.ops, tol = 2 * (view.b - view.a) / W, t = tOf(px);
    var lo = 0, hi = ops.length;
    while (lo < hi) { var mid = (lo + hi) >> 1; if (ops[mid][0] <= t + tol) lo = mid + 1; else hi = mid; }
    for (var k = lo - 1; k >= 0 && k > lo - 6; k--) {
      var o = ops[k];
      if (t >= o[0] - tol && t <= o[0] + o[1] + tol) return { run: g.run, row: g.row, idx: k, o: o };
    }
    return null;
  }

  function tipHtml(h) {
    var run = RUNS[h.run], row = run.rows[h.row], o = h.o, r = R[o[2]], rows = [];
    function kv(a, b) { rows.push("<tr><td>" + a + "</td><td class=\"num\">" + b + "</td></tr>"); }
    if (row.unit < 0) {
      kv("Interval", fint(o[0]) + " to " + fint(o[0] + o[1]) + " cycles");
      kv("Length", fint(o[1]) + " cycles, " + ftime(fus(o[1]), 1));
      if (r) { kv("Group", esc(GL[r.group])); kv("Layer", esc(layerName(r.layer))); kv("Frontier cost", fint(r.cost) + " cycles"); }
      return '<div class="h">' + esc(run.label) + " issue, " + (r ? esc(recLabel(r)) : "record " + o[2]) + "</div><table>" + rows.join("") + "</table>";
    }
    var un = D.units[row.unit];
    kv("Start", fint(o[0]) + " cycles");
    kv("Duration", fint(o[1]) + " cycles, " + ftime(fus(o[1]), 1));
    kv("Queue wait", fint(o[5]) + " cycles"); kv("Head wait", fint(o[6]) + " cycles");
    if (o[7]) { kv("Ideal", fx(o[7], 0) + " cycles"); kv("Efficiency", fpct(o[7] / o[1]) + " %"); }
    if (r) kv("Layer", esc(layerName(r.layer)));
    return '<div class="h">' + esc(un + " " + opName(o[3]) + "  " + (r ? recLabel(r) : "launch setup")) + "</div><table>" + rows.join("") + "</table>";
  }

  // pointer handling: drag pans, a click without movement selects, the wheel zooms
  var drag = null;
  function localXY(e, el) { var b = el.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top]; }
  function zoomAt(px, f) {
    var tc = tOf(px), w = (view.b - view.a) * f;
    if (w > SPAN) w = SPAN; if (w < 40) w = 40;
    var a = tc - (px / W) * w; view.a = a; view.b = a + w; clampView(); draw();
  }
  [cvR, cvT].forEach(function (el) {
    el.addEventListener("pointerdown", function (e) {
      if (e.button !== 0) return;
      drag = { x: e.clientX, y: e.clientY, a: view.a, b: view.b, moved: false, el: el };
      try { el.setPointerCapture(e.pointerId); } catch (x) { /* no capture */ }
    });
    el.addEventListener("pointermove", function (e) {
      var p = localXY(e, el);
      if (drag) {
        var dx = e.clientX - drag.x;
        if (Math.abs(dx) > 3 || Math.abs(e.clientY - drag.y) > 3) drag.moved = true;
        if (drag.moved) { var w = drag.b - drag.a; view.a = drag.a - dx * w / W; view.b = view.a + w; clampView(); tip.hidden = true; }
        cursor.x = p[0]; draw(); return;
      }
      if (el === cvR) {
        cursor.x = p[0];
        var h = hitTest(p[0], p[1]);
        if (h) { tip.innerHTML = tipHtml(h); tip.hidden = false; placeTip(e); } else tip.hidden = true;
      } else cursor.x = p[0];
      $("#st-cur").textContent = fint(tOf(p[0])) + " cycles  " + ftime(fus(tOf(p[0])), 1);
      draw();
    });
    el.addEventListener("pointerup", function (e) {
      var d = drag; drag = null;
      if (d && !d.moved && el === cvR) {
        var p = localXY(e, el), h = hitTest(p[0], p[1]);
        if (h) {
          var rec = h.o[2];
          select(rec, h.o && RUNS[h.run].rows[h.row].unit >= 0 ? { run: h.run, row: h.row, idx: h.idx } : null, "timeline");
        }
      }
    });
    el.addEventListener("pointerleave", function () { if (!drag) { cursor.x = -1; tip.hidden = true; $("#st-cur").textContent = ""; draw(); } });
    el.addEventListener("dblclick", function () { view.a = 0; view.b = SPAN; draw(); });
    el.addEventListener("wheel", function (e) {
      e.preventDefault();
      var p = localXY(e, el);
      if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) { var w = view.b - view.a, d = (e.deltaX || e.deltaY) * w / W; view.a += d; view.b += d; clampView(); draw(); }
      else zoomAt(p[0], Math.exp(e.deltaY * 0.0016));
    }, { passive: false });
  });
  function placeTip(e) {
    var w = tip.offsetWidth, h = tip.offsetHeight, x = e.clientX + 14, y = e.clientY + 14;
    if (x + w > window.innerWidth - 4) x = e.clientX - w - 10;
    if (y + h > window.innerHeight - 4) y = e.clientY - h - 10;
    tip.style.left = Math.max(x, 2) + "px"; tip.style.top = Math.max(y, 2) + "px";
  }
  $("#zin").addEventListener("click", function () { zoomAt(W / 2, 0.5); });
  $("#zout").addEventListener("click", function () { zoomAt(W / 2, 2); });
  $("#zfit").addEventListener("click", function () { view.a = 0; view.b = SPAN; draw(); });
  function zoomSel() {
    var r = st.sel >= 0 ? R[st.sel] : null; if (!r || r.first == null) return;
    var w = Math.max((r.last - r.first) * 1.25, 3000), mid = (r.first + r.last) / 2;
    view.a = mid - w / 2; view.b = mid + w / 2; clampView(); draw();
  }
  $("#zsel").addEventListener("click", zoomSel);
  tlscroll.addEventListener("keydown", function (e) {
    var w = view.b - view.a;
    if (e.key === "+" || e.key === "=") zoomAt(W / 2, 0.5);
    else if (e.key === "-") zoomAt(W / 2, 2);
    else if (e.key === "0") { view.a = 0; view.b = SPAN; draw(); }
    else if (e.key === "ArrowLeft") { view.a -= w * 0.2; view.b -= w * 0.2; clampView(); draw(); }
    else if (e.key === "ArrowRight") { view.a += w * 0.2; view.b += w * 0.2; clampView(); draw(); }
    else return;
    e.preventDefault();
  });
  function revealRec(rec) {
    var r = R[rec]; if (!r || r.first == null) return;
    var w = view.b - view.a;
    if (r.first >= view.a && r.last <= view.b) return;
    var mid = (r.first + r.last) / 2; view.a = mid - w / 2; view.b = mid + w / 2; clampView();
  }
  if (window.ResizeObserver) new ResizeObserver(function () { drawAll(); }).observe(tlscroll); else window.addEventListener("resize", drawAll);

  // splitter
  (function () {
    var sp = $("#splitter"), main = $("#main"), on = false;
    sp.addEventListener("pointerdown", function (e) { on = true; sp.classList.add("drag"); sp.setPointerCapture(e.pointerId); });
    sp.addEventListener("pointermove", function (e) {
      if (!on) return;
      var b = main.getBoundingClientRect(), h = Math.min(Math.max(e.clientY - b.top, 120), b.height - 90);
      main.style.setProperty("--tl-h", h + "px");
    });
    sp.addEventListener("pointerup", function () { on = false; sp.classList.remove("drag"); });
  })();

  // ---------------------------------------------------------------- tree
  var treelist = $("#treelist");
  function recMatch(r, f) { return (r.kn + " " + r.name + " " + r.group + " " + layerName(r.layer)).toLowerCase().indexOf(f) >= 0; }
  function treeRows() {
    var rows = [], f = st.treeFilter.toLowerCase();
    D.launches.forEach(function (l) {
      var det = l.k === M.launch, id = "L" + l.k;
      if (f && !det) return;
      rows.push({ id: id, depth: 0, kind: "L", l: l, det: det });
      if (!det || st.expanded[id] === false) return;
      D.layers.forEach(function (ly) {
        var ids = byLayer[ly] || [], m = f ? ids.filter(function (i) { return recMatch(R[i], f); }) : ids;
        if (!m.length) return;
        var yid = "y" + ly;
        rows.push({ id: yid, depth: 1, kind: "y", ly: ly, n: m.length, cost: m.reduce(function (a, i) { return a + R[i].cost; }, 0) });
        if (f || st.expanded[yid]) m.forEach(function (i) { rows.push({ id: "r" + i, depth: 2, kind: "r", r: R[i] }); });
      });
    });
    return rows;
  }
  function renderTree() {
    var rows = treeRows(), top = treelist.scrollTop, h = "";
    rows.forEach(function (x) {
      var sel = (x.kind === "r" && x.r.i === st.sel) || (x.id === st.treeSel && x.kind !== "r");
      var pad = 4 + x.depth * 14;
      if (x.kind === "L") {
        var open = x.det && st.expanded[x.id] !== false;
        h += '<div class="tr' + (x.det ? "" : " off") + (sel ? " sel" : "") + '" role="treeitem" data-id="' + x.id + '" style="padding-left:' + pad + 'px"' + (x.det ? "" : ' title="No records in this report. Run prof_report.py report with --launch ' + x.l.k + '."') + '><span class="ch">' + (x.det ? (open ? "-" : "+") : "") +
             '</span><span class="lb">Launch ' + x.l.k + '  <span class="kd">' + (x.l.decode ? "decode" : "other") + '</span></span><span class="cy">' + fint(x.l.cycles) + "</span></div>";
      } else if (x.kind === "y") {
        var op = st.treeFilter || st.expanded[x.id];
        h += '<div class="tr' + (sel ? " sel" : "") + '" role="treeitem" data-id="' + x.id + '" style="padding-left:' + pad + 'px"><span class="ch">' + (op ? "-" : "+") + '</span><span class="lb">' + esc(layerName(x.ly)) + ' <span class="kd">' + x.n + ' rec</span></span><span class="cy">' + fint(x.cost) + "</span></div>";
      } else {
        h += '<div class="tr' + (sel ? " sel" : "") + '" role="treeitem" data-id="' + x.id + '" style="padding-left:' + (pad + 14) + 'px"><span class="lb"><i class="' + gcls(x.r.group) + '"></i>' + esc(x.r.kn) + ' <span class="kd">' + esc(x.r.name) + '</span></span><span class="cy">' + fint(x.r.cost) + "</span></div>";
      }
    });
    treelist.innerHTML = h;
    treelist.scrollTop = top;
    var s = $(".tr.sel", treelist);
    if (s) { var b = s.getBoundingClientRect(), p = treelist.getBoundingClientRect(); if (b.top < p.top || b.bottom > p.bottom) treelist.scrollTop += (b.top - p.top) - (p.height / 2); }
  }
  treelist.addEventListener("click", function (e) {
    var t = e.target.closest(".tr"); if (!t) return;
    var id = t.getAttribute("data-id"), k = id[0];
    if (k === "r") { select(+id.slice(1), null, "tree"); return; }
    st.treeSel = id;
    if (k === "L") { if (t.classList.contains("off")) { $("#st-sel").textContent = "Launch " + id.slice(1) + " has no records in this report. Run prof_report.py report with --launch " + id.slice(1) + "."; return; } st.expanded[id] = st.expanded[id] === false; }
    else st.expanded[id] = !st.expanded[id];
    renderTree();
  });
  treelist.addEventListener("keydown", function (e) {
    var rows = treeRows(); if (!rows.length) return;
    var cur = st.sel >= 0 && !st.treeSel ? "r" + st.sel : (st.treeSel || ("r" + st.sel));
    var i = rows.findIndex(function (x) { return x.id === cur; });
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      i = Math.min(Math.max(i + (e.key === "ArrowDown" ? 1 : -1), 0), rows.length - 1);
      var x = rows[i];
      if (x.kind === "r") { st.treeSel = ""; select(x.r.i, null, "tree"); } else { st.treeSel = x.id; renderTree(); }
    } else if ((e.key === "ArrowRight" || e.key === "ArrowLeft" || e.key === "Enter") && i >= 0 && rows[i].kind === "y") {
      st.expanded[rows[i].id] = e.key === "ArrowLeft" ? false : true; renderTree();
    } else return;
    e.preventDefault();
  });
  $("#treefilter").addEventListener("input", function (e) { st.treeFilter = e.target.value.trim(); renderTree(); });
  $("#treebtn").addEventListener("click", function () { app.classList.toggle("tree-open"); renderTree(); });

  // ---------------------------------------------------------------- selection
  function select(rec, op, src) {
    st.sel = rec; st.op = op || null;
    if (rec >= 0) { st.treeSel = ""; st.expanded["y" + R[rec].layer] = true; }
    if (src !== "timeline" && rec >= 0) revealRec(rec);
    if (src === "tree" || src === "timeline") { if (st.tab === "summary" || st.tab === "graph") setTab("details"); }
    renderTree(); markRows(); renderDetails(); draw(); statusSel();
  }
  function statusSel() {
    var r = st.sel >= 0 ? R[st.sel] : null;
    $("#st-sel").textContent = r ? "r" + r.i + "  " + recLabel(r) + "  " + fint(r.cost) + " cycles  " + fpct(div(r.cost, M.span)) + " %" : "";
  }
  function markRows() {
    $$("#p-records tr.sel").forEach(function (t) { t.classList.remove("sel"); });
    if (st.sel < 0) return;
    var r = R[st.sel], t;
    if (st.recView === "record") t = $('#p-records tr[data-r="' + st.sel + '"]'); else t = $('#p-records tr[data-kind="' + r.kn + '"]');
    if (t) { t.classList.add("sel"); scrollInto(t, $("#p-records")); }
    $$("#p-summary tr.sel").forEach(function (x) { x.classList.remove("sel"); });
    $$('#p-summary tr[data-r="' + st.sel + '"]').forEach(function (x) { x.classList.add("sel"); });
  }
  function scrollInto(el, box) {
    var b = el.getBoundingClientRect(), p = box.getBoundingClientRect();
    if (b.top < p.top + 52) box.scrollTop -= (p.top + 52 - b.top) + 20; else if (b.bottom > p.bottom) box.scrollTop += (b.bottom - p.bottom) + 20;
  }

  // ---------------------------------------------------------------- tabs
  function setTab(t) {
    st.tab = t;
    $$("#tabs button").forEach(function (b) { b.setAttribute("aria-selected", b.getAttribute("data-tab") === t ? "true" : "false"); });
    $$(".pane").forEach(function (p) { p.classList.toggle("on", p.id === "p-" + t); });
    if (t === "records" && !$("#p-records").firstChild) renderRecords();
    if (t === "graph" && !$("#p-graph").firstChild) renderGraph();
    if (t === "records") markRows();
  }
  $("#tabs").addEventListener("click", function (e) { var b = e.target.closest("button"); if (b) setTab(b.getAttribute("data-tab")); });

  // ---------------------------------------------------------------- summary
  function renderSummary() {
    var s = S, h = "", on = s.llama_on || {}, off = s.llama_off || {};
    var run = [
      mrow("Model", "", esc(M.model || "-")), mrow("Image", "", '<span class="num">' + esc(M.image) + "</span>"), mrow("Bitstream", "", '<span class="num">' + esc(M.xclbin || "-") + "</span>"),
      mrow("Commit", "", '<span class="num">' + esc(M.commit || "-") + "</span>"), mrow("Captured", "", esc(M.date || "-")),
      mrow("Launch", "", fint(M.launch) + " of " + fint(M.n_launches)), mrow("Decode tokens captured", "", fint(s.n_decode))
    ];
    var speed = [];
    if (off.toks) speed.push(mrow("Decode speed, profiler off", "tok/s", fx(off.toks, 2)));
    else if (s.ref_toks) speed.push(mrow("Decode speed, reference", "tok/s", fx(s.ref_toks, 2)));
    if (on.toks) speed.push(mrow("Decode speed, profiler on", "tok/s", fx(on.toks, 2)));
    if (s.tok_s) { speed.push(mrow("Decode speed, token period", "tok/s", fx(s.tok_s, 2))); speed.push(mrow("Token period", "ms", fx(s.token_period_ms, 2))); }
    speed.push(mrow("Device time per token", "ms", fx(s.t_dev_ms, 2)));
    speed.push(mrow("Device speed", "tok/s", fx(s.tok_s_dev, 2)));
    var dev = [
      mrow("Device cycles (PERF)", "cycle", fint(s.cycles)), mrow("Launch span", "cycle", fint(s.span)), mrow("Setup", "cycle", fint(s.setup)), mrow("Tail", "cycle", fint(s.tail)),
      mrow("Core clock", "MHz", fx(F / 1e6, 1)), mrow("MVM busy", "cycle", fint(s.mvm_cycles)), mrow("MVM busy share", "%", fpct(div(s.mvm_cycles, s.cycles))),
      mrow("MVM ideal", "cycle", fint(s.ideal_cycles)), mrow("MVM ideal streams", "", s.mvm_dual ? esc("2 for T <= " + (s.t_max >> 1)) : "1"),
      mrow("MVM empty", "%", fpct(s.empty)), mrow("MVM wait", "%", fpct(s.wait)),
      mrow("DDR read (PERF)", "GB", fx(s.rd_meas / 1e9, 3)), mrow("DDR read (records)", "GB", fx(s.rd_model / 1e9, 3)), mrow("Weights", "GB", fx(s.w_bytes / 1e9, 3))
    ];
    h += '<div class="grid">';
    h += section("Run", table([["Field", ""], ["", "u"], ["Value", ""]], run));
    h += section("Speed", table(H3, speed));
    h += section("Device", table(H3, dev));
    if (s.U != null) {
      var u = [
        ["H", "No launch active (host)", "%", fpct(s.H)], ["G", "Launch cycles without MVM", "%", fpct(s.G)], ["F", "MVM cycles above ideal", "%", fpct(s.F)],
        ["E", "Read rate over the stream rate", "%", fpct(s.E)], ["U", "(1 - H)(1 - G)(1 - F) E", "%", fpct(s.U)], ["E at clock", "Stream rate ceiling", "%", fpct(s.E_clock)]
      ].map(function (x) { return [esc(x[0]), esc(x[1]), esc(x[2]), x[3]]; });
      h += section("Utilization", table([["Factor", ""], ["Meaning", ""], ["Unit", "u"], ["Value", "n"]], u));
    }
    // groups
    var gt = D.group_totals, tot = 0, grows = [];
    D.groups.forEach(function (g) {
      var t = gt[g.id]; tot += t.cycles;
      grows.push([ '<i class="' + gcls(g.id) + '"></i>' + esc(g.label), fint(t.cycles), fpct(t.cycles / SPAN), fint(t.ideal), t.ideal && t.cycles ? fpct(t.ideal / t.cycles) : "-"].concat(M.has_sim ? [fint(t.sim)] : []));
    });
    grows.push(["Setup and tail", fint(s.setup + s.tail), fpct((s.setup + s.tail) / SPAN), "-", "-"].concat(M.has_sim ? ["-"] : []));
    h += section("Cycles by group", table([["Group", ""], ["Cycles", "n"], ["Share %", "n"], ["Ideal", "n"], ["Eff. %", "n"]].concat(M.has_sim ? [["Sim cycles", "n"]] : []), grows));
    // hotspots
    var top = R.slice().sort(function (a, b) { return b.cost - a.cost; }).slice(0, 10).map(function (r) {
      return { cells: ["r" + r.i, '<i class="' + gcls(r.group) + '"></i>' + esc(recLabel(r)), fint(r.cost), fpct(r.cost / SPAN)], cls: "clk", attr: ' data-r="' + r.i + '"' };
    });
    h += section("Top records", table([["#", "id"], ["Record", ""], ["Cycles", "n"], ["Share %", "n"]], top));
    if (M.has_sim && D.sim) {
      var sm = D.sim, p = D.perf, q = sm.perf;
      function cmp(name, a, b) { return [esc(name), fint(a), fint(b), b && a ? fx(b / a, 4) : "-"]; }
      var cr = [cmp("Launch span", s.span, sm.span), cmp("PERF cycles", p.CYCLES, q.CYCLES), cmp("MVM cycles", p.MVM_CYCLES, q.MVM_CYCLES), cmp("VU cycles", p.VU_CYCLES, q.VU_CYCLES),
                cmp("DMA cycles", p.DMA_CYCLES, q.DMA_CYCLES), cmp("QACT cycles", p.QACT_CYCLES, q.QACT_CYCLES), cmp("Retired ops", p.RETIRED, q.RETIRED)];
      h += section("Card and simulator", table([["Metric (cycle)", ""], ["Card", "n"], ["Simulator", "n"], ["Sim / card", "n"]], cr));
    } else if (M.sim_note) h += section("Card and simulator", '<div class="msg">' + esc(M.sim_note) + "</div>");
    var ck = D.checks.map(function (c) { return [c.ok === true ? '<span class="ok">Pass</span>' : c.ok === false ? '<span class="bad">Fail</span>' : '<span class="dim">Info</span>', '<span style="white-space:normal">' + esc(c.text) + "</span>"]; });
    h += section("Checks", table([["Status", ""], ["Check", ""]], ck));
    var pt = D.per_token.map(function (x) { return [fint(x.k), fint(x.card), x.sim != null ? fint(x.sim) : "-", x.sim ? fx(x.sim / x.card, 4) : "-"]; });
    if (pt.length) h += section("Decode launches", table([["Launch", "n"], ["Card cycles", "n"], ["Sim cycles", "n"], ["Sim / card", "n"]], pt));
    if (M.commands) h += section("Commands", '<pre class="cmd">' + esc(M.commands) + "</pre>");
    h += "</div>";
    $("#p-summary").innerHTML = h;
  }
  $("#p-summary").addEventListener("click", function (e) { var t = e.target.closest("tr[data-r]"); if (t) select(+t.getAttribute("data-r"), null, "table"); });

  // ---------------------------------------------------------------- records
  function sortRows(rows, key, dir) {
    return rows.sort(function (a, b) {
      var x = a[key], y = b[key];
      if (x == null && y == null) return a._o - b._o; if (x == null) return 1; if (y == null) return -1;
      if (typeof x === "string") { var c = x < y ? -1 : x > y ? 1 : 0; return c ? c * dir : a._o - b._o; }
      return x === y ? a._o - b._o : (x - y) * dir;
    });
  }
  var KIND_COLS = [["Kind", "s", "kind"], ["Count", "s n", "count"], ["Cycles", "s n", "cycles"], ["Share %", "s n", "share"], ["Ideal", "s n", "ideal"], ["Eff. %", "s n", "eff"], ["GB/s", "s n", "gbs"]];
  var REC_COLS = [["#", "s n", "i"], ["Kind", "s", "kn"], ["Name", "s", "name"], ["Layer", "s", "lk"], ["Group", "s", "group"], ["Cycles", "s n", "cost"], ["Share %", "s n", "share"], ["Ideal", "s n", "ideal"], ["Eff. %", "s n", "eff"], ["GB/s", "s n", "gbs"]];
  function renderRecords() {
    var p = $("#p-records");
    p.innerHTML = '<div class="bar"><span class="seg"><button type="button" data-v="kind">By kind</button><button type="button" data-v="record">By record</button></span>' +
      '<input id="recfilter" type="search" placeholder="Filter" aria-label="Filter records"><select id="recgroup" aria-label="Group"><option value="">All groups</option>' +
      D.groups.map(function (g) { return '<option value="' + g.id + '">' + esc(g.label) + "</option>"; }).join("") + '</select><span id="reccount" class="dim num"></span></div><div class="tw" id="rectw"></div>';
    $$(".seg button", p).forEach(function (b) { b.setAttribute("aria-pressed", b.getAttribute("data-v") === st.recView ? "true" : "false"); });
    renderRecTable();
  }
  function renderRecTable() {
    var tw = $("#rectw"), f = st.recFilter.toLowerCase(), view_ = st.recView, sort = st.sort[view_], rows, heads, h, n = 0;
    var sim = M.has_sim;
    if (view_ === "kind") {
      var gk = {};
      R.forEach(function (r) { if (st.recGroup && r.group !== st.recGroup) return; if (f && r.kn.toLowerCase().indexOf(f) < 0) return; var d = gk[r.kn] = gk[r.kn] || { kind: r.kn, count: 0, cycles: 0, ideal: 0, bytes: 0, sim: 0, best: r.i }; d.count++; d.cycles += r.cost; d.ideal += r.ideal; d.bytes += r.rd + r.wr; d.sim += r.sim || 0; if (r.cost > R[d.best].cost) d.best = r.i; });
      rows = Object.keys(gk).map(function (k, o) { var d = gk[k]; d._o = o; d.share = d.cycles / SPAN; d.eff = d.ideal && d.cycles ? d.ideal / d.cycles : null; d.gbs = d.cycles ? d.bytes / (d.cycles / F) / 1e9 : null; d.ratio = sim && d.cycles ? d.sim / d.cycles : null; return d; });
      sortRows(rows, sort[0], sort[1]);
      heads = KIND_COLS.concat(sim ? [["Sim cycles", "s n", "sim"], ["Sim / card", "s n", "ratio"]] : []);
      h = rows.map(function (d) {
        return '<tr class="clk" data-kind="' + esc(d.kind) + '" data-r="' + d.best + '"><td>' + esc(d.kind) + '</td><td class="n">' + fint(d.count) + '</td><td class="n">' + fint(d.cycles) + '</td><td class="n cell"><i style="width:' + Math.min(d.share * 100, 100) + '%"></i>' + fpct(d.share) + '</td><td class="n">' + (d.ideal ? fint(d.ideal) : "-") + '</td><td class="n">' + (d.eff != null ? fpct(d.eff) : "-") + '</td><td class="n">' + fx(d.gbs, 2) + "</td>" +
          (sim ? '<td class="n">' + fint(d.sim) + '</td><td class="n">' + (d.ratio != null ? fx(d.ratio, 3) : "-") + "</td>" : "") + "</tr>";
      }).join("");
      n = rows.length;
    } else {
      rows = [];
      R.forEach(function (r) {
        if (st.recGroup && r.group !== st.recGroup) return;
        if (f && !recMatch(r, f)) return;
        rows.push({ r: r, i: r.i, kn: r.kn, name: r.name, lk: r.layer === "in" ? -1 : r.layer === "out" ? 1e6 : r.layer, group: r.group, cost: r.cost, share: r.cost / SPAN, ideal: r.ideal, eff: r.ideal && r.cost ? r.ideal / r.cost : null,
          gbs: r.cost ? (r.rd + r.wr) / (r.cost / F) / 1e9 : null, sim: r.sim != null ? r.sim : null, ratio: r.sim != null && r.cost ? r.sim / r.cost : null, _o: rows.length });
      });
      sortRows(rows, sort[0], sort[1]);
      heads = REC_COLS.concat(sim ? [["Sim cycles", "s n", "sim"], ["Sim / card", "s n", "ratio"]] : []);
      h = rows.map(function (x) {
        var r = x.r;
        return '<tr class="clk" data-r="' + r.i + '"><td class="n">' + r.i + "</td><td>" + esc(r.kn) + "</td><td>" + esc(r.name) + "</td><td>" + esc(layerName(r.layer)) + '</td><td><i class="' + gcls(r.group) + '"></i>' + esc(GL[r.group]) + '</td><td class="n">' + fint(r.cost) + '</td><td class="n cell"><i style="width:' + Math.min(x.share * 100 * 8, 100) + '%"></i>' + fpct(x.share, 2) + '</td><td class="n">' + (r.ideal ? fint(r.ideal) : "-") + '</td><td class="n">' + (x.eff != null ? fpct(x.eff) : "-") + '</td><td class="n">' + fx(x.gbs, 2) + "</td>" +
          (sim ? '<td class="n">' + (x.sim != null ? fint(x.sim) : "-") + '</td><td class="n">' + (x.ratio != null ? fx(x.ratio, 3) : "-") + "</td>" : "") + "</tr>";
      }).join("");
      n = rows.length;
    }
    var th = heads.map(function (c) {
      var on = sort[0] === c[2];
      return '<th class="' + c[1] + '" data-k="' + c[2] + '"' + (on ? ' data-dir="' + sort[1] + '"' : "") + ">" + c[0] + "</th>";
    }).join("");
    tw.innerHTML = "<table><thead><tr>" + th + "</tr></thead><tbody>" + h + "</tbody></table>";
    $("#reccount").textContent = fint(n) + (view_ === "kind" ? " kinds" : " records");
    markRows();
  }
  $("#p-records").addEventListener("click", function (e) {
    var b = e.target.closest(".seg button");
    if (b) { st.recView = b.getAttribute("data-v"); $$(".seg button", $("#p-records")).forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); }); renderRecTable(); return; }
    var th = e.target.closest("th.s");
    if (th) { var k = th.getAttribute("data-k"), s = st.sort[st.recView]; st.sort[st.recView] = [k, s[0] === k ? -s[1] : (k === "kind" || k === "kn" || k === "name" || k === "group" ? 1 : -1)]; renderRecTable(); return; }
    var tr = e.target.closest("tr[data-r]"); if (tr) select(+tr.getAttribute("data-r"), null, "table");
  });
  $("#p-records").addEventListener("dblclick", function (e) { var tr = e.target.closest("tr[data-r]"); if (tr) setTab("details"); });
  $("#p-records").addEventListener("input", function (e) { if (e.target.id === "recfilter") { st.recFilter = e.target.value.trim(); renderRecTable(); } });
  $("#p-records").addEventListener("change", function (e) { if (e.target.id === "recgroup") { st.recGroup = e.target.value; renderRecTable(); } });

  // ---------------------------------------------------------------- graph
  function renderGraph() {
    var p = $("#p-graph");
    p.innerHTML = '<div class="bar"><span class="seg"><button type="button" data-v="splits">Splits</button><button type="button" data-v="nodes">Nodes</button><button type="button" data-v="cpu">CPU fallbacks</button></span><span id="gbar"></span></div><div class="tw" id="gv"></div>';
    renderGraphView();
  }
  function renderGraphView() {
    var v = st.graphView, gv = $("#gv"), h = "", gbar = $("#gbar");
    $$("#p-graph .seg button").forEach(function (b) { b.setAttribute("aria-pressed", b.getAttribute("data-v") === v ? "true" : "false"); });
    gbar.innerHTML = "";
    if (v === "splits") {
      var cnt = { device: 0, absorbed: 0, noop: 0 };
      D.nodes.forEach(function (n) { cnt[n.place] = (cnt[n.place] || 0) + 1; });
      var kinds = {}; D.fusion.forEach(function (g) { kinds[g.kind] = (kinds[g.kind] || 0) + 1; });
      h += '<div class="grid">';
      var sp = D.sched ? D.sched.map(function (s) { return [fint(s.idx), esc(s.backend), fint(s.n_nodes), fint((s.inputs || []).length), s.backend.indexOf("CPU") === 0 ? esc((s.node_names || []).slice(0, 4).join(", ")) : ""]; }) : [];
      h += section("Scheduler splits", D.sched ? table([["Split", "n"], ["Backend", ""], ["Nodes", "n"], ["Inputs", "n"], ["First nodes", ""]], sp) : '<div class="msg">No scheduler log in this capture.</div>');
      h += section("Nodes", table([["Placement", ""], ["Nodes", "n"]], [["Device", fint(cnt.device)], ["Absorbed by a fusion group", fint(cnt.absorbed)], ["View or reshape", fint(cnt.noop)], ["CPU", fint(D.cpu.length)], ["Fusion groups", fint(D.fusion.length)], ["Records", fint(R.length)]]));
      h += section("Lists", table([["List", "n"], ["Records", "n"], ["JOIN", "n"], ["Op list B", "n"], ["Preload entries", "n"], ["Preload lines", "n"], ["Ended by", ""]],
        D.lists.map(function (l) { return [fint(l.idx), fint(l.n_ops), fint(l.joins), fint(l.blob), fint(l.pre[0]), fint(l.pre[1]), esc(l.end)]; })));
      h += section("Fusion groups", table([["Kind", ""], ["Groups", "n"]], Object.keys(kinds).sort().map(function (k) { return [esc(k), fint(kinds[k])]; })));
      h += section("Launches", table([["Launches", ""], ["What", ""], ["Count", "n"], ["Records", "n"], ["PERF cycles", "n"], ["Plan", ""]], D.launch_groups.map(function (g) {
        var ks = g.launches, cy = g.cycles, mn = Math.min.apply(null, cy), mx = Math.max.apply(null, cy);
        return [ks.length === 1 ? String(ks[0]) : ks[0] + " to " + ks[ks.length - 1], g.decode ? "decode token" : (ks[0] === 0 ? "warm-up pass, " : "prompt, ") + g.tokens + " tokens", fint(ks.length), fint(g.n_ops), mn === mx ? fint(mn) : fint(mn) + " to " + fint(mx), g.hits ? "plan cache, " + g.hits + " hits" : "built"];
      })));
      if (D.dynamic.length) h += section("Dynamic plan", '<div class="msg">' + esc(D.dynamic.join("; ")) + "</div>");
      h += "</div>";
    } else if (v === "nodes") {
      gbar.innerHTML = '<input id="nodefilter" type="search" placeholder="Filter" aria-label="Filter nodes" value="' + esc(st.nodeFilter) + '"><select id="nodeplace" aria-label="Placement"><option value="">All</option><option value="device">Device</option><option value="absorbed">Absorbed</option><option value="noop">View</option></select><span id="nodecount" class="dim num"></span>';
      $("#nodeplace").value = st.nodePlace;
      h = '<div id="nodetab"></div>';
    } else {
      h = D.cpu.length ? table([["Node", ""], ["Op", ""], ["Reason", ""]], D.cpu.map(function (c) { return [esc(c.name), esc(c.op), esc(c.why)]; })) : '<div class="msg">No node of the decode graph runs on the CPU.</div>';
    }
    gv.innerHTML = h;
    if (v === "nodes") renderNodes();
  }
  function renderNodes() {
    var f = st.nodeFilter.toLowerCase(), rows = [], sort = st.sort.node;
    D.nodes.forEach(function (n) {
      if (st.nodePlace && n.place !== st.nodePlace) return;
      var key = (n.name + " " + n.op + " " + n.type).toLowerCase();
      if (f && key.indexOf(f) < 0) return;
      rows.push({ n: n, i: n.i, name: n.name, op: n.op, type: n.type, place: n.place, grp: n.g, layer: n.layer === "in" ? -1 : n.layer === "out" ? 1e6 : n.layer, nrec: n.recs.length, cost: n.cost, _o: rows.length });
    });
    sortRows(rows, sort[0], sort[1]);
    var cols = [["#", "s n", "i"], ["Node", "s", "name"], ["Op", "s", "op"], ["Type", "s", "type"], ["Shape", "", ""], ["Placement", "s", "place"], ["Group", "s", "grp"], ["Layer", "s", "layer"], ["Records", "s n", "nrec"], ["Cycles", "s n", "cost"]];
    var shown = rows.slice(0, 1500);
    var th = cols.map(function (c) { var on = sort[0] === c[2]; return '<th class="' + c[1] + '"' + (c[2] ? ' data-k="' + c[2] + '"' : "") + (on ? ' data-dir="' + sort[1] + '"' : "") + ">" + c[0] + "</th>"; }).join("");
    var h = shown.map(function (x) {
      var n = x.n;
      return '<tr class="' + (n.recs.length ? "clk" : "") + '"' + (n.recs.length ? ' data-r="' + n.recs[0] + '"' : "") + '><td class="n">' + n.i + "</td><td>" + esc(n.name) + "</td><td>" + esc(n.op) + "</td><td>" + esc(n.type) + '</td><td class="id">' + esc(n.ne.join(" x ")) + "</td><td>" + esc(n.place) + '</td><td><i class="' + gcls(n.g) + '"></i>' + esc(GL[n.g] || n.g) + "</td><td>" + esc(layerName(n.layer)) + '</td><td class="n">' + n.recs.length + '</td><td class="n">' + (n.cost ? fint(n.cost) : "-") + "</td></tr>";
    }).join("");
    $("#nodetab").innerHTML = "<table><thead><tr>" + th + "</tr></thead><tbody>" + h + "</tbody></table>";
    $("#nodecount").textContent = fint(rows.length) + " nodes" + (rows.length > shown.length ? ", first " + fint(shown.length) : "");
  }
  $("#p-graph").addEventListener("click", function (e) {
    var b = e.target.closest(".seg button");
    if (b) { st.graphView = b.getAttribute("data-v"); renderGraphView(); return; }
    var th = e.target.closest("th.s");
    if (th && st.graphView === "nodes") { var k = th.getAttribute("data-k"), s = st.sort.node; st.sort.node = [k, s[0] === k ? -s[1] : (k === "name" || k === "op" || k === "type" || k === "place" ? 1 : -1)]; renderNodes(); return; }
    var tr = e.target.closest("tr[data-r]"); if (tr) select(+tr.getAttribute("data-r"), null, "table");
  });
  $("#p-graph").addEventListener("input", function (e) { if (e.target.id === "nodefilter") { st.nodeFilter = e.target.value.trim(); renderNodes(); } });
  $("#p-graph").addEventListener("change", function (e) { if (e.target.id === "nodeplace") { st.nodePlace = e.target.value; renderNodes(); } });

  // ---------------------------------------------------------------- details
  function renderDetails() {
    var p = $("#p-details");
    if (st.sel < 0) { p.innerHTML = '<div class="msg">No record selected.</div>'; return; }
    var r = R[st.sel], n = r.node >= 0 ? D.nodes[r.node] : null, sim = M.has_sim, h = '<div class="grid">';
    var dep = r.dep.map(function (d) { return d < 0 ? "none" : "r" + d; }).join(", ");
    h += section("Record", table([["Field", ""], ["Value", ""]], [["Record", "r" + r.i], ["Kind", esc(r.kn)], ["Name", esc(r.name)], ["Node", r.node >= 0 ? String(r.node) : "-"], ["Layer", esc(layerName(r.layer))],
      ["Group", '<i class="' + gcls(r.group) + '"></i>' + esc(GL[r.group])], ["Shape", esc(r.shape || "-")], ["Depends on", esc(dep)]].map(function (x) { return [esc(x[0]), x[1]]; })));
    var eff = r.ideal && r.cost ? r.ideal / r.cost : null;
    var rows = [
      ["Cycles", "cycle", fint(r.cost), sim ? fint(r.sim) : null], ["Share of launch", "%", fpct(r.cost / SPAN, 2), sim && r.sim != null ? fpct(r.sim / D.sim.span, 2) : null],
      ["Ideal", "cycle", r.ideal ? fint(r.ideal) : "-", null], ["Efficiency", "%", eff != null ? fpct(eff) : "-", sim && r.sim && r.ideal ? fpct(r.ideal / r.sim) : null],
      ["Ops", "op", fint(r.n_ops), sim ? fint(r.sim_ops) : null], ["First op start", "cycle", r.first != null ? fint(r.first) : "-", null], ["Last op end", "cycle", r.last != null ? fint(r.last) : "-", null]
    ];
    h += section("Cycles", table([["Metric", ""], ["Unit", "u"], ["Card", "n"]].concat(sim ? [["Simulator", "n"]] : []), rows.map(function (x) { return [esc(x[0]), esc(x[1]), x[2]].concat(sim ? [x[3] == null ? "" : x[3]] : []); })));
    h += section("Traffic", table(H3, [mrow("DRAM read", "B", fint(r.rd)), mrow("DRAM written", "B", fint(r.wr)), mrow("Weights", "B", fint(r.w)), mrow("Rate", "GB/s", r.cost ? fx((r.rd + r.wr) / (r.cost / F) / 1e9, 2) : "-")]));
    var ring = D.spm_lines - 16 - 4096 - 256 - 1024;
    h += section("SPM tiles", table(H3, [mrow(r.grouped ? "Tiles (planned least)" : "Tiles", "", fint(r.tiles)), mrow("Rows per tile", "", fint(r.tile_rows)), mrow("Scratch", "line", fint(r.scratch)), mrow("Share of the scratch ring", "%", fpct(r.scratch / ring))]));
    h += section("Arguments", table([["Word", ""], ["Value", "n"]], r.args.map(function (a) { return [esc(a[0]), esc(a[1])]; })));
    function rng(dir, x) { return [dir, "ch" + x.ch, "0x" + x.off.toString(16), fint(x.bytes), esc(x.region), x.rows ? x.rows[0] + " of " + x.rows[1] : ""]; }
    var rr = r.rdr.map(function (x) { return rng("Read", x); }).concat(r.wrr.map(function (x) { return rng("Write", x); }));
    h += section("DRAM ranges", rr.length ? table([["Dir", ""], ["Channel", ""], ["Offset", "n"], ["Bytes", "n"], ["Region", ""], ["Row of stride", "n"]], rr) : '<div class="msg">None.</div>');
    if (n) {
      var fg = n.grp >= 0 ? D.fusion[n.grp] : null;
      var nrows = [["Node", esc(n.name)], ["Op", esc(n.op)], ["Type", esc(n.type)], ["Shape", esc(n.ne.join(" x "))], ["Placement", esc(n.place)]];
      if (fg) nrows.push(["Fusion group", esc(fg.kind) + ", " + fg.members.length + " nodes"]);
      n.src.forEach(function (s, k) { nrows.push(["Source " + k, esc(s[0]) + (s[1] >= 0 ? " (node " + s[1] + ")" : "")]); });
      h += section("Graph node", table([["Field", ""], ["Value", ""]], nrows.map(function (x) { return [esc(x[0]), x[1]]; })));
    }
    RUNS.forEach(function (run, ri) {
      var refs = recIdx[ri][r.i] || [], ops = [];
      refs.forEach(function (ref) { var row = run.rows[ref[0]]; if (row.unit >= 0) ops.push({ ref: ref, row: row, o: row.ops[ref[1]] }); });
      ops.sort(function (a, b) { return a.o[0] - b.o[0]; });
      if (!ops.length) return;
      var tr = ops.map(function (x) {
        var o = x.o, un = D.units[x.row.unit], on = st.op && st.op.run === ri && st.op.row === x.ref[0] && st.op.idx === x.ref[1];
        return { cells: [esc(opId(run.tag, r.i, un, o[8])), esc(un), esc(opName(o[3])), fint(o[0]), fint(o[1]), fint(o[5]), fint(o[6]), o[7] ? fint(o[7]) : "-", o[7] ? fpct(o[7] / o[1]) : "-"],
                 cls: "clk" + (on ? " sel" : ""), attr: ' data-op="' + ri + "," + x.ref[0] + "," + x.ref[1] + '"' };
      });
      h += '<div class="sec" style="grid-column:1/-1">' + "<h4>Ops, " + esc(run.label) + "</h4>" + table([["Id", "id"], ["Unit", ""], ["Op", ""], ["Start", "n"], ["Cycles", "n"], ["Queue wait", "n"], ["Head wait", "n"], ["Ideal", "n"], ["Eff. %", "n"]], tr) + "</div>";
    });
    h += "</div>";
    p.innerHTML = h;
  }
  $("#p-details").addEventListener("click", function (e) {
    var t = e.target.closest("tr[data-op]"); if (!t) return;
    var a = t.getAttribute("data-op").split(",").map(Number); select(st.sel, { run: a[0], row: a[1], idx: a[2] }, "table");
    var o = RUNS[a[0]].rows[a[1]].ops[a[2]], w = view.b - view.a;
    if (o[0] < view.a || o[0] + o[1] > view.b) { view.a = o[0] + o[1] / 2 - w / 2; view.b = view.a + w; clampView(); draw(); }
  });

  // ---------------------------------------------------------------- start
  readColors(); themeLabel();
  $("#legend").innerHTML = D.groups.map(function (g) { return '<span style="margin-left:10px"><i class="' + gcls(g.id) + '"></i>' + esc(g.label) + "</span>"; }).join("");
  buildGeo();
  renderSummary(); renderDetails(); renderTree();
  // the longest record is selected at the start
  var best = 0; R.forEach(function (r) { if (r.cost > R[best].cost) best = r.i; });
  if (R.length) { st.sel = best; st.expanded["L" + M.launch] = true; st.expanded["y" + R[best].layer] = true; statusSel(); renderTree(); renderDetails(); }
  setTab("summary");
  drawAll();
  window.__prof = { select: select, setTab: setTab, view: view, draw: draw, zoomSel: zoomSel };
})();
