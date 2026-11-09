(function () {
'use strict';
var D = JSON.parse(document.getElementById('data').textContent);
function $(s, r) { return (r || document).querySelector(s); }
function $$(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }
function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
function fn(n) { return Math.round(n).toLocaleString('en-US'); }
function fc(n) { return n >= 1e6 ? (n / 1e6).toFixed(2) + ' M' : n >= 1e3 ? (n / 1e3).toFixed(1) + ' K' : String(Math.round(n)); }
function pc(x) { return (100 * x).toFixed(1) + ' %'; }
function us(c) { return (c * 1e6 / D.f).toFixed(2) + ' µs'; }
function hex(n) { return '0x' + Number(n).toString(16); }

// ---- theme (the stored choice is a convenience: every access is guarded)
var root = document.documentElement;
function setTheme(t) { if (t) { root.setAttribute('data-theme', t); } else { root.removeAttribute('data-theme'); } }
try { var st = localStorage.getItem('tsr-prof-theme'); if (st === 'light' || st === 'dark') { setTheme(st); } } catch (e) { }
(function () { var m = /[?&#]theme=(light|dark)/.exec(location.href); if (m) { setTheme(m[1]); } })();
$('#theme').addEventListener('click', function () {
  var cur = root.getAttribute('data-theme');
  if (!cur) { cur = (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light'; }
  var nx = cur === 'dark' ? 'light' : 'dark';
  setTheme(nx);
  try { localStorage.setItem('tsr-prof-theme', nx); } catch (e) { }
});

// ---- the detail drawer
var detail = $('#detail');
function openDetail(title, html) {
  $('#dtitle').textContent = title;
  $('#dbody').innerHTML = html;
  detail.hidden = false;
}
$('#dclose').addEventListener('click', function () { detail.hidden = true; });
function kv(pairs) {
  return '<div class="kv">' + pairs.map(function (p) { return '<b>' + esc(p[0]) + '</b><span>' + p[1] + '</span>'; }).join('') + '</div>';
}
function lk(attr, val, text) { return '<a class="lk" data-' + attr + '="' + esc(val) + '">' + esc(text) + '</a>'; }
function rangeRows(list) {
  if (!list || !list.length) { return '<tr><td colspan="4">none</td></tr>'; }
  return list.map(function (r) {
    return '<tr><td>ch' + r.ch + '</td><td>' + esc(r.region) + '</td><td class="mono">' + hex(r.off) + '</td><td class="num">' + fn(r.bytes) + ' B' + (r.rows ? ' (rows of ' + r.rows[0] + ' B every ' + r.rows[1] + ' B)' : '') + '</td></tr>';
  }).join('');
}
function nodeRows(ids) {
  return ids.map(function (i) {
    var n = D.nodes[i];
    return '<tr><td>' + lk('node', i, n.name) + '</td><td>' + esc(n.op) + '</td><td>' + esc(n.place) + (n.host >= 0 ? ' into ' + lk('node', n.host, D.nodes[n.host].name) : '') + '</td></tr>';
  }).join('');
}
function showRecord(i, nodes) {
  var r = D.records[i];
  if (!r) { return; }
  var eff = r.ideal && r.cost ? pc(r.ideal / r.cost) : '–';
  var simtxt = r.sim ? fn(r.sim.cost) + ' (' + (r.cost ? (r.sim.cost / r.cost).toFixed(3) : '–') + ' of the card)' : '–';
  var busy = Object.keys(r.busy).map(function (u) { return u + ' ' + fn(r.busy[u]); }).join(', ') || '–';
  var scr = r.scratch ? fn(r.scratch) + ' lines (' + (r.scratch * 64 / 1024).toFixed(0) + ' KiB), ' + pc(r.scratch / D.ring_lines) + ' of the ' + fn(D.ring_lines) + '-line ring budget' : '–';
  var links = [];
  if (r.slice) { links.push(lk('slice', r.slice, 'longest slice')); }
  if (r.node >= 0) { links.push(lk('node', r.node, 'graph node')); }
  var cols = '<div>' + kv([
    ['record', r.i + ' (op-list index)'], ['kind', esc(r.kn)], ['name', esc(r.name)], ['layer / group', esc(r.layer) + ' / ' + esc(D.kind_labels[r.group] || r.group)],
    ['shape', esc(r.shape || '–')], ['card cycles', fn(r.cost) + ' (' + pc(r.cost / D.span) + ' of the launch)'], ['ideal cycles', r.ideal ? fn(r.ideal) : '–'], ['efficiency', eff],
    ['simulator cycles', simtxt], ['ops', r.n_ops + '; busy cycles: ' + esc(busy)], ['dependencies', r.dep.filter(function (d) { return d >= 0; }).map(function (d) { return lk('rec', d, '#' + d); }).join(' ') || 'none'],
    ['traffic', 'reads ' + fn(r.rd) + ' B, writes ' + fn(r.wr) + ' B' + (r.w ? ', weights ' + fn(r.w) + ' B' : '')], ['open', links.join(' · ')]]) + '</div>';
  cols += '<div><h3>Argument words</h3><table><tbody>' + r.args.map(function (a) { return '<tr><td>' + esc(a[0]) + '</td><td class="mono">' + esc(a[1]) + '</td></tr>'; }).join('') + '</tbody></table></div>';
  cols += '<div><h3>Memory</h3><table><thead><tr><th></th><th>Region</th><th>Offset</th><th class="num">Bytes</th></tr></thead><tbody><tr><td colspan="4"><b>reads</b></td></tr>' + rangeRows(r.rdr) +
    '<tr><td colspan="4"><b>writes</b></td></tr>' + rangeRows(r.wrr) + '</tbody></table><p class="note">SPM tile plan: ' + (r.tiles ? r.tiles + ' tile(s) of ' + r.tile_rows + ' row(s); ' : '') + scr + '</p></div>';
  if (nodes && nodes.length) { cols += '<div><h3>Graph nodes of this box</h3><table><tbody>' + nodeRows(nodes) + '</tbody></table></div>'; }
  openDetail('Record ' + r.i + ': ' + r.kn + ' ' + r.name, '<div class="cols">' + cols + '</div>');
}
function showNode(i) {
  var n = D.nodes[i];
  if (!n) { return; }
  var gr = n.grp >= 0 ? D.groups_g[n.grp] : null;
  var recs = n.recs.map(function (r) { return lk('rec', r, '#' + r + ' ' + D.records[r].kn); }).join(' ') || 'none';
  var cols = '<div>' + kv([['node', n.i], ['name', esc(n.name)], ['op', esc(n.op)], ['type / shape', esc(n.type) + ' [' + n.ne.join(' x ') + ']'], ['placement', esc(n.place) + (n.host >= 0 ? ' into ' + lk('node', n.host, D.nodes[n.host].name) : '')],
    ['layer / group', esc(n.layer) + ' / ' + esc(D.kind_labels[n.g] || n.g)], ['fusion group', gr ? esc(gr.kind) + ' (head ' + lk('node', gr.head, D.nodes[gr.head].name) + ')' : 'none'], ['records', recs], ['cycles', fn(n.cost)]]) + '</div>';
  cols += '<div><h3>Sources</h3><table><tbody>' + n.src.map(function (s) { return '<tr><td>' + (s[1] >= 0 ? lk('node', s[1], s[0]) : esc(s[0])) + '</td><td>' + esc(s[2]) + '</td></tr>'; }).join('') + '</tbody></table></div>';
  if (gr) { cols += '<div><h3>Group members</h3><table><tbody>' + nodeRows(gr.members) + '</tbody></table></div>'; }
  openDetail('Node ' + n.i + ': ' + n.op + ' ' + n.name, '<div class="cols">' + cols + '</div>');
}
function sliceId(tag, row) { return tag + ':' + (row[0] >= 0 ? 'r' + row[0] : 'setup') + '.' + D.units[row[1]] + row[9]; }
function showSlice(tag, k) {
  var row = D.ops[tag][k];
  var r = row[0] >= 0 ? D.records[row[0]] : null;
  var opn = D.opnames[String(row[6])] || ('op ' + hex(row[6]));
  var eff = row[8] && row[3] ? pc(row[8] / row[3]) : '–';
  var cols = '<div>' + kv([['slice id', '<span class="mono">' + esc(sliceId(tag, row)) + '</span>'], ['run', esc(D.units[row[1]]) + ' ' + esc(opn)], ['start', fn(row[2]) + ' cycles (' + us(row[2]) + ')'], ['cycles', fn(row[3]) + ' (' + us(row[3]) + ')'],
    ['queue wait / head wait', fn(row[4]) + ' / ' + fn(row[5]) + ' cycles'], ['push index', row[10]], ['ideal cycles', row[8] ? row[8].toFixed(1) : '–'], ['efficiency', eff],
    ['record', r ? lk('rec', r.i, '#' + r.i + ' ' + r.kn + ' ' + r.name) : 'launch setup'], ['node', r && r.node >= 0 ? lk('node', r.node, D.nodes[r.node].name) : '–']]) + '</div>';
  openDetail('Slice ' + sliceId(tag, row), '<div class="cols">' + cols + '</div>');
}
function showCpu(i) {
  var c = D.cpu[i];
  if (!c) { return; }
  openDetail('CPU node ' + c.name, '<div class="cols"><div>' + kv([['name', esc(c.name)], ['op', esc(c.op)], ['why', esc(c.why)]]) + '</div></div>');
}

// ---- the graph
var curLayer = null;
function selectLayer(l) {
  l = String(l);
  $$('#graph .gvl').forEach(function (d) { d.hidden = d.getAttribute('data-layer') !== l; });
  $$('#layerbtns button').forEach(function (b) { b.setAttribute('aria-pressed', b.getAttribute('data-layer') === l ? 'true' : 'false'); });
  $('#layername').textContent = l;
  curLayer = l;
}
function clearSel() { $$('.gn.sel').forEach(function (g) { g.classList.remove('sel'); }); }
function selectBox(g) {
  clearSel();
  g.classList.add('sel');
  var cpu = g.getAttribute('data-cpu');
  if (cpu !== null) { showCpu(Number(cpu)); return; }
  var recs = (g.getAttribute('data-rec') || '').split(' ').filter(Boolean).map(Number);
  var nodes = (g.getAttribute('data-nodes') || '').split(' ').filter(Boolean).map(Number);
  if (recs.length) { showRecord(recs[0], nodes); } else if (nodes.length) { showNode(nodes[0]); }
}
function goLayer(l) {
  selectLayer(l);
  var b = $('#layerbtns');
  if (b && b.scrollIntoView) { b.scrollIntoView(); }
}
function goRecord(i) {
  var r = D.records[i];
  if (!r) { return; }
  selectLayer(r.layer);
  var g = $('#graph .gvl:not([hidden]) .gn[data-rec~="' + i + '"]');
  if (g) { clearSel(); g.classList.add('sel'); if (g.scrollIntoView) { g.scrollIntoView({ block: 'center', inline: 'center' }); } }
  showRecord(i, g ? (g.getAttribute('data-nodes') || '').split(' ').filter(Boolean).map(Number) : null);
}
function goNode(i) {
  var n = D.nodes[i];
  if (!n) { return; }
  var l = n.layer;
  if (n.host >= 0) { l = D.nodes[n.host].layer; }
  selectLayer(l);
  var g = $('#graph .gvl:not([hidden]) .gn[data-nodes~="' + i + '"]');
  if (g) { clearSel(); g.classList.add('sel'); if (g.scrollIntoView) { g.scrollIntoView({ block: 'center', inline: 'center' }); } }
  showNode(i);
}
$('#graph').addEventListener('click', function (e) {
  var g = e.target.closest ? e.target.closest('.gn') : null;
  if (g) { selectBox(g); return; }
  var s = e.target.closest ? e.target.closest('.stage') : null;
  if (s) { goLayer(s.getAttribute('data-layer')); }
});
$('#layerbtns').addEventListener('click', function (e) {
  var b = e.target.closest ? e.target.closest('button') : null;
  if (b) { selectLayer(b.getAttribute('data-layer')); }
});
$('#summary').addEventListener('click', function (e) {
  var b = e.target.closest ? e.target.closest('g.bar') : null;
  if (b) { goLayer(b.getAttribute('data-layer')); }
});

// ---- the timeline
var vx0 = 0, vx1 = D.span;
var sliceMap = null;
function buildMap() {
  if (sliceMap) { return; }
  sliceMap = {};
  ['card', 'sim'].forEach(function (tag) {
    var ops = D.ops[tag];
    if (!ops) { return; }
    for (var k = 0; k < ops.length; k++) { sliceMap[sliceId(tag, ops[k])] = { tag: tag, k: k }; }
  });
}
function niceStep(r) {
  var e = Math.pow(10, Math.floor(Math.log(r) / Math.LN10)), m = r / e;
  return (m < 2 ? 0.2 : m < 5 ? 0.5 : 1) * e;
}
function drawAxis() {
  var ax = $('#axis'), a = vx0 * 1e6 / D.f, b = vx1 * 1e6 / D.f, step = niceStep((b - a) / 6), html = '';
  for (var t = Math.ceil(a / step) * step; t <= b; t += step) {
    var left = 100 * (t - a) / (b - a);
    html += '<i style="left:' + left.toFixed(3) + '%"></i><span style="left:' + left.toFixed(3) + '%">' + (step >= 1000 ? (t / 1000).toFixed(step >= 10000 ? 0 : 1) + ' ms' : (step < 1 ? t.toFixed(2) : t.toFixed(0)) + ' µs') + '</span>';
  }
  ax.innerHTML = html;
  $('#tlrange').textContent = us(vx0) + ' to ' + us(vx1) + ' (' + fn(vx1 - vx0) + ' cycles shown)';
}
function setView(a, b) {
  var w = b - a;
  if (w < 400) { var c = (a + b) / 2; a = c - 200; b = c + 200; w = 400; }
  if (w > D.span) { a = 0; b = D.span; w = D.span; }
  if (a < 0) { b -= a; a = 0; }
  if (b > D.span) { a -= b - D.span; b = D.span; }
  vx0 = Math.max(a, 0); vx1 = Math.min(b, D.span);
  $$('svg.trk').forEach(function (s) {
    var h = s.getAttribute('height');
    s.setAttribute('viewBox', vx0.toFixed(0) + ' 0 ' + (vx1 - vx0).toFixed(0) + ' ' + h);
  });
  drawAxis();
}
function clearHl() { $$('svg.trk rect.hl').forEach(function (r) { r.classList.remove('hl'); }); }
function selectSlice(tag, k, zoom) {
  clearHl();
  var el = document.getElementById(tag.charAt(0) + k);
  if (el) { el.classList.add('hl'); }
  var row = D.ops[tag][k];
  showSlice(tag, k);
  if (zoom) { var c = row[2] + row[3] / 2, w = Math.max(row[3] * 10, D.span / 400); setView(c - w / 2, c + w / 2); }
  var t = $('#timeline');
  if (t && t.scrollIntoView) { t.scrollIntoView(); }
}
function goSlice(id) {
  buildMap();
  var e = sliceMap[id];
  if (!e) { $('#tlinfo').textContent = 'No slice with the id ' + id; return; }
  selectSlice(e.tag, e.k, true);
}
$('#zin').addEventListener('click', function () { var c = (vx0 + vx1) / 2, w = (vx1 - vx0) / 3; setView(c - w / 2, c + w / 2); });
$('#zout').addEventListener('click', function () { var c = (vx0 + vx1) / 2, w = (vx1 - vx0) * 3; setView(c - w / 2, c + w / 2); });
$('#zreset').addEventListener('click', function () { setView(0, D.span); });
$('#slicego').addEventListener('click', function () { goSlice($('#slicesearch').value.trim()); });
$('#slicesearch').addEventListener('keydown', function (e) { if (e.key === 'Enter') { goSlice(this.value.trim()); } });
var tlEl = $('.tl'), drag = null, moved = false;
tlEl.addEventListener('mousedown', function (e) {
  var s = e.target.closest ? e.target.closest('svg.trk') : null;
  if (!s || e.button !== 0) { return; }
  drag = { x: e.clientX, a: vx0, b: vx1, w: s.getBoundingClientRect().width };
  moved = false;
});
window.addEventListener('mousemove', function (e) {
  if (!drag) { return; }
  var dx = e.clientX - drag.x;
  if (Math.abs(dx) > 3) { moved = true; }
  if (moved) { var d = -dx / drag.w * (drag.b - drag.a); setView(drag.a + d, drag.b + d); }
});
window.addEventListener('mouseup', function () { drag = null; });
tlEl.addEventListener('wheel', function (e) {
  if (!(e.ctrlKey || e.metaKey || e.altKey)) { return; }
  e.preventDefault();
  var s = e.target.closest ? e.target.closest('svg.trk') : null;
  var fr = 0.5;
  if (s) { var bb = s.getBoundingClientRect(); fr = (e.clientX - bb.left) / bb.width; }
  var w = vx1 - vx0, c = vx0 + fr * w, nw = w * (e.deltaY > 0 ? 1.4 : 1 / 1.4);
  setView(c - fr * nw, c + (1 - fr) * nw);
}, { passive: false });
tlEl.addEventListener('click', function (e) {
  if (moved) { moved = false; return; }
  var t = e.target, s = t.closest ? t.closest('svg.trk') : null;
  if (!s || !t.getAttribute) { return; }
  var k = t.getAttribute('data-k');
  if (k !== null) { selectSlice(s.getAttribute('data-tag'), Number(k), false); return; }
  var r = t.getAttribute('data-rec');
  if (r !== null) { showRecord(Number(r)); }
});
tlEl.addEventListener('mouseover', function (e) {
  var t = e.target, s = t.closest ? t.closest('svg.trk') : null;
  if (!s || !t.getAttribute) { return; }
  var k = t.getAttribute('data-k'), r = t.getAttribute('data-rec'), info = $('#tlinfo');
  if (k !== null) {
    var row = D.ops[s.getAttribute('data-tag')][Number(k)], rec = row[0] >= 0 ? D.records[row[0]] : null;
    info.textContent = D.units[row[1]] + ' ' + (D.opnames[String(row[6])] || '') + '  ' + (rec ? rec.kn + ' ' + rec.name : 'launch setup') + '  ' + fn(row[3]) + ' cycles (' + us(row[3]) + ')' + (row[8] ? ', ideal ' + row[8].toFixed(0) : '');
  } else if (r !== null) {
    var rr = D.records[Number(r)];
    info.textContent = 'record ' + rr.i + '  ' + rr.kn + ' ' + rr.name + '  ' + fn(rr.cost) + ' cycles';
  }
});

// ---- links in tables, drawer and boxes
document.addEventListener('click', function (e) {
  var a = e.target.closest ? e.target.closest('a.lk') : null;
  if (!a) { return; }
  e.preventDefault();
  var v;
  if ((v = a.getAttribute('data-slice')) !== null) { goSlice(v); } else if ((v = a.getAttribute('data-rec')) !== null) { goRecord(Number(v)); } else if ((v = a.getAttribute('data-node')) !== null) { goNode(Number(v)); }
});
$('#rectab').addEventListener('click', function (e) {
  if (e.target.closest && e.target.closest('a.lk')) { return; }
  var tr = e.target.closest ? e.target.closest('tr.cl') : null;
  if (!tr) { return; }
  $$('#rectab tr.sel').forEach(function (r) { r.classList.remove('sel'); });
  tr.classList.add('sel');
  showRecord(Number(tr.getAttribute('data-rec')));
});
function filterRecords() {
  var q = $('#recfilter').value.toLowerCase(), g = $('#grpfilter').value;
  $$('#rectab tbody tr').forEach(function (tr) {
    var ok = (!g || tr.getAttribute('data-group') === g) && (!q || tr.textContent.toLowerCase().indexOf(q) >= 0);
    tr.style.display = ok ? '' : 'none';
  });
}
$('#recfilter').addEventListener('input', filterRecords);
$('#grpfilter').addEventListener('change', filterRecords);

function fromHash() {
  var m = /^#(slice|rec|node|layer)=(.+)$/.exec(decodeURIComponent(location.hash));
  if (!m) { return; }
  if (m[1] === 'slice') { goSlice(m[2]); } else if (m[1] === 'rec') { goRecord(Number(m[2])); } else if (m[1] === 'node') { goNode(Number(m[2])); } else { goLayer(m[2]); }
}
window.addEventListener('hashchange', fromHash);

// ---- start
var first = D.layers.indexOf(0) >= 0 ? '0' : String(D.layers[0]);
selectLayer(first);
setView(0, D.span);
if (location.hash) { fromHash(); }
})();
