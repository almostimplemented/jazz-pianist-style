/* Where the style lives — characteristic regions of real performances.
 *
 * For one held-out recording per pianist: the classifier's within-track
 * z-scored margin across the whole performance, and fifteen seconds from its
 * highest and lowest points, each playable.
 */
(() => {
  "use strict";

  const DATA_URL = "data/characteristic.json";
  const LOOKAHEAD_VISIBLE_MS = 150;
  const LOOKAHEAD_HIDDEN_MS = 3000;

  const el = (id) => document.getElementById(id);
  const nodes = {
    list: el("rg-list"), title: el("rg-title"), curve: el("rg-curve"),
    note: el("rg-note"), status: el("rg-status"),
  };
  const clips = ["peak", "trough"].map((which) => ({
    which,
    play: el(`rg-${which}-play`),
    cap: el(`rg-${which}-cap`),
    canvas: el(`rg-${which}-roll`),
  }));

  const css = getComputedStyle(document.documentElement);
  const COLOR = {
    peak: css.getPropertyValue("--accent").trim() || "#e0a33e",
    trough: css.getPropertyValue("--miss").trim() || "#7d5a72",
    line: css.getPropertyValue("--line").trim() || "#2e2823",
    dim: css.getPropertyValue("--ink-dim").trim() || "#a89e91",
  };

  let tracks = [], active = 0;
  let sampler = null, playingClip = null, startedAt = 0, scheduledMs = 0;

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const signed = (z) => (z >= 0 ? "+" : "−") + Math.abs(z).toFixed(1);
  const track = () => tracks[active];
  const notesOf = (c) => track()[`${c.which}_notes`];
  const durOf = (c) => track()[`${c.which}_duration_ms`];
  const nowMs = () => (playingClip ? (Tone.now() - startedAt) * 1000 : 0);

  /* ---------- curve ---------- */

  const cctx = nodes.curve.getContext("2d");
  function sizeCurve() {
    const dpr = window.devicePixelRatio || 1;
    nodes.curve.width = Math.round(nodes.curve.clientWidth * dpr);
    nodes.curve.height = Math.round(nodes.curve.clientHeight * dpr);
    cctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  const fmt = (sec) => `${Math.floor(sec / 60)}:${String(Math.floor(sec % 60)).padStart(2, "0")}`;
  const EXCERPT_S = () => (track().peak_duration_ms || 15000) / 1000;

  // z at a moment of the performance, interpolated between window centres
  function zAt(t, sec) {
    const ts = t.times;
    if (sec <= ts[0]) return t.curve[0];
    for (let i = 1; i < ts.length; i++) {
      if (sec <= ts[i]) {
        const f = (sec - ts[i - 1]) / Math.max(1e-6, ts[i] - ts[i - 1]);
        return t.curve[i - 1] + f * (t.curve[i] - t.curve[i - 1]);
      }
    }
    return t.curve[t.curve.length - 1];
  }

  function drawCurve() {
    const t = track();
    const w = nodes.curve.clientWidth, h = nodes.curve.clientHeight;
    const padT = 18, padB = 20;
    const zMax = Math.max(2.2, ...t.curve.map(Math.abs)) + 0.2;
    const yOf = (z) => padT + (1 - (clamp(z, -zMax, zMax) + zMax) / (2 * zMax)) * (h - padT - padB);
    const span = Math.max(1, t.duration_s);
    const xOf = (sec) => (sec / span) * (w - 2) + 1;

    cctx.clearRect(0, 0, w, h);

    // the two excerpt regions, shaded as in the paper's figure
    for (const c of clips) {
      const x0 = xOf(t[`${c.which}_start_s`]), x1 = xOf(t[`${c.which}_start_s`] + EXCERPT_S());
      cctx.globalAlpha = playingClip === c ? 0.22 : 0.1;
      cctx.fillStyle = c.which === "peak" ? COLOR.peak : COLOR.trough;
      cctx.fillRect(x0, 0, x1 - x0, h - padB + 4);
    }
    cctx.globalAlpha = 1;

    cctx.strokeStyle = COLOR.line;
    cctx.lineWidth = 1;
    for (const z of [-2, -1, 1, 2]) {
      const y = Math.round(yOf(z)) + 0.5;
      cctx.setLineDash([2, 4]);
      cctx.beginPath(); cctx.moveTo(0, y); cctx.lineTo(w, y); cctx.stroke();
    }
    cctx.setLineDash([]);
    const y0 = Math.round(yOf(0)) + 0.5;
    cctx.beginPath(); cctx.moveTo(0, y0); cctx.lineTo(w, y0); cctx.stroke();

    // fill above zero in gold, below in mauve
    const first = t.times[0], last = t.times[t.times.length - 1];
    for (const [sign, color] of [[1, COLOR.peak], [-1, COLOR.trough]]) {
      cctx.beginPath();
      cctx.moveTo(xOf(first), y0);
      t.times.forEach((sec, i) => {
        const z = t.curve[i];
        cctx.lineTo(xOf(sec), sign * z > 0 ? yOf(z) : y0);
      });
      cctx.lineTo(xOf(last), y0);
      cctx.closePath();
      cctx.globalAlpha = 0.28;
      cctx.fillStyle = color;
      cctx.fill();
    }
    cctx.globalAlpha = 1;
    cctx.strokeStyle = COLOR.dim;
    cctx.beginPath();
    t.times.forEach((sec, i) => {
      const x = xOf(sec), y = yOf(t.curve[i]);
      i ? cctx.lineTo(x, y) : cctx.moveTo(x, y);
    });
    cctx.stroke();

    cctx.font = "11px ui-monospace, monospace";
    for (const c of clips) {
      const x = xOf(t[`${c.which}_time_s`]), y = yOf(t[`${c.which}_z`]);
      cctx.fillStyle = c.which === "peak" ? COLOR.peak : COLOR.trough;
      cctx.beginPath(); cctx.arc(x, y, 4.5, 0, Math.PI * 2); cctx.fill();
      const label = `${c.which === "peak" ? "most" : "least"} characteristic`;
      const tw = cctx.measureText(label).width;
      const lx = clamp(x - tw / 2, 2, w - tw - 2);
      cctx.fillStyle = COLOR.dim;
      cctx.fillText(label, lx, c.which === "peak" ? Math.max(11, y - 10) : Math.min(h - padB - 2, y + 18));
    }

    // time axis
    cctx.fillStyle = COLOR.dim;
    const step = span > 240 ? 60 : 30;
    for (let sec = 0; sec <= span; sec += step) {
      const label = fmt(sec), tw = cctx.measureText(label).width;
      cctx.fillText(label, clamp(xOf(sec) - tw / 2, 1, w - tw - 1), h - 4);
    }

    // live playhead while an excerpt plays
    if (playingClip) {
      const sec = t[`${playingClip.which}_start_s`] + nowMs() / 1000;
      const x = xOf(sec), y = yOf(zAt(t, sec));
      cctx.strokeStyle = "#fff";
      cctx.globalAlpha = 0.7;
      cctx.beginPath(); cctx.moveTo(Math.round(x) + 0.5, 0); cctx.lineTo(Math.round(x) + 0.5, h - padB + 4); cctx.stroke();
      cctx.globalAlpha = 1;
      cctx.fillStyle = "#fff";
      cctx.beginPath(); cctx.arc(x, y, 5, 0, Math.PI * 2); cctx.fill();
    }
  }

  /* ---------- excerpts ---------- */

  const rolls = new Map();

  // Both excerpts share one pitch range so their registers can be compared.
  function pitchRange() {
    let lo = 108, hi = 21;
    for (const c of clips) {
      const n = notesOf(c);
      for (let i = 2; i < n.length; i += 4) { lo = Math.min(lo, n[i]); hi = Math.max(hi, n[i]); }
    }
    return hi < lo ? [21, 108] : [Math.max(21, lo - 3), Math.min(108, hi + 3)];
  }
  function drawClips() {
    for (const c of clips) {
      const playingThis = playingClip === c;
      rolls.get(c).draw({
        streams: [{ notes: notesOf(c), color: c.which === "peak" ? COLOR.peak : "#b58aa6" }],
        nowMs: playingThis ? nowMs() : 0,
        fromMs: 0,
        windowMs: Math.max(1000, durOf(c)),
        pitchRange: pitchRange(),
      });
    }
  }

  function pump() {
    if (!playingClip || !sampler) return;
    const cur = nowMs();
    const until = cur + (document.hidden ? LOOKAHEAD_HIDDEN_MS : LOOKAHEAD_VISIBLE_MS);
    const t0 = Tone.now();
    const notes = notesOf(playingClip);
    for (let i = 0; i < notes.length; i += 4) {
      const start = notes[i];
      if (start < scheduledMs) continue;
      if (start >= until) break;
      try {
        sampler.triggerAttackRelease(midiToNote(notes[i + 2]),
          Math.max(0.05, notes[i + 1] / 1000),
          t0 + Math.max(0, (start - cur) / 1000),
          clamp(notes[i + 3] / 127, 0.05, 1));
      } catch (e) { /* out-of-range note */ }
    }
    scheduledMs = Math.max(scheduledMs, until);
    if (cur >= durOf(playingClip) + 400) stop();
  }

  async function play(c) {
    await Tone.start();
    if (!sampler) {
      nodes.status.textContent = "Loading piano samples…";
      sampler = await getSampler();
      nodes.status.textContent = "";
    }
    document.dispatchEvent(new CustomEvent("demo:play", { detail: "regions" }));
    stop();
    playingClip = c;
    startedAt = Tone.now();
    scheduledMs = 0;
    c.play.innerHTML = "&#10073;&#10073;";
    c.play.setAttribute("aria-label", "Pause");
  }

  function stop() {
    if (!playingClip) return;
    playingClip.play.innerHTML = "&#9654;";
    playingClip.play.setAttribute("aria-label", "Play");
    playingClip = null;
    if (sampler) sampler.releaseAll();
  }

  /* ---------- selection ---------- */

  function select(idx) {
    stop();
    active = idx;
    const t = track();
    [...nodes.list.children].forEach((b, i) => b.setAttribute("aria-pressed", String(i === idx)));
    nodes.title.innerHTML = `<b>${t.artist}</b><span class="sub">&ldquo;${t.title}&rdquo;</span>`;
    for (const c of clips) {
      const z = t[`${c.which}_z`];
      const t0 = t[`${c.which}_start_s`];
      c.cap.innerHTML = `${c.which === "peak" ? "Most" : "Least"} characteristic` +
        `<span class="z">${fmt(t0)}&ndash;${fmt(t0 + EXCERPT_S())} &middot; z ${signed(z)}</span>`;
    }
    nodes.note.textContent = t.peak_z < 0.8
      ? `No passage stands out as especially characteristic: the classifier recognizes ` +
        `${t.artist} about as surely everywhere. Its evidence is spread through the ` +
        `playing rather than concentrated in particular passages.`
      : "";
    drawCurve();
    drawClips();
  }

  for (const c of clips) {
    c.play.addEventListener("click", () => (playingClip === c ? stop() : play(c)));
    rolls.set(c, makeRoll(c.canvas));
  }
  document.addEventListener("demo:play", (e) => { if (e.detail !== "regions") stop(); });
  window.addEventListener("resize", () => { if (tracks.length) { sizeCurve(); drawCurve(); drawClips(); } });

  fetch(DATA_URL, { cache: "no-cache" })
    .then((r) => r.json())
    .then((payload) => {
      tracks = payload.tracks;
      document.getElementById("regions").hidden = false;
      for (const r of rolls.values()) r.resize();  // sized while hidden
      tracks.forEach((t, i) => {
        const b = document.createElement("button");
        b.textContent = t.artist;
        b.addEventListener("click", () => select(i));
        nodes.list.appendChild(b);
      });
      sizeCurve();
      select(Math.max(0, tracks.findIndex((t) => t.artist === "Art Tatum")));
      setInterval(pump, 50);
      (function frame() {
        if (playingClip) { drawClips(); drawCurve(); }
        requestAnimationFrame(frame);
      })();
      let lastLit = null;  // repaint once more when playback stops
      setInterval(() => { if (lastLit !== playingClip) { lastLit = playingClip; drawCurve(); } }, 100);
    })
    .catch((e) => console.error(e));  // the section stays hidden
})();
