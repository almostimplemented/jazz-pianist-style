/* One prompt, twelve pianists — live style-swap player.
 *
 * Every continuation shares an opening prompt and then diverges, so playback
 * keeps a single clock: switching pianist mid-performance simply changes which
 * note stream feeds the scheduler from that moment on.
 */
(() => {
  "use strict";

  const DATA_URL = "data/aint_misbehavin.json";
  const LOOKAHEAD_MS = 150;   // schedule this far ahead of the playhead
  const WINDOW_MS = 8000;     // piano-roll time span
  const PLAYHEAD_FRAC = 0.28; // playhead position within that span

  const el = {
    canvas: document.getElementById("roll"),
    play: document.getElementById("play"),
    restart: document.getElementById("restart"),
    seek: document.getElementById("seek"),
    time: document.getElementById("time"),
    current: document.getElementById("current"),
    artists: document.getElementById("artists"),
    status: document.getElementById("status"),
  };
  const ctx = el.canvas.getContext("2d");
  const css = getComputedStyle(document.documentElement);
  const COLOR = {
    prompt: css.getPropertyValue("--prompt").trim() || "#6f8fa8",
    gen: css.getPropertyValue("--accent").trim() || "#e0a33e",
    line: css.getPropertyValue("--line").trim() || "#2e2823",
    dim: css.getPropertyValue("--ink-dim").trim() || "#a89e91",
  };

  let data = null;         // parsed payload
  let active = 0;          // index into data.artists
  let sampler = null;
  let playing = false;
  let startedAt = 0;       // Tone context time when playback (re)started
  let offsetMs = 0;        // position in the piece at that moment
  let scheduledMs = 0;     // notes strictly before this are already scheduled
  let durationMs = 1;      // longest continuation: one shared timeline for all

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const fmt = (ms) => {
    const s = Math.max(0, Math.floor(ms / 1000));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };
  const midiToNote = (p) => {
    const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    return names[p % 12] + (Math.floor(p / 12) - 1);
  };

  function nowMs() {
    if (!playing) return offsetMs;
    return offsetMs + (Tone.now() - startedAt) * 1000;
  }

  /* ---------- audio ---------- */

  async function initAudio() {
    if (sampler) return;
    el.status.textContent = "Loading piano samples…";
    const base = "https://tonejs.github.io/audio/salamander/";
    sampler = new Tone.Sampler({
      urls: {
        A0: "A0.mp3", C1: "C1.mp3", "D#1": "Ds1.mp3", "F#1": "Fs1.mp3",
        A1: "A1.mp3", C2: "C2.mp3", "D#2": "Ds2.mp3", "F#2": "Fs2.mp3",
        A2: "A2.mp3", C3: "C3.mp3", "D#3": "Ds3.mp3", "F#3": "Fs3.mp3",
        A3: "A3.mp3", C4: "C4.mp3", "D#4": "Ds4.mp3", "F#4": "Fs4.mp3",
        A4: "A4.mp3", C5: "C5.mp3", "D#5": "Ds5.mp3", "F#5": "Fs5.mp3",
        A5: "A5.mp3", C6: "C6.mp3", "D#6": "Ds6.mp3", "F#6": "Fs6.mp3",
        A6: "A6.mp3", C7: "C7.mp3", "D#7": "Ds7.mp3", "F#7": "Fs7.mp3",
        A7: "A7.mp3", C8: "C8.mp3",
      },
      baseUrl: base,
      release: 1.2,
    }).toDestination();
    sampler.volume.value = -4;
    await Tone.loaded();
    el.status.textContent = "";
  }

  /* Schedule any notes of the active stream that fall inside the lookahead. */
  function pump() {
    if (!playing || !sampler) return;
    const cur = nowMs();
    const until = cur + LOOKAHEAD_MS;
    const notes = data.artists[active].notes;
    const t0 = Tone.now();

    for (let i = 0; i < notes.length; i += 4) {
      const start = notes[i];
      if (start < scheduledMs) continue;
      if (start >= until) break;
      const dur = notes[i + 1], pitch = notes[i + 2], vel = notes[i + 3];
      const when = t0 + Math.max(0, (start - cur) / 1000);
      try {
        sampler.triggerAttackRelease(
          midiToNote(pitch), Math.max(0.05, dur / 1000), when, clamp(vel / 127, 0.05, 1));
      } catch (e) { /* a note outside the sampled range is not worth stopping for */ }
    }
    scheduledMs = Math.max(scheduledMs, until);

    if (cur >= durationMs) stop(true);
  }

  async function play() {
    await Tone.start();
    await initAudio();
    playing = true;
    startedAt = Tone.now();
    scheduledMs = offsetMs;
    el.play.innerHTML = "&#10073;&#10073;";
    el.play.setAttribute("aria-label", "Pause");
  }

  function stop(reachedEnd = false) {
    if (playing) offsetMs = reachedEnd ? durationMs : nowMs();
    playing = false;
    if (sampler) sampler.releaseAll();
    el.play.innerHTML = "&#9654;";
    el.play.setAttribute("aria-label", "Play");
  }

  function seekTo(ms) {
    const wasPlaying = playing;
    if (wasPlaying) stop();
    offsetMs = clamp(ms, 0, durationMs);
    if (wasPlaying) play();
    draw();
  }

  /* ---------- artist swap ---------- */

  function setArtist(idx) {
    active = idx;
    const a = data.artists[idx];
    el.current.textContent = a.name;
    [...el.artists.children].forEach((b, i) =>
      b.setAttribute("aria-pressed", String(i === idx)));
    // Hand over immediately: keep whatever is already sounding, take new notes
    // from this stream starting now.
    if (playing) {
      scheduledMs = nowMs();
    }
    draw();
  }

  /* ---------- drawing ---------- */

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = el.canvas.clientWidth, h = el.canvas.clientHeight;
    el.canvas.width = Math.round(w * dpr);
    el.canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function draw() {
    if (!data) return;
    const w = el.canvas.clientWidth, h = el.canvas.clientHeight;
    const cur = nowMs();
    const from = cur - WINDOW_MS * PLAYHEAD_FRAC;
    const to = from + WINDOW_MS;
    const notes = data.artists[active].notes;

    ctx.clearRect(0, 0, w, h);

    // pitch range: fixed piano span keeps the view stable while swapping
    const LO = 21, HI = 108;
    const yOf = (p) => h - ((p - LO) / (HI - LO)) * (h - 8) - 4;
    const xOf = (ms) => ((ms - from) / WINDOW_MS) * w;

    // octave guides
    ctx.strokeStyle = COLOR.line;
    ctx.lineWidth = 1;
    for (let p = 24; p <= HI; p += 12) {
      const y = Math.round(yOf(p)) + 0.5;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // branch marker: where the human prompt ends and generation begins
    const bx = xOf(data.branch_ms);
    if (bx > -20 && bx < w + 20) {
      ctx.save();
      ctx.strokeStyle = COLOR.dim;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(bx, 0); ctx.lineTo(bx, h); ctx.stroke();
      ctx.restore();
      ctx.fillStyle = COLOR.dim;
      ctx.font = "11px ui-monospace, monospace";
      ctx.fillText("model takes over", bx + 6, 14);
    }

    // notes
    const noteH = Math.max(2.5, (h - 8) / (HI - LO) * 1.6);
    for (let i = 0; i < notes.length; i += 4) {
      const start = notes[i], dur = notes[i + 1], pitch = notes[i + 2], vel = notes[i + 3];
      if (start + dur < from) continue;
      if (start > to) break;
      const x = xOf(start), wpx = Math.max(2, (dur / WINDOW_MS) * w);
      const y = yOf(pitch);
      const isPrompt = start < data.branch_ms;
      const played = start <= cur;
      ctx.globalAlpha = played ? 1 : 0.42;
      ctx.fillStyle = isPrompt ? COLOR.prompt : COLOR.gen;
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(x, y - noteH / 2, wpx, noteH, Math.min(2, noteH / 2));
      } else {  // older Safari
        ctx.rect(x, y - noteH / 2, wpx, noteH);
      }
      ctx.fill();
      // velocity as a subtle brightness cue on the played side
      if (played && vel > 90) {
        ctx.globalAlpha = 0.25;
        ctx.fillStyle = "#fff";
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;

    // playhead
    const px = Math.round(w * PLAYHEAD_FRAC) + 0.5;
    ctx.strokeStyle = "#fff";
    ctx.globalAlpha = 0.8;
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function frame() {
    pump();
    draw();
    const cur = nowMs();
    const ended = data && cur > data.artists[active].duration_ms;
    el.time.textContent = `${fmt(cur)} / ${fmt(durationMs)}${ended ? " · ended" : ""}`;
    if (document.activeElement !== el.seek) el.seek.value = String(Math.round(cur));
    requestAnimationFrame(frame);
  }

  /* ---------- wiring ---------- */

  el.play.addEventListener("click", () => (playing ? stop() : play()));
  el.restart.addEventListener("click", () => seekTo(0));
  el.seek.addEventListener("input", (e) => seekTo(Number(e.target.value)));
  window.addEventListener("resize", () => { resize(); draw(); });
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && e.target === document.body) {
      e.preventDefault();
      playing ? stop() : play();
    }
  });

  fetch(DATA_URL)
    .then((r) => r.json())
    .then((payload) => {
      data = payload;
      data.artists.forEach((a, i) => {
        const b = document.createElement("button");
        b.textContent = a.name;
        b.setAttribute("aria-pressed", "false");
        b.addEventListener("click", () => setArtist(i));
        el.artists.appendChild(b);
      });
      durationMs = Math.max(...data.artists.map((a) => a.duration_ms));
      el.seek.max = String(durationMs);
      resize();
      setArtist(0);
      el.status.textContent = "";
      requestAnimationFrame(frame);
    })
    .catch((err) => {
      el.status.textContent = "Could not load the performance data.";
      console.error(err);
    });
})();
