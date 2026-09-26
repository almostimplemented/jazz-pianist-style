/* One prompt, twelve pianists — live style-swap player.
 *
 * Every continuation shares an opening prompt and then diverges, so playback
 * keeps a single clock: switching pianist mid-performance simply changes which
 * note stream feeds the scheduler from that moment on.
 */
(() => {
  "use strict";

  const DATA_URL = "data/aint_misbehavin.json";
  const LOOKAHEAD_VISIBLE_MS = 150;  // responsive swaps while watching
  const LOOKAHEAD_HIDDEN_MS = 3000;  // background tabs clamp timers to ~1s
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
  let durationMs = 1;      // the selected take's own length

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const fmt = (ms) => {
    const s = Math.max(0, Math.floor(ms / 1000));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };

  function nowMs() {
    if (!playing) return offsetMs;
    return offsetMs + (Tone.now() - startedAt) * 1000;
  }

  /* ---------- audio ---------- */

  async function initAudio() {
    if (sampler) return;
    el.status.textContent = "Loading piano samples…";
    sampler = await getSampler();
    el.status.textContent = "";
  }

  /* Schedule any notes of the active stream that fall inside the lookahead. */
  function pump() {
    if (!playing || !sampler) return;
    const cur = nowMs();
    const until = cur + (document.hidden ? LOOKAHEAD_HIDDEN_MS : LOOKAHEAD_VISIBLE_MS);
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
    document.dispatchEvent(new CustomEvent("demo:play", { detail: "shared-prompt" }));
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
    // Each take has its own length: keep the listener at the same moment in
    // the music, and rescale the timeline to the take they are now hearing.
    durationMs = a.duration_ms;
    el.seek.max = String(durationMs);
    const at = nowMs();
    if (at >= durationMs) {          // this take is already over by that point
      stop();
      offsetMs = durationMs;
    } else if (playing) {
      scheduledMs = at;
    }
    draw();
  }

  /* ---------- drawing ---------- */

  const roll = makeRoll(el.canvas);

  function resize() { roll.resize(); }

  function draw() {
    if (!data) return;
    const a = data.artists[active];
    roll.draw({
      streams: [
        { notes: a.notes.slice(0, 4 * data.shared_notes), color: COLOR.prompt },
        { notes: a.notes.slice(4 * data.shared_notes), color: COLOR.gen },
      ],
      nowMs: nowMs(),
      windowMs: WINDOW_MS,
      playheadFrac: PLAYHEAD_FRAC,
      branchMs: data.branch_ms,
      branchLabel: "model takes over",
    });
  }

  function frame() {
    draw();
    const cur = nowMs();
    el.time.textContent = `${fmt(cur)} / ${fmt(durationMs)}`;
    if (document.activeElement !== el.seek) el.seek.value = String(Math.round(cur));
    requestAnimationFrame(frame);
  }

  /* ---------- wiring ---------- */

  el.play.addEventListener("click", () => (playing ? stop() : play()));
  el.restart.addEventListener("click", () => seekTo(0));
  el.seek.addEventListener("input", (e) => seekTo(Number(e.target.value)));
  window.addEventListener("resize", () => { resize(); draw(); });
  document.addEventListener("demo:play", (e) => {
    if (e.detail !== "shared-prompt" && playing) stop();
  });
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && e.target === document.body) {
      e.preventDefault();
      playing ? stop() : play();
    }
  });

  fetch(DATA_URL, { cache: "no-cache" })
    .then((r) => r.json())
    .then((payload) => {
      data = payload;
      data.artists.forEach((a, i) => {
        const b = document.createElement("button");
        const secs = a.duration_ms / 1000;
        b.innerHTML = `${a.name}<span class="meta">${fmt(a.duration_ms)}` +
                      ` &middot; ${(a.n_notes / secs).toFixed(1)} notes/s</span>`;
        b.title = `${a.n_notes} notes in ${secs.toFixed(1)}s`;
        b.setAttribute("aria-pressed", "false");
        b.addEventListener("click", () => setArtist(i));
        el.artists.appendChild(b);
      });
      resize();
      setArtist(Math.max(0, data.artists.findIndex((a) => a.name === "Art Tatum")));
      el.status.textContent = "";
      requestAnimationFrame(frame);
      // Audio runs on a timer, not the animation frame: hidden tabs stop
      // painting, and playback must not stop with them.
      setInterval(pump, 50);
    })
    .catch((err) => {
      el.status.textContent = "Could not load the performance data.";
      console.error(err);
    });
})();
