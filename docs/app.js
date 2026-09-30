/* One prompt, twelve pianists.
 *
 * Every take shares an opening prompt and then diverges. Choosing a pianist
 * plays that take from the top.
 */
(() => {
  "use strict";

  const DATA_URL = "data/aint_misbehavin.json";
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
  const css = getComputedStyle(document.documentElement);
  const COLOR = {
    prompt: css.getPropertyValue("--prompt").trim() || "#6f8fa8",
    gen: css.getPropertyValue("--accent").trim() || "#e0a33e",
  };

  let data = null;         // parsed payload
  let active = 0;          // index into data.artists
  let durationMs = 1;      // the selected take's own length

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const fmt = (ms) => {
    const s = Math.max(0, Math.floor(ms / 1000));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };
  const clipId = (a) => `am__${clipSlug(a.name)}`;

  function setPlayButton(on) {
    el.play.innerHTML = on ? "&#10073;&#10073;" : "&#9654;";
    el.play.setAttribute("aria-label", on ? "Pause" : "Play");
  }

  const voice = makeVoice({
    onStatus: (s) => { el.status.textContent = s; },
    onEnded: () => setPlayButton(false),
  });

  /* ---------- transport ---------- */

  async function play() {
    document.dispatchEvent(new CustomEvent("demo:play", { detail: "shared-prompt" }));
    if (voice.position >= durationMs) voice.position = 0;
    const a = data.artists[active];
    try {
      await voice.play(clipId(a), a.notes);
    } catch (err) {
      return;                    // the browser refused to start playback
    }
    setPlayButton(true);
  }

  function stop() {
    voice.stop();
    setPlayButton(false);
  }

  function seekTo(ms) {
    voice.seek(clamp(ms, 0, durationMs));
    draw();
  }

  /* ---------- artist swap ---------- */

  function setArtist(idx) {
    const wasPlaying = voice.playing;
    if (wasPlaying) voice.stop();
    active = idx;
    const a = data.artists[idx];
    el.current.textContent = a.name;
    [...el.artists.children].forEach((b, i) =>
      b.setAttribute("aria-pressed", String(i === idx)));
    // Each take is its own piece of music with its own length: start it from
    // the top, so every pianist is heard coming out of the same opening.
    durationMs = a.duration_ms;
    el.seek.max = String(durationMs);
    voice.setEnd(durationMs);
    voice.position = 0;
    if (wasPlaying) play();
    else setPlayButton(false);
    draw();
  }

  /* ---------- drawing ---------- */

  const roll = makeRoll(el.canvas);

  function draw() {
    if (!data) return;
    const a = data.artists[active];
    roll.draw({
      streams: [
        { notes: a.notes.slice(0, 4 * data.shared_notes), color: COLOR.prompt },
        { notes: a.notes.slice(4 * data.shared_notes), color: COLOR.gen },
      ],
      nowMs: voice.nowMs(),
      windowMs: WINDOW_MS,
      playheadFrac: PLAYHEAD_FRAC,
      branchMs: data.branch_ms,
      branchLabel: "model takes over",
    });
  }

  function frame() {
    draw();
    const cur = voice.nowMs();
    el.time.textContent = `${fmt(cur)} / ${fmt(durationMs)}`;
    if (document.activeElement !== el.seek) el.seek.value = String(Math.round(cur));
    requestAnimationFrame(frame);
  }

  /* ---------- wiring ---------- */

  el.play.addEventListener("click", () => (voice.playing ? stop() : play()));
  el.restart.addEventListener("click", () => seekTo(0));
  el.seek.addEventListener("input", (e) => seekTo(Number(e.target.value)));
  window.addEventListener("resize", () => { roll.resize(); draw(); });
  document.addEventListener("demo:play", (e) => {
    if (e.detail !== "shared-prompt" && voice.playing) stop();
  });
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && e.target === document.body) {
      e.preventDefault();
      voice.playing ? stop() : play();
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
      roll.resize();
      setArtist(Math.max(0, data.artists.findIndex((a) => a.name === "Art Tatum")));
      el.status.textContent = "";
      requestAnimationFrame(frame);
    })
    .catch((err) => {
      el.status.textContent = "Could not load the performance data.";
      console.error(err);
    });
})();
