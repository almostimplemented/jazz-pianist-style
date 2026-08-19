/* Scored continuations + blindfold test.
 *
 * Each pianist has two takes: continuations of their own held-out
 * performances, drawn from the paper's synthetic corpus and scored by the
 * sliding-window classifier. The score shown is the score each take got.
 */
(() => {
  "use strict";

  const DATA_URL = "data/continuations.json";
  const LOOKAHEAD_VISIBLE_MS = 150;
  const LOOKAHEAD_HIDDEN_MS = 3000;

  const el = (id) => document.getElementById(id);
  const nodes = {
    list: el("cont-list"),
    play: el("cont-play"),
    seek: el("cont-seek"),
    canvas: el("cont-roll"),
    time: el("cont-time"),
    label: el("cont-label"),
    takes: el("cont-takes"),
    strip: el("cont-strip"),
    score: el("cont-score"),
    blind: el("blind-toggle"),
    guess: el("guess-panel"),
    guessBtns: el("guess-buttons"),
    reveal: el("reveal"),
    status: el("cont-status"),
  };

  const css = getComputedStyle(document.documentElement);
  const COLOR = {
    prompt: css.getPropertyValue("--prompt").trim() || "#6f8fa8",
    gen: css.getPropertyValue("--accent").trim() || "#e0a33e",
  };
  let roll = null;
  let items = [], active = 0, takeIdx = 0, blind = false, guessed = null;
  let sampler = null, playing = false, startedAt = 0, offsetMs = 0, scheduledMs = 0;

  const fmt = (ms) => {
    const s = Math.max(0, Math.floor(ms / 1000));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };
  const midiToNote = (p) => {
    const n = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
    return n[p % 12] + (Math.floor(p / 12) - 1);
  };
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const pct = (a) => `${Math.round(a * 100)}%`;
  const take = () => items[active].takes[takeIdx];
  const nowMs = () => (playing ? offsetMs + (Tone.now() - startedAt) * 1000 : offsetMs);

  async function initAudio() {
    if (sampler) return;
    nodes.status.textContent = "Loading piano samples…";
    const urls = {};
    for (const oct of [1, 2, 3, 4, 5, 6, 7]) for (const p of ["A", "C", "D#", "F#"]) {
      urls[p + oct] = p.replace("#", "s") + oct + ".mp3";
    }
    urls["A0"] = "A0.mp3"; urls["C8"] = "C8.mp3";
    sampler = new Tone.Sampler({
      urls, baseUrl: "https://tonejs.github.io/audio/salamander/", release: 1.2,
    }).toDestination();
    sampler.volume.value = -4;
    await Tone.loaded();
    nodes.status.textContent = "";
  }

  function pump() {
    if (!playing || !sampler) return;
    const cur = nowMs();
    const until = cur + (document.hidden ? LOOKAHEAD_HIDDEN_MS : LOOKAHEAD_VISIBLE_MS);
    const t0 = Tone.now();
    const t = take();
    for (const notes of [t.prompt_notes, t.notes]) {
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
    }
    scheduledMs = Math.max(scheduledMs, until);
    if (cur >= take().duration_ms) stopPlayback(true);
  }

  async function startPlayback() {
    await Tone.start();
    await initAudio();
    playing = true;
    startedAt = Tone.now();
    scheduledMs = offsetMs;
    nodes.play.innerHTML = "&#10073;&#10073;";
  }
  function stopPlayback(ended = false) {
    if (playing) offsetMs = ended ? 0 : nowMs();
    playing = false;
    if (sampler) sampler.releaseAll();
    nodes.play.innerHTML = "&#9654;";
  }

  function select(idx, tIdx = 0) {
    active = idx;
    takeIdx = tIdx;
    stopPlayback();
    offsetMs = 0;
    guessed = null;
    nodes.reveal.innerHTML = "";
    [...nodes.list.children].forEach((b, i) =>
      b.setAttribute("aria-pressed", String(i === idx)));
    nodes.seek.max = String(take().duration_ms);
    render();
  }

  function renderStrip() {
    const t = take();
    nodes.strip.innerHTML = "";
    for (const w of t.windows) {
      const cell = document.createElement("div");
      cell.className = "cell " + (w.correct ? "hit" : "miss");
      cell.style.opacity = String(0.35 + 0.65 * w.confidence);
      cell.title = blind
        ? `window at token ${w.position}`
        : `token ${w.position}: heard ${w.pred} (${Math.round(w.confidence * 100)}%)`;
      nodes.strip.appendChild(cell);
    }
  }

  function render() {
    const it = items[active];
    const t = take();
    const hidden = blind && guessed === null;

    nodes.label.innerHTML = hidden
      ? `<span class="masked">a held-out performance</span>`
      : `<b>${it.artist}</b>` +
        `<span class="sub">continuing &ldquo;${t.prompt_title}&rdquo;</span>`;

    nodes.takes.innerHTML = "";
    if (!blind && it.takes.length > 1) {
      it.takes.forEach((tk, i) => {
        const b = document.createElement("button");
        b.textContent = `Take ${i + 1} · ${pct(tk.agreement)}`;
        b.setAttribute("aria-pressed", String(i === takeIdx));
        b.addEventListener("click", () => { if (i !== takeIdx) select(active, i); });
        nodes.takes.appendChild(b);
      });
    }

    nodes.score.innerHTML = hidden ? "" :
      `<span class="big">${pct(t.agreement)}</span>` +
      `<span class="cap">classifier agreement<br>across ${t.windows.length} windows</span>`;
    nodes.strip.classList.toggle("masked", hidden);
    renderStrip();
    nodes.guess.hidden = !blind || guessed !== null;
    nodes.time.textContent = `${fmt(nowMs())} / ${fmt(t.duration_ms)}`;
  }

  function makeGuess(name) {
    guessed = name;
    const it = items[active];
    const right = name === it.artist;
    nodes.reveal.innerHTML =
      `<div class="verdict ${right ? "right" : "wrong"}">` +
      `You said <b>${name}</b>. It was <b>${it.artist}</b>.<br>` +
      `The classifier agreed with ${it.artist} on ` +
      `<b>${pct(take().agreement)}</b> of windows.</div>`;
    render();
  }

  function rebuildList() {
    nodes.list.innerHTML = "";
    items.forEach((it, i) => {
      const b = document.createElement("button");
      b.innerHTML = `<span class="idx">${String(i + 1).padStart(2, "0")}</span>` +
        `<span class="nm">${blind ? "performance" : it.artist}</span>` +
        (blind ? "" : `<span class="ag">${pct(it.takes[0].agreement)}</span>`);
      b.setAttribute("aria-pressed", String(i === active));
      b.addEventListener("click", () => select(i));
      nodes.list.appendChild(b);
    });
  }

  function seekTo(ms) {
    const was = playing;
    if (was) stopPlayback();
    offsetMs = clamp(ms, 0, take().duration_ms);
    if (was) startPlayback();
  }

  nodes.play.addEventListener("click", () => (playing ? stopPlayback() : startPlayback()));
  nodes.seek.addEventListener("input", (e) => seekTo(Number(e.target.value)));
  nodes.blind.addEventListener("click", () => {
    blind = !blind;
    nodes.blind.setAttribute("aria-pressed", String(blind));
    nodes.blind.textContent = blind ? "showing: blind" : "showing: labelled";
    guessed = null;
    nodes.reveal.innerHTML = "";
    takeIdx = 0;
    rebuildList();
    render();
  });

  fetch(DATA_URL)
    .then((r) => r.json())
    .then((payload) => {
      items = payload.items;
      rebuildList();
      const names = items.map((i) => i.artist).sort();
      for (const n of names) {
        const g = document.createElement("button");
        g.textContent = n;
        g.addEventListener("click", () => makeGuess(n));
        nodes.guessBtns.appendChild(g);
      }
      roll = makeRoll(nodes.canvas);
      select(0);
      setInterval(pump, 50);
      (function frame() {
        if (items.length) {
          const t = take();
          const cur = nowMs();
          nodes.time.textContent = `${fmt(cur)} / ${fmt(t.duration_ms)}`;
          if (document.activeElement !== nodes.seek)
            nodes.seek.value = String(Math.round(cur));
          roll.draw({
            streams: [
              { notes: t.prompt_notes, color: COLOR.prompt },
              { notes: t.notes, color: COLOR.gen },
            ],
            nowMs: cur,
            branchMs: t.branch_ms,
            branchLabel: blind && guessed === null ? "model takes over" : "model takes over",
          });
        }
        requestAnimationFrame(frame);
      })();
    })
    .catch((e) => { nodes.status.textContent = "Could not load continuations."; console.error(e); });
})();
