/* Scored continuations + blindfold test.
 *
 * Each item is a continuation of one pianist's own held-out performance,
 * carrying the classifier's per-window verdicts. Two ways to hear it:
 * openly (with the pianist and score shown) or blind (guess first).
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
    time: el("cont-time"),
    label: el("cont-label"),
    strip: el("cont-strip"),
    score: el("cont-score"),
    blind: el("blind-toggle"),
    guess: el("guess-panel"),
    guessBtns: el("guess-buttons"),
    reveal: el("reveal"),
    status: el("cont-status"),
  };

  let data = null, items = [], active = 0, blind = false, guessed = null;
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
  const item = () => items[active];
  const nowMs = () => (playing ? offsetMs + (Tone.now() - startedAt) * 1000 : offsetMs);

  async function initAudio() {
    if (sampler) return;
    nodes.status.textContent = "Loading piano samples…";
    const base = "https://tonejs.github.io/audio/salamander/";
    const urls = {};
    for (const oct of [1,2,3,4,5,6,7]) for (const p of ["A","C","D#","F#"]) {
      urls[p + oct] = p.replace("#","s") + oct + ".mp3";
    }
    urls["A0"] = "A0.mp3"; urls["C8"] = "C8.mp3";
    sampler = new Tone.Sampler({ urls, baseUrl: base, release: 1.2 }).toDestination();
    sampler.volume.value = -4;
    await Tone.loaded();
    nodes.status.textContent = "";
  }

  /* notes of the current item, prompt then continuation, as one stream */
  function stream() {
    const it = item();
    return { prompt: it.prompt_notes, gen: it.notes, branch: it.branch_ms,
             duration: it.duration_ms };
  }

  function pump() {
    if (!playing || !sampler) return;
    const cur = nowMs();
    const until = cur + (document.hidden ? LOOKAHEAD_HIDDEN_MS : LOOKAHEAD_VISIBLE_MS);
    const t0 = Tone.now();
    const s = stream();
    for (const arr of [s.prompt, s.gen]) {
      for (let i = 0; i < arr.length; i += 4) {
        const start = arr[i];
        if (start < scheduledMs) continue;
        if (start >= until) break;
        try {
          sampler.triggerAttackRelease(midiToNote(arr[i + 2]),
            Math.max(0.05, arr[i + 1] / 1000),
            t0 + Math.max(0, (start - cur) / 1000),
            clamp(arr[i + 3] / 127, 0.05, 1));
        } catch (e) { /* out-of-range note */ }
      }
    }
    scheduledMs = Math.max(scheduledMs, until);
    if (cur >= s.duration) stopPlayback(true);
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

  function select(idx, { keepBlind = false } = {}) {
    active = idx;
    stopPlayback();
    offsetMs = 0;
    guessed = null;
    if (!keepBlind) nodes.reveal.innerHTML = "";
    [...nodes.list.children].forEach((b, i) =>
      b.setAttribute("aria-pressed", String(i === idx)));
    render();
  }

  /* the per-window verdict strip: what the classifier hears, position by position */
  function renderStrip() {
    const it = item();
    nodes.strip.innerHTML = "";
    for (const w of it.windows) {
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
    const it = item();
    const hidden = blind && guessed === null;
    nodes.label.innerHTML = hidden
      ? `<span class="masked">a held-out performance</span>`
      : `<b>${it.artist}</b>${it.take === "weaker"
          ? ' <span class="tag">a weaker take</span>' : ""}` +
        `<span class="sub">continuing &ldquo;${it.prompt_title}&rdquo;</span>`;
    nodes.score.innerHTML = hidden ? "" :
      `<span class="big">${Math.round(it.agreement * 100)}%</span>` +
      `<span class="cap">classifier agreement<br>across ${it.windows.length} windows</span>`;
    nodes.strip.classList.toggle("masked", hidden);
    renderStrip();
    nodes.guess.hidden = !blind || guessed !== null;
    nodes.time.textContent = `${fmt(nowMs())} / ${fmt(it.duration_ms)}`;
  }

  function makeGuess(name) {
    guessed = name;
    const it = item();
    const right = name === it.artist;
    nodes.reveal.innerHTML =
      `<div class="verdict ${right ? "right" : "wrong"}">` +
      `You said <b>${name}</b>. It was <b>${it.artist}</b>` +
      (it.mode === "ablation" ? " (with conditioning removed)" : "") + `.<br>` +
      `The classifier agreed with ${it.artist} on ` +
      `<b>${Math.round(it.agreement * 100)}%</b> of windows.</div>`;
    render();
  }

  function buildList() {
    items.forEach((it, i) => {
      const b = document.createElement("button");
      const tag = it.take === "weaker" ? ' <span class="take">weaker take</span>' : "";
      b.innerHTML = `<span class="idx">${String(i + 1).padStart(2, "0")}</span>` +
        `<span class="nm">${blind ? "unknown take" : it.artist + tag}</span>` +
        (blind ? "" : `<span class="ag">${Math.round(it.agreement * 100)}%</span>`);
      b.setAttribute("aria-pressed", "false");
      b.addEventListener("click", () => select(i));
      nodes.list.appendChild(b);
    });
  }

  function rebuildList() {
    nodes.list.innerHTML = "";
    buildList();
    [...nodes.list.children].forEach((b, i) =>
      b.setAttribute("aria-pressed", String(i === active)));
  }

  nodes.play.addEventListener("click", () => (playing ? stopPlayback() : startPlayback()));
  nodes.blind.addEventListener("click", () => {
    blind = !blind;
    nodes.blind.setAttribute("aria-pressed", String(blind));
    nodes.blind.textContent = blind ? "showing: blind" : "showing: labelled";
    guessed = null;
    nodes.reveal.innerHTML = "";
    rebuildList();
    render();
  });

  fetch(DATA_URL)
    .then((r) => r.json())
    .then((payload) => {
      data = payload;
      items = payload.items;
      buildList();
      // guess buttons: every pianist in the set
      const names = [...new Set(items.map((i) => i.artist))].sort();
      for (const n of names) {
        const g = document.createElement("button");
        g.textContent = n;
        g.addEventListener("click", () => makeGuess(n));
        nodes.guessBtns.appendChild(g);
      }
      select(0);
      setInterval(pump, 50);
      (function frame() {
        if (data) nodes.time.textContent =
          `${fmt(nowMs())} / ${fmt(item().duration_ms)}`;
        requestAnimationFrame(frame);
      })();
    })
    .catch((e) => { nodes.status.textContent = "Could not load continuations."; console.error(e); });
})();
