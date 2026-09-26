/* Scored continuations + blindfold test.
 *
 * Each pianist has two takes: continuations of a few bars of their own
 * playing, drawn from the paper's synthetic-transfer corpus and scored by the
 * sliding-window classifier. The score shown is the score each take got.
 * In blind mode the list is shuffled, so its order gives nothing away.
 */
(() => {
  "use strict";

  const DATA_URL = "data/continuations.json";

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
  let items = [], order = [], active = 0, takeIdx = 0, blind = false, guessed = null;
  let answers = new Map();  // blind session: item index -> guessed name

  const fmt = (ms) => {
    const s = Math.max(0, Math.floor(ms / 1000));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const pct = (a) => `${Math.round(a * 100)}%`;
  const take = () => items[active].takes[takeIdx];
  const clipId = () => `sc__${clipSlug(items[active].artist)}__take${takeIdx + 1}`;

  function setPlayButton(on) {
    nodes.play.innerHTML = on ? "&#10073;&#10073;" : "&#9654;";
    nodes.play.setAttribute("aria-label", on ? "Pause" : "Play");
  }

  const voice = makeVoice({
    onStatus: (s) => { nodes.status.textContent = s; },
    onEnded: () => { voice.position = 0; setPlayButton(false); },
  });

  async function startPlayback() {
    document.dispatchEvent(new CustomEvent("demo:play", { detail: "showcase" }));
    const t = take();
    await voice.play(clipId(), t.prompt_notes.concat(t.notes));
    setPlayButton(true);
  }
  function stopPlayback() {
    voice.stop();
    setPlayButton(false);
  }

  function select(idx, tIdx = 0) {
    stopPlayback();
    active = idx;
    takeIdx = tIdx;
    voice.setEnd(take().duration_ms);
    voice.position = 0;
    guessed = blind && answers.has(idx) ? answers.get(idx) : null;
    showVerdict();
    [...nodes.list.children].forEach((b, i) =>
      b.setAttribute("aria-pressed", String(order[i] === idx)));
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
      ? `<span class="masked">mystery pianist</span>`
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
    nodes.time.textContent = `${fmt(voice.nowMs())} / ${fmt(t.duration_ms)}`;
  }

  function showVerdict() {
    if (!blind || guessed === null) { nodes.reveal.innerHTML = ""; return; }
    const it = items[active];
    const right = guessed === it.artist;
    const correct = [...answers].filter(([i, n]) => items[i].artist === n).length;
    nodes.reveal.innerHTML =
      `<div class="verdict ${right ? "right" : "wrong"}">` +
      (right ? `Yes &mdash; <b>${it.artist}</b>.` :
               `You said ${guessed}; it was <b>${it.artist}</b>.`) +
      ` The classifier named ${it.artist} on <b>${pct(take().agreement)}</b> of windows.` +
      `<span class="tally">Your score: ${correct} of ${answers.size}` +
      (answers.size < items.length ? " &middot; pick another pianist to keep going" : " &middot; that&rsquo;s all twelve") +
      `</span></div>`;
  }

  function makeGuess(name) {
    guessed = name;
    answers.set(active, name);
    const pos = order.indexOf(active);
    const chip = nodes.list.children[pos];
    const right = name === items[active].artist;
    chip.textContent = `${right ? "✓" : "✗"} ${items[active].artist}`;
    chip.classList.add(right ? "got" : "missed");
    showVerdict();
    render();
  }

  function rebuildList() {
    nodes.list.innerHTML = "";
    order = items.map((_, i) => i);
    if (blind) {
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
    }
    order.forEach((i, pos) => {
      const it = items[i];
      const b = document.createElement("button");
      b.textContent = blind ? `Pianist ${String.fromCharCode(65 + pos)}` : it.artist;
      b.setAttribute("aria-pressed", String(i === active));
      b.addEventListener("click", () => select(i));
      nodes.list.appendChild(b);
    });
  }

  nodes.play.addEventListener("click", () => (voice.playing ? stopPlayback() : startPlayback()));
  nodes.seek.addEventListener("input", (e) =>
    voice.seek(clamp(Number(e.target.value), 0, take().duration_ms)));
  nodes.blind.addEventListener("click", () => {
    blind = !blind;
    answers = new Map();
    nodes.blind.setAttribute("aria-pressed", String(blind));
    nodes.blind.textContent = blind ? "blindfold: on" : "blindfold: off";
    guessed = null;
    nodes.reveal.innerHTML = "";
    rebuildList();
    select(order[0]);
  });
  document.addEventListener("demo:play", (e) => {
    if (e.detail !== "showcase" && voice.playing) stopPlayback();
  });

  fetch(DATA_URL, { cache: "no-cache" })
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
      select(Math.max(0, items.findIndex((i) => i.artist === "Art Tatum")));
      (function frame() {
        if (items.length) {
          const t = take();
          const cur = voice.nowMs();
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
            branchLabel: "model takes over",
          });
        }
        requestAnimationFrame(frame);
      })();
    })
    .catch((e) => { nodes.status.textContent = "Could not load continuations."; console.error(e); });
})();
