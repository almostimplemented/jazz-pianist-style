/* Shared audio for every player on the page.
 *
 * Clips play from pre-rendered audio files when the site ships them
 * (audio/index.json lists them, keyed by clip id); otherwise a sampled piano
 * plays the notes directly in the browser. makeVoice() hides the difference:
 * each player owns one voice, hands it a clip id plus the clip's notes, and
 * reads the position back from it. Players announce themselves with a
 * "demo:play" event so that only one sounds at a time.
 */
(() => {
  "use strict";

  const AUDIO_BASE = window.AUDIO_BASE || "audio/";           // where the clip files live
  const AUDIO_INDEX = window.AUDIO_INDEX || "audio/index.json"; // small, kept with the page
  const LOOKAHEAD_VISIBLE_MS = 150;   // responsive while watching
  const LOOKAHEAD_HIDDEN_MS = 3000;   // background tabs clamp timers to ~1 s

  const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  window.midiToNote = (p) => NAMES[p % 12] + (Math.floor(p / 12) - 1);
  // Same rule as scripts/site/build_render_sheet.py, so ids match the files
  window.clipSlug = (s) => s.replace(/[^A-Za-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  /* ---------- fallback: sampled piano ---------- */

  let samplerLoading = null;
  window.getSampler = () => {
    if (!samplerLoading) {
      samplerLoading = (async () => {
        const urls = { A0: "A0.mp3", C8: "C8.mp3" };
        for (const oct of [1, 2, 3, 4, 5, 6, 7]) {
          for (const p of ["A", "C", "D#", "F#"]) urls[p + oct] = p.replace("#", "s") + oct + ".mp3";
        }
        const sampler = new Tone.Sampler({
          urls, baseUrl: "https://tonejs.github.io/audio/salamander/", release: 1.2,
        }).toDestination();
        sampler.volume.value = -4;
        await Tone.loaded();
        return sampler;
      })();
    }
    return samplerLoading;
  };

  /* ---------- pre-rendered clips ---------- */

  let index = null;
  let audioBroken = false;   // a file failed to play: stay on the sampler from then on
  const elements = new Map();
  const indexReady = fetch(AUDIO_INDEX, { cache: "no-cache" })
    .then((r) => (r.ok ? r.json() : null))
    .then((idx) => { index = idx; return idx; })
    .catch(() => null);
  window.audioReady = indexReady;

  const hasAudio = (id) => !audioBroken && !!(index && index[id]);
  function getEl(id) {
    let e = elements.get(id);
    if (!e) {
      e = new Audio(AUDIO_BASE + index[id].file);
      e.preload = "auto";
      elements.set(id, e);
    }
    return e;
  }

  /* ---------- one voice per player ---------- */

  window.makeVoice = ({ onStatus, onEnded } = {}) => {
    let mode = null;           // "audio" | "sampler"
    let playing = false;
    let clipId = null, notes = [];
    let el = null, ringing = null;
    let sampler = null, startedAt = 0, scheduledMs = 0;
    let offsetMs = 0, endMs = Infinity;

    const nowMs = () => {
      if (!playing) return offsetMs;
      return mode === "audio" ? el.currentTime * 1000
                              : offsetMs + (Tone.now() - startedAt) * 1000;
    };

    function quiet() { if (ringing) { ringing.pause(); ringing = null; } }

    const metadata = (e) => new Promise((res) => {
      e.addEventListener("loadedmetadata", res, { once: true });
      e.addEventListener("error", res, { once: true });
      setTimeout(res, 4000);
    });

    // Start an element at a position. With metadata in hand the seek goes
    // first; otherwise play() goes first so the user's gesture is not lost
    // on browsers that load nothing in advance, and the seek follows.
    async function startAt(e, ms) {
      if (e.readyState >= 1) {
        e.currentTime = ms / 1000;
        await e.play();
      } else {
        const p = e.play();
        await metadata(e);
        if (ms > 0) e.currentTime = ms / 1000;
        await p;
      }
      if (Math.abs(e.currentTime * 1000 - ms) > 500) e.currentTime = ms / 1000;
    }

    async function startAudio(id) {
      const e = getEl(id);
      await startAt(e, offsetMs);
      return e;
    }

    async function startSampler() {
      await Tone.start();
      if (!sampler) {
        onStatus?.("Loading piano samples…");
        sampler = await getSampler();
        onStatus?.("");
      }
      startedAt = Tone.now();
      scheduledMs = offsetMs;
    }

    async function play(id, clipNotes) {
      quiet();
      if (playing) stop();
      await indexReady;
      clipId = id; notes = clipNotes;
      if (hasAudio(id)) {
        try {
          el = await startAudio(id);
          mode = "audio"; playing = true;
          return;
        } catch (err) {
          console.warn("pre-rendered audio unavailable, using the sampler", id, err);
          audioBroken = true;
        }
      }
      await startSampler();
      mode = "sampler"; playing = true;
    }

    function stop() {
      quiet();
      if (!playing) return;
      offsetMs = Math.min(nowMs(), endMs);
      playing = false;
      if (mode === "audio") el.pause();
      else if (sampler) sampler.releaseAll();
    }

    // The clip reached its musical end: let the last notes ring, report it.
    function finish() {
      playing = false;
      offsetMs = endMs;
      if (mode === "audio") ringing = el;
      onEnded?.();
    }

    // Keep the position, change the clip: the live style swap.
    async function switchTo(id, clipNotes) {
      const pos = nowMs();
      if (id === clipId) { notes = clipNotes; return; }
      if (!playing) { clipId = id; notes = clipNotes; offsetMs = pos; return; }
      if (mode === "audio" && hasAudio(id)) {
        const next = getEl(id), prev = el;
        el = next; clipId = id; notes = clipNotes;
        try { await startAt(next, pos); }
        catch (err) {
          console.warn("pre-rendered audio unavailable, using the sampler", id, err);
          audioBroken = true; el = prev; stop(); offsetMs = pos;
          await play(id, clipNotes);
          return;
        }
        prev.pause();
        return;
      }
      if (mode === "sampler") { clipId = id; notes = clipNotes; scheduledMs = pos; return; }
      stop(); offsetMs = pos;          // audio mode, but this clip has no file
      await play(id, clipNotes);
    }

    async function seek(ms) {
      quiet();
      ms = clamp(ms, 0, endMs);
      if (playing && mode === "audio") { el.currentTime = ms / 1000; return; }
      const was = playing;
      if (was) stop();
      offsetMs = ms;
      if (was) await play(clipId, notes);
    }

    // Schedule the sampler's notes a little ahead of the clock.
    function pump() {
      if (!playing || mode !== "sampler" || !sampler) return;
      const cur = nowMs();
      const until = cur + (document.hidden ? LOOKAHEAD_HIDDEN_MS : LOOKAHEAD_VISIBLE_MS);
      const t0 = Tone.now();
      for (let i = 0; i < notes.length; i += 4) {
        const start = notes[i];
        if (start < scheduledMs) continue;
        if (start >= until) break;
        try {
          sampler.triggerAttackRelease(midiToNote(notes[i + 2]),
            Math.max(0.05, notes[i + 1] / 1000),
            t0 + Math.max(0, (start - cur) / 1000),
            clamp(notes[i + 3] / 127, 0.05, 1));
        } catch (e) { /* a note outside the sampled range is not worth stopping for */ }
      }
      scheduledMs = Math.max(scheduledMs, until);
    }
    // A timer, not the animation frame: hidden tabs stop painting, and
    // playback must not stop with them.
    setInterval(() => { pump(); if (playing && nowMs() >= endMs) finish(); }, 50);

    const voice = {
      play, stop, switchTo, seek, nowMs,
      debug: () => ({ mode, playing, clipId, position: Math.round(nowMs()),
                      el: el && { src: el.src.split("/").pop(), paused: el.paused, t: +el.currentTime.toFixed(2) } }),
      get playing() { return playing; },
      get position() { return offsetMs; },
      set position(ms) { if (!playing) offsetMs = clamp(ms, 0, endMs); },
      setEnd(ms) { endMs = ms; },
      // Create the elements early so later swaps have data ready
      warm(ids) { indexReady.then(() => { if (!audioBroken && index) ids.forEach((id) => index[id] && getEl(id)); }); },
    };
    (window.__voices = window.__voices || []).push(voice);
    return voice;
  };
})();
