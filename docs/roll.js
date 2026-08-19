/* Shared piano-roll renderer.
 *
 * makeRoll(canvas) returns { draw(state) } where state is:
 *   streams: [{ notes: flat [start_ms,dur_ms,pitch,vel], color }]
 *   nowMs, windowMs, playheadFrac
 *   branchMs (optional): dashed marker with a label
 */
(() => {
  "use strict";

  const css = getComputedStyle(document.documentElement);
  const COLOR = {
    line: css.getPropertyValue("--line").trim() || "#2e2823",
    dim: css.getPropertyValue("--ink-dim").trim() || "#a89e91",
  };
  const LO = 21, HI = 108;

  function makeRoll(canvas) {
    const ctx = canvas.getContext("2d");

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(canvas.clientWidth * dpr);
      canvas.height = Math.round(canvas.clientHeight * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    window.addEventListener("resize", resize);

    function draw(state) {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      const windowMs = state.windowMs || 8000;
      const frac = state.playheadFrac ?? 0.28;
      const from = state.nowMs - windowMs * frac;
      const to = from + windowMs;
      const yOf = (p) => h - ((p - LO) / (HI - LO)) * (h - 8) - 4;
      const xOf = (ms) => ((ms - from) / windowMs) * w;

      ctx.clearRect(0, 0, w, h);

      ctx.strokeStyle = COLOR.line;
      ctx.lineWidth = 1;
      for (let p = 24; p <= HI; p += 12) {
        const y = Math.round(yOf(p)) + 0.5;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }

      if (state.branchMs != null) {
        const bx = xOf(state.branchMs);
        if (bx > -20 && bx < w + 20) {
          ctx.save();
          ctx.strokeStyle = COLOR.dim;
          ctx.setLineDash([4, 4]);
          ctx.beginPath(); ctx.moveTo(bx, 0); ctx.lineTo(bx, h); ctx.stroke();
          ctx.restore();
          ctx.fillStyle = COLOR.dim;
          ctx.font = "11px ui-monospace, monospace";
          ctx.fillText(state.branchLabel || "", bx + 6, 14);
        }
      }

      const noteH = Math.max(2.5, (h - 8) / (HI - LO) * 1.6);
      for (const stream of state.streams) {
        const notes = stream.notes;
        for (let i = 0; i < notes.length; i += 4) {
          const start = notes[i], dur = notes[i + 1], pitch = notes[i + 2], vel = notes[i + 3];
          if (start + dur < from) continue;
          if (start > to) break;
          const x = xOf(start), wpx = Math.max(2, (dur / windowMs) * w);
          const y = yOf(pitch);
          const played = start <= state.nowMs;
          ctx.globalAlpha = played ? 1 : 0.42;
          ctx.fillStyle = stream.color;
          ctx.beginPath();
          if (ctx.roundRect) {
            ctx.roundRect(x, y - noteH / 2, wpx, noteH, Math.min(2, noteH / 2));
          } else {
            ctx.rect(x, y - noteH / 2, wpx, noteH);
          }
          ctx.fill();
          if (played && vel > 90) {
            ctx.globalAlpha = 0.25;
            ctx.fillStyle = "#fff";
            ctx.fill();
          }
        }
      }
      ctx.globalAlpha = 1;

      const px = Math.round(w * frac) + 0.5;
      ctx.strokeStyle = "#fff";
      ctx.globalAlpha = 0.8;
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
      ctx.globalAlpha = 1;
    }

    return { draw, resize };
  }

  window.makeRoll = makeRoll;
})();
