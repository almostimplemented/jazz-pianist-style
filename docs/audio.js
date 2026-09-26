/* Shared piano for every player on the page.
 *
 * getSampler() loads the Salamander samples once and hands the same sampler
 * to whichever player asks. Players announce themselves with a "demo:play"
 * event so that only one sounds at a time.
 */
(() => {
  "use strict";

  let loading = null;

  window.getSampler = () => {
    if (!loading) {
      loading = (async () => {
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
    return loading;
  };

  const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  window.midiToNote = (p) => NAMES[p % 12] + (Math.floor(p / 12) - 1);
})();
