// Fixed chordwise stations at which surrogate Cp is learned / reported.
export const CP_STATIONS = Array.from({ length: 32 }, (_, k) => {
  const t = (k + 0.5) / 32;
  return 0.5 * (1 - Math.cos(Math.PI * t));
});

/** Linear interpolation of Cp(x) onto CP_STATIONS. */
export function resampleCp(x, cp) {
  return CP_STATIONS.map((xs) => {
    let i = 1;
    while (i < x.length - 1 && x[i] < xs) i++;
    const t = (xs - x[i - 1]) / (x[i] - x[i - 1]);
    return cp[i - 1] + Math.max(0, Math.min(1, t)) * (cp[i] - cp[i - 1]);
  });
}
