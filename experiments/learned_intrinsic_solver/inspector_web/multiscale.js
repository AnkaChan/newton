// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

export function setupMultiscale(catalog, initial, replaceData) {
  if (!catalog) return;
  document.title = 'Multiscale initial shapes · Learned intrinsic solver';
  document.querySelector('h1').textContent = 'Multiscale initial shapes';
  document.querySelector('.intro').textContent = 'Explore 20 seeded shapes. Compare the broad, medium, and fine deformation, then click a cell to inspect its frame on the deformed grid.';
  const section = document.createElement('section');
  section.className = 'sample-panel panel';
  section.innerHTML = `<div class="sample-heading"><div><p class="eyebrow">AUTOMATIC CONTROL GRIDS · CANONICAL REST SHAPE</p><h2>Choose an initial shape</h2></div><div class="sample-selects"><label>Random seed <select id="sample-seed"></select></label><label>Deformation contribution <select id="sample-mode"></select></label></div></div><div class="sample-info"><div><div id="level-settings" class="level-settings"></div><p class="small-note">Numbers count control points, including both ends. Each grid spans the same rest bounds; the fine grid is the actual shared-corner grid.</p></div><div><p id="sample-metrics"></p><p id="sample-status" class="small-note" aria-live="polite"></p><a id="sample-download" class="small-note" download>Download exact initial state</a></div></div><details><summary>How these shapes are generated</summary><p id="hierarchy-note"></p><p id="amplitude-note"></p><p id="orientation-note"></p><p id="limitation-note"></p></details>`;
  document.querySelector('main').prepend(section);
  const seedSelect = document.getElementById('sample-seed');
  const modeSelect = document.getElementById('sample-mode');
  const status = document.getElementById('sample-status');
  for (const seed of catalog.seeds) seedSelect.add(new Option(String(seed), String(seed)));
  for (const mode of catalog.modes) modeSelect.add(new Option(mode.label, mode.id));
  for (const [id, text] of [['hierarchy-note',catalog.hierarchy_rule],['amplitude-note',catalog.amplitude_rule],['orientation-note',catalog.orientation_rule],['limitation-note',catalog.limitation]]) document.getElementById(id).textContent = text;
  let active = { seed: initial.metadata.seed, mode: initial.metadata.mode };
  let requestNumber = 0;
  let abort = null;
  const key = (seed, mode) => `${seed}:${mode}`;
  const cache = new Map([[key(active.seed, active.mode), initial]]);
  const showSettings = state => {
    const levels = document.getElementById('level-settings');
    levels.replaceChildren();
    catalog.levels.forEach((level, index) => {
      const item = document.createElement('div');
      item.className = 'level-card' + (state.mode !== 'combined' && state.mode !== level.name ? ' inactive' : '');
      const heading = document.createElement('strong');
      heading.textContent = level.name[0].toUpperCase() + level.name.slice(1);
      const count = document.createElement('span');
      count.textContent = level.control_counts.join(' × ');
      const amount = document.createElement('span');
      amount.textContent = `${(state.effective_amplitudes_m[index] * 1000).toFixed(2)} mm effective`;
      const requested = document.createElement('small');
      requested.textContent = `${(level.amplitude_m * 1000).toFixed(2)} mm requested`;
      item.append(heading, count, amount, requested);
      levels.append(item);
    });
    document.getElementById('sample-metrics').textContent = `Corner displacement: ${(state.rms_displacement_m * 1000).toFixed(1)} mm RMS · ${(state.max_displacement_m * 1000).toFixed(1)} mm maximum. Minimum sampled volume ratio: ${Math.min(state.min_tet_volume_ratio, state.min_sampled_jacobian).toFixed(3)}.`;
    status.textContent = `Seed ${state.seed} · ${catalog.modes.find(mode => mode.id === state.mode).label} · common amplitude scale ${state.effective_scale}× (${state.backtracking_steps} halvings). The fixed end is stationary at every level.`;
    const download = document.getElementById('sample-download');
    download.href = state.download;
    document.getElementById('generation-note').textContent = `Rest cell: ${(initial.metadata.cell_size * 1000).toFixed(0)} mm · automatic hierarchy · seed ${state.seed} · ${state.mode}`;
    document.body.dataset.seed = state.seed;
    document.body.dataset.mode = state.mode;
  };
  const stateFor = (seed, mode) => catalog.states.find(state => state.seed === seed && state.mode === mode);
  const loadSelection = async () => {
    const seed = Number(seedSelect.value), mode = modeSelect.value;
    const state = stateFor(seed, mode);
    const request = ++requestNumber;
    abort?.abort();
    abort = new AbortController();
    document.body.dataset.loading = 'true';
    status.textContent = `Loading seed ${seed}, ${mode}… The current grid remains visible until ready.`;
    try {
      let payload = cache.get(key(seed, mode));
      if (!payload) {
        const response = await fetch(state.file, { signal: abort.signal });
        if (!response.ok) throw new Error(`The shape could not be loaded (${response.status}).`);
        payload = await response.json();
        // Keep memory bounded while making the most recent comparisons immediate.
        if (cache.size >= 8) cache.delete(cache.keys().next().value);
        cache.set(key(seed, mode), payload);
      }
      if (request !== requestNumber) return;
      replaceData(payload);
      active = { seed, mode };
      showSettings(state);
    } catch (error) {
      if (request !== requestNumber || error.name === 'AbortError') return;
      status.textContent = error.message;
      seedSelect.value = active.seed;
      modeSelect.value = active.mode;
    } finally {
      if (request === requestNumber) document.body.dataset.loading = 'false';
    }
  };
  seedSelect.value = active.seed;
  modeSelect.value = active.mode;
  seedSelect.addEventListener('change', loadSelection);
  modeSelect.addEventListener('change', loadSelection);
  showSettings(stateFor(active.seed, active.mode));
  document.body.dataset.loading = 'false';
}
