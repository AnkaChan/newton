// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
import { THREE, createView, clearGroup, addAxes, matrixPoints, makeEdges, makeCellSurface, column } from './scene.js';

const IDENTITY = [[1,0,0],[0,1,0],[0,0,1]];
const NAMES = ['X', 'Y', 'Z'];

export function createAxesInspector() {
  const view = createView(document.getElementById('axes-view'));
  const content = new THREE.Group();
  view.scene.add(content);
  const spaceSelect = document.getElementById('axis-space');
  let selection = null;
  let fitted = false;
  let extent = 2.2;
  let center = new THREE.Vector3(0.45, 0.45, 0.45);
  const reset = () => view.fit(center, extent, new THREE.Vector3(1.6, -2.7, 1.8));
  const update = item => {
    selection = item;
    const local = spaceSelect.value === 'local';
    const unavailable = local && !item.valid;
    const matrix = local ? item.U : item.F;
    clearGroup(content);
    const points = matrixPoints(unavailable ? IDENTITY : matrix);
    content.add(makeEdges(matrixPoints(IDENTITY), 0xa6b3be, { opacity: 0.45 }));
    addAxes(content, IDENTITY, new THREE.Vector3(), 1, { labels: false, colorOverride: 0xa6b3be });
    if (!unavailable) {
      content.add(makeCellSurface(points, 0x79aac5, 0.075));
      content.add(makeEdges(points, 0x7c9fb3, { opacity: 0.55 }));
      addAxes(content, matrix, new THREE.Vector3(), 1, { overlay: true });
    }
    const bound = new THREE.Box3().setFromPoints(points.map(point => new THREE.Vector3(...point)));
    center = bound.getCenter(new THREE.Vector3());
    extent = Math.max(1.6, bound.getSize(new THREE.Vector3()).length() * 1.12);
    if (!fitted) { reset(); fitted = true; }
    document.getElementById('detail-title').textContent = `Cell (${item.ijk.join(', ')})`;
    document.getElementById('cell-id').textContent = item.id;
    document.getElementById('determinant').textContent = item.determinant.toFixed(5);
    document.getElementById('axes-explanation').textContent = unavailable
      ? 'Frame unavailable: inverted or nearly singular center gradient. Switch to world coordinates to inspect the raw deformation axes.'
      : local
      ? 'In the cell’s own frame: rotation removed, stretch and shear preserved. Gray arrows are the undeformed unit axes.'
      : 'In world coordinates: rotation, stretch, and shear preserved. Gray arrows are the undeformed unit axes.';
    const lengths = document.getElementById('axis-lengths');
    lengths.replaceChildren();
    NAMES.forEach((name, axis) => {
      const box = document.createElement('div');
      box.className = `axis-${name.toLowerCase()}`;
      box.textContent = `${name} axis length`;
      const value = document.createElement('strong');
      value.textContent = unavailable ? '—' : column(matrix, axis).length().toFixed(4);
      box.appendChild(value);
      lengths.appendChild(box);
    });
    const matrices = document.getElementById('matrix-values');
    matrices.replaceChildren();
    for (const [title, value] of [['F · axes in world coordinates', item.F], ['R · orthonormal cell frame', item.R], ['U = RᵀF · axes in the cell’s frame', item.U]]) {
      const block = document.createElement('div');
      block.className = 'matrix-block';
      const heading = document.createElement('h4');
      heading.textContent = title;
      const pre = document.createElement('pre');
      pre.textContent = !item.valid && value !== item.F
        ? 'Unavailable for this cell.'
        : value.map(row => row.map(v => v.toFixed(5).padStart(9)).join(' ')).join('\n');
      block.append(heading, pre);
      matrices.appendChild(block);
    }
    document.body.dataset.selectedCell = item.id;
    document.body.dataset.axisSpace = spaceSelect.value;
  };
  spaceSelect.addEventListener('change', () => { if (selection) update(selection); });
  document.getElementById('reset-axes')?.addEventListener('click', reset);
  return { update, view, reset, getState: () => ({ selectedId: selection?.id, space: spaceSelect.value, selection }) };
}
