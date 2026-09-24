// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
import { THREE, createView, clearGroup, addAxes, makeEdges, makeCellSurface, FACES, column } from './scene.js';
import { createAxesInspector } from './axes.js';
import { setupMultiscale } from './multiscale.js';

try {
  let data = window.CELL_FRAME_DATA;
  if (!data?.positions?.length) throw new Error('The cell data could not be loaded. Serve this folder with a local HTTP server or use the published link.');
  const counts = data.metadata.cell_counts;
  const cellSize = data.metadata.cell_size;
  const cellCount = data.cells.length;
  const view = createView(document.getElementById('grid-view'));
  const axesInspector = createAxesInspector();
  const gridGroup = new THREE.Group();
  const selectedGroup = new THREE.Group();
  view.scene.add(gridGroup, selectedGroup);
  let boundingBox = new THREE.Box3().setFromPoints(data.positions.map(p => new THREE.Vector3(...p)));
  let center = boundingBox.getCenter(new THREE.Vector3());
  let fullExtent = boundingBox.getSize(new THREE.Vector3()).length();
  const sliceAxis = document.getElementById('slice-axis');
  const sliceLayer = document.getElementById('slice-layer');
  const indexInputs = ['cell-i','cell-j','cell-k'].map(id => document.getElementById(id));
  let selectedId = -1;
  let gridMesh;
  let triangleCells = [];
  let visibleIds = [];
  let popup = null;
  let currentSelection = null;
  const ijkOf = id => [Math.floor(id / (counts[1] * counts[2])), Math.floor(id / counts[2]) % counts[1], id % counts[2]];
  const idOf = ([i,j,k]) => (i * counts[1] + j) * counts[2] + k;
  const isVisible = ijk => sliceAxis.value === 'all' || ijk[Number(sliceAxis.value)] === Number(sliceLayer.value);
  const reset = () => view.fit(center, fullExtent);

  function rebuildGrid() {
    clearGroup(gridGroup);
    triangleCells = [];
    visibleIds = [];
    const surfaceVertices = [];
    const lineVertices = [];
    const edgeKeys = new Set();
    for (let id = 0; id < cellCount; id++) {
      const ijk = ijkOf(id);
      if (!isVisible(ijk)) continue;
      visibleIds.push(id);
      const corners = data.cells[id];
      for (let face = 0; face < 6; face++) {
        const neighbor = [...ijk];
        const axis = Math.floor(face / 2);
        neighbor[axis] += face % 2 === 0 ? -1 : 1;
        if (neighbor[axis] >= 0 && neighbor[axis] < counts[axis] && isVisible(neighbor)) continue;
        const vertices = FACES[face];
        for (const corner of [vertices[0], vertices[1], vertices[2], vertices[0], vertices[2], vertices[3]]) {
          surfaceVertices.push(...data.positions[corners[corner]]);
        }
        triangleCells.push(id, id);
        for (let edge = 0; edge < 4; edge++) {
          const a = corners[vertices[edge]];
          const b = corners[vertices[(edge + 1) % 4]];
          const key = a < b ? `${a}:${b}` : `${b}:${a}`;
          if (edgeKeys.has(key)) continue;
          edgeKeys.add(key);
          lineVertices.push(...data.positions[a], ...data.positions[b]);
        }
      }
    }
    const surface = new THREE.BufferGeometry();
    surface.setAttribute('position', new THREE.Float32BufferAttribute(surfaceVertices, 3));
    surface.computeVertexNormals();
    gridMesh = new THREE.Mesh(surface, new THREE.MeshStandardMaterial({ color: 0x8cb3c9, roughness: 0.85, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }));
    gridGroup.add(gridMesh);
    const lines = new THREE.BufferGeometry();
    lines.setAttribute('position', new THREE.Float32BufferAttribute(lineVertices, 3));
    gridGroup.add(new THREE.LineSegments(lines, new THREE.LineBasicMaterial({ color: 0x476a80, opacity: 0.58, transparent: true })));
    document.getElementById('visible-count').textContent = `${visibleIds.length.toLocaleString()} of ${cellCount.toLocaleString()} cells shown`;
    document.getElementById('slice-value').textContent = sliceAxis.value === 'all' ? '—' : sliceLayer.value;
  }

  function sendSelection() {
    if (popup && !popup.closed && currentSelection) {
      popup.postMessage({ type: 'cell-frame-selection', selection: currentSelection }, window.location.origin);
    }
  }

  function selectCell(id, { reveal = false } = {}) {
    if (!Number.isInteger(id) || id < 0 || id >= cellCount) throw new RangeError('Cell ID is outside this grid.');
    selectedId = id;
    const ijk = ijkOf(id);
    if (reveal) {
      const interior = ijk.every((value, axis) => value > 0 && value < counts[axis] - 1);
      if (!isVisible(ijk) || (sliceAxis.value === 'all' && interior)) {
        if (sliceAxis.value === 'all') sliceAxis.value = '0';
        sliceLayer.disabled = false;
        sliceLayer.max = counts[Number(sliceAxis.value)] - 1;
        sliceLayer.value = ijk[Number(sliceAxis.value)];
        rebuildGrid();
      }
    }
    const position = new THREE.Vector3(...data.centers[id]);
    const points = data.cells[id].map(index => data.positions[index]);
    clearGroup(selectedGroup);
    const highlight = makeCellSurface(points, 0xf6c269, 0.3);
    highlight.material.depthTest = false;
    highlight.renderOrder = 8;
    selectedGroup.add(highlight, makeEdges(points, 0xc78922, { overlay: true }));
    if (data.valid[id]) addAxes(selectedGroup, data.frames[id], position, cellSize * 0.8, { overlay: true });
    ijk.forEach((value, axis) => { indexInputs[axis].value = value; });
    document.getElementById('selection-label').textContent = `Cell (${ijk.join(', ')}) · ${data.valid[id] ? 'frame at its deformed center' : 'frame unavailable'}`;
    document.body.dataset.selectedCell = id;
    currentSelection = { id, ijk, F: data.deformation[id], R: data.frames[id], U: data.local_axes[id], determinant: data.determinants[id], valid: data.valid[id], center: data.centers[id], seed: data.metadata.seed, mode: data.metadata.mode };
    axesInspector.update(currentSelection);
    sendSelection();
  }

  sliceAxis.addEventListener('change', () => {
    sliceLayer.disabled = sliceAxis.value === 'all';
    if (!sliceLayer.disabled) {
      const axis = Number(sliceAxis.value);
      sliceLayer.max = counts[axis] - 1;
      sliceLayer.value = ijkOf(selectedId)[axis];
    }
    rebuildGrid();
  });
  sliceLayer.addEventListener('input', () => {
    rebuildGrid();
    const ijk = ijkOf(selectedId);
    ijk[Number(sliceAxis.value)] = Number(sliceLayer.value);
    selectCell(idOf(ijk));
  });
  document.getElementById('cell-picker').addEventListener('submit', event => {
    event.preventDefault();
    const ijk = indexInputs.map(input => Number(input.value));
    if (ijk.some((value, axis) => !Number.isInteger(value) || value < 0 || value >= counts[axis])) return;
    selectCell(idOf(ijk), { reveal: true });
  });
  document.getElementById('focus-cell').addEventListener('click', () => {
    const direction = view.camera.position.clone().sub(view.controls.target);
    view.fit(new THREE.Vector3(...data.centers[selectedId]), cellSize * 5, direction);
  });
  document.getElementById('reset-grid').addEventListener('click', reset);
  document.getElementById('open-detail').addEventListener('click', () => {
    if (popup && !popup.closed) {
      popup.focus();
      sendSelection();
    } else {
      popup = window.open('detail.html', window.CELL_FRAME_CATALOG ? 'learned-intrinsic-multiscale-axes' : 'learned-intrinsic-cell-axes', 'popup,width=700,height=900');
    }
    document.getElementById('popup-status').textContent = popup
      ? 'Linked window open. Select another cell to update both views.'
      : 'The browser blocked the window. Allow popups for this page, or use the axes view here.';
  });
  window.addEventListener('message', event => {
    if (event.origin === window.location.origin && event.source === popup && event.data?.type === 'cell-frame-detail-ready') sendSelection();
  });
  const raycaster = new THREE.Raycaster();
  let pointerStart = null;
  view.renderer.domElement.addEventListener('pointerdown', event => { pointerStart = [event.clientX, event.clientY]; });
  view.renderer.domElement.addEventListener('pointerup', event => {
    if (!pointerStart || event.button !== 0 || Math.hypot(event.clientX - pointerStart[0], event.clientY - pointerStart[1]) > 5) return;
    const rect = view.renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
    raycaster.setFromCamera(pointer, view.camera);
    const hit = raycaster.intersectObject(gridMesh)[0];
    if (hit) selectCell(triangleCells[hit.faceIndex]);
    pointerStart = null;
  });

  document.getElementById('grid-dimensions').textContent = counts.join(' × ');
  document.getElementById('dataset-description').textContent = `${cellCount.toLocaleString()} cells · seed ${data.metadata.seed}`;
  document.getElementById('generation-note').textContent = `Rest cell: ${(cellSize * 1000).toFixed(0)} mm · random deformation amplitude ${data.metadata.deformation_amplitude} · seed ${data.metadata.seed}`;
  indexInputs.forEach((input, axis) => { input.max = counts[axis] - 1; });
  rebuildGrid();
  reset();
  selectCell(idOf([counts[0] - 1, 0, Math.floor(counts[2] / 2)]));
  window.inspector = {
    selectCell,
    getState: () => ({ seed: data.metadata.seed, mode: data.metadata.mode ?? null, loading: document.body.dataset.loading === 'true', selectedCell: selectedId, selectedId, visibleIds: [...visibleIds], sliceAxis: sliceAxis.value, sliceLayer: Number(sliceLayer.value), center: data.centers[selectedId], frameCenter: data.centers[selectedId], frameDirections: [0,1,2].map(axis => column(data.frames[selectedId], axis).toArray()), selection: currentSelection, axisSpace: document.getElementById('axis-space').value }),
    projectCell: id => {
      view.camera.updateMatrixWorld();
      const vector = new THREE.Vector3(...data.centers[id]).project(view.camera);
      const rect = view.renderer.domElement.getBoundingClientRect();
      return { x: rect.left + (vector.x + 1) / 2 * rect.width, y: rect.top + (1 - vector.y) / 2 * rect.height };
    },
    reset,
  };
  setupMultiscale(window.CELL_FRAME_CATALOG, data, payload => {
    data = { ...data, ...payload };
    window.CELL_FRAME_DATA = data;
    boundingBox = new THREE.Box3().setFromPoints(data.positions.map(p => new THREE.Vector3(...p)));
    center = boundingBox.getCenter(new THREE.Vector3());
    fullExtent = boundingBox.getSize(new THREE.Vector3()).length();
    rebuildGrid();
    selectCell(selectedId);
    reset();
    document.getElementById('dataset-description').textContent = `${cellCount.toLocaleString()} cells · seed ${data.metadata.seed}`;
  });
  document.body.dataset.ready = 'true';
} catch (error) {
  const message = document.getElementById('error-message');
  message.hidden = false;
  message.textContent = `Unable to start the 3D inspector: ${error.message}`;
  console.error(error);
}
