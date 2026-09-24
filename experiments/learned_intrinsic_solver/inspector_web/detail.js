// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
import { createAxesInspector } from './axes.js';

const inspector = createAxesInspector();
window.axesInspector = inspector;
const update = selection => {
  inspector.update(selection);
  document.getElementById('link-status').textContent = (selection.mode ? `Seed ${selection.seed} · ${selection.mode} · ` : '') + 'Linked to the deformed grid. Select another cell there to update this view.';
};
window.addEventListener('message', event => {
  if (event.origin !== window.location.origin || event.source !== window.opener) return;
  if (event.data?.type === 'cell-frame-selection') update(event.data.selection);
});
if (window.opener && !window.opener.closed) {
  window.opener.postMessage({ type: 'cell-frame-detail-ready' }, window.location.origin);
} else {
  document.getElementById('link-status').textContent = 'Open this window using “Open window” on the grid inspector.';
}
