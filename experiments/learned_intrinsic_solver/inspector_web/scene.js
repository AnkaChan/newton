// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

export { THREE };
export const COLORS = [0xdc535d, 0x179f74, 0x377ee0];
export const EDGES = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
export const FACES = [[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]];
export const column = (matrix, axis) => new THREE.Vector3(matrix[0][axis], matrix[1][axis], matrix[2][axis]);

export function createView(element) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf0f5f8);
  const camera = new THREE.PerspectiveCamera(38, 1, 0.0001, 1000);
  camera.up.set(0, 0, 1);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  element.appendChild(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a9bab, 2.5));
  const light = new THREE.DirectionalLight(0xffffff, 2.0);
  light.position.set(2, -3, 4);
  scene.add(light);
  const resize = () => {
    const width = Math.max(1, element.clientWidth);
    const height = Math.max(1, element.clientHeight);
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(element);
  resize();
  const fit = (center, extent, direction = new THREE.Vector3(1.2, -1.8, 0.9)) => {
    const factor = Math.max(1, 1 / camera.aspect);
    const distance = extent * factor / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.2;
    controls.target.copy(center);
    camera.position.copy(center).addScaledVector(direction.clone().normalize(), distance);
    controls.minDistance = extent * 0.035;
    controls.maxDistance = Math.max(20, extent * 20);
    camera.near = Math.max(0.00001, extent * 0.0001);
    camera.far = Math.max(100, extent * 100);
    camera.updateProjectionMatrix();
    controls.update();
  };
  const animate = () => {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  };
  animate();
  return { scene, camera, renderer, controls, fit, element };
}

export function clearGroup(group) {
  for (const child of [...group.children]) {
    child.traverse(object => {
      object.geometry?.dispose();
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of materials) {
        material?.map?.dispose();
        material?.dispose();
      }
    });
    group.remove(child);
  }
}

export function textSprite(text, color, size) {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const context = canvas.getContext('2d');
  context.font = 'bold 76px system-ui';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.strokeStyle = '#f0f5f8';
  context.lineWidth = 14;
  context.strokeText(text, 64, 67);
  context.fillStyle = color;
  context.fillText(text, 64, 67);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({ map: texture, depthTest: false, depthWrite: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(size, size, size);
  sprite.renderOrder = 15;
  return sprite;
}

export function addAxes(group, matrix, origin, scale, { overlay = false, labels = true, colorOverride = null } = {}) {
  for (let axis = 0; axis < 3; axis++) {
    const vector = column(matrix, axis);
    const length = vector.length() * scale;
    if (!(length > 0)) continue;
    const color = colorOverride ?? COLORS[axis];
    const arrow = new THREE.ArrowHelper(vector.clone().normalize(), origin, length, color, scale * 0.19, scale * 0.075);
    if (overlay) {
      for (const object of [arrow.line, arrow.cone]) {
        object.material.depthTest = false;
        object.material.depthWrite = false;
        object.renderOrder = 12;
      }
    }
    group.add(arrow);
    if (labels) {
      const label = textSprite(['X', 'Y', 'Z'][axis], `#${color.toString(16).padStart(6, '0')}`, scale * 0.28);
      label.position.copy(origin).addScaledVector(vector, scale * 1.17);
      group.add(label);
    }
  }
}

export function makeEdges(points, color, { overlay = false, opacity = 1 } = {}) {
  const vertices = EDGES.flatMap(([start, end]) => [...points[start], ...points[end]]);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  const material = new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity, depthTest: !overlay, depthWrite: !overlay });
  const lines = new THREE.LineSegments(geometry, material);
  lines.renderOrder = overlay ? 10 : 1;
  return lines;
}

export function makeCellSurface(points, color, opacity = 1) {
  const vertices = FACES.flatMap(face => [face[0],face[1],face[2],face[0],face[2],face[3]].flatMap(i => points[i]));
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.computeVertexNormals();
  return new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color, roughness: 0.9, side: THREE.DoubleSide, transparent: opacity < 1, opacity, depthWrite: opacity >= 1, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }));
}

export function matrixPoints(matrix) {
  const points = [];
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) for (let k = 0; k < 2; k++) {
    points.push(column(matrix, 0).multiplyScalar(i).addScaledVector(column(matrix, 1), j).addScaledVector(column(matrix, 2), k).toArray());
  }
  return points;
}
