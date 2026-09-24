// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const data = window.ROUND_TRIP_DATA;
const report = data.report;
const q = selector => document.querySelector(selector);
const caseSelect = q('#case');
const decoderSelect = q('#decoder');
const gainControl = q('#gain');
const metric = number => number === 0 ? '0' : number.toExponential(2);
const mm = number => number < 1e-9 ? `${metric(number * 1000)} mm` : `${(number * 1000).toFixed(6)} mm`;

q('#grid-summary').textContent = `${report.cell_counts.join(' × ')} cells · ${report.corner_count.toLocaleString()} shared corners`;
for (const item of report.cases) {
  const option = document.createElement('option');
  option.value = item.id;
  option.textContent = item.label;
  caseSelect.append(option);
}

const [nx, ny, nz] = report.cell_counts;
const vertex = (x, y, z) => x * (ny + 1) * (nz + 1) + y * (nz + 1) + z;
const faces = [];
const edges = new Map();
function addQuad(a, b, c, d) {
  faces.push(a, b, c, a, c, d);
  for (const [i, j] of [[a,b],[b,c],[c,d],[d,a]]) {
    edges.set(`${Math.min(i,j)}:${Math.max(i,j)}`, [i,j]);
  }
}
for (const x of [0,nx]) for (let y=0;y<ny;y++) for (let z=0;z<nz;z++) {
  addQuad(vertex(x,y,z),vertex(x,y+1,z),vertex(x,y+1,z+1),vertex(x,y,z+1));
}
for (const y of [0,ny]) for (let x=0;x<nx;x++) for (let z=0;z<nz;z++) {
  addQuad(vertex(x,y,z),vertex(x+1,y,z),vertex(x+1,y,z+1),vertex(x,y,z+1));
}
for (const z of [0,nz]) for (let x=0;x<nx;x++) for (let y=0;y<ny;y++) {
  addQuad(vertex(x,y,z),vertex(x+1,y,z),vertex(x+1,y+1,z),vertex(x,y+1,z));
}
const edgeIndices = [...edges.values()].flat();

function createView(element) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf0f5f8);
  const camera = new THREE.PerspectiveCamera(36, 1, 0.001, 100);
  camera.up.set(0,0,1);
  const renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  element.append(renderer.domElement);
  const controls = new OrbitControls(camera,renderer.domElement);
  controls.enableDamping = false;
  scene.add(new THREE.HemisphereLight(0xffffff,0x6d8794,2.5));
  const light = new THREE.DirectionalLight(0xffffff,2);
  light.position.set(2,-3,4);
  scene.add(light);
  const group = new THREE.Group();
  scene.add(group);
  const resize = () => {
    renderer.setSize(element.clientWidth,element.clientHeight);
    camera.aspect = element.clientWidth/element.clientHeight;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(element);
  resize();
  return {scene,camera,renderer,controls,group};
}
const views = [createView(q('#original-view')),createView(q('#recovered-view'))];
let syncing = false;
for (let i=0;i<2;i++) views[i].controls.addEventListener('change',()=>{
  if (syncing) return;
  syncing = true;
  const from=views[i], to=views[1-i];
  to.camera.position.copy(from.camera.position);
  to.camera.quaternion.copy(from.camera.quaternion);
  to.controls.target.copy(from.controls.target);
  to.controls.update();
  syncing = false;
});

function clear(group) {
  for (const object of [...group.children]) {
    object.geometry?.dispose();
    object.material?.dispose();
    group.remove(object);
  }
}
function errorColor(t) {
  const blue = new THREE.Color(0x237dad), yellow = new THREE.Color(0xf2d16b), red = new THREE.Color(0xdc564b);
  return t<0.5 ? blue.lerp(yellow,t*2) : yellow.lerp(red,(t-0.5)*2);
}
function draw(view,positions,errors=null) {
  clear(view.group);
  const maxError = Math.max(...(errors || [1]));
  const flat = new Float32Array(positions.flat());
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.BufferAttribute(flat,3));
  geometry.setIndex(faces);
  geometry.computeVertexNormals();
  if (errors) {
    const colors = new Float32Array(positions.length*3);
    for (let i=0;i<positions.length;i++) errorColor(maxError>1e-12 ? errors[i]/maxError : 0).toArray(colors,3*i);
    geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));
  }
  const material = new THREE.MeshStandardMaterial({color:errors?0xffffff:0x3cabc1,vertexColors:!!errors,side:THREE.DoubleSide,roughness:0.8,metalness:0});
  view.group.add(new THREE.Mesh(geometry,material));
  const lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute('position',new THREE.BufferAttribute(flat,3));
  lineGeometry.setIndex(edgeIndices);
  view.group.add(new THREE.LineSegments(lineGeometry,new THREE.LineBasicMaterial({color:0x1b465b,transparent:true,opacity:0.28,depthTest:false})));
  const fixedGeometry = new THREE.BufferGeometry();
  fixedGeometry.setAttribute('position',new THREE.Float32BufferAttribute(data.fixed_indices.flatMap(i=>positions[i]),3));
  view.group.add(new THREE.Points(fixedGeometry,new THREE.PointsMaterial({color:0xf18b30,size:4,sizeAttenuation:false,depthTest:false})));
}
function current() {
  const index = report.cases.findIndex(item=>item.id===caseSelect.value);
  return {info:report.cases[index],shape:data.cases[index]};
}
function resetView() {
  const {shape}=current();
  const bounds=new THREE.Box3().setFromPoints(shape.original.map(p=>new THREE.Vector3(...p)));
  const center=bounds.getCenter(new THREE.Vector3());
  const radius=bounds.getSize(new THREE.Vector3()).length()/2;
  const direction=new THREE.Vector3(1.4,-2.0,0.65).normalize();
  for (const view of views) {
    const distance=radius/Math.sin(THREE.MathUtils.degToRad(view.camera.fov/2))*1.08*Math.max(1,1/view.camera.aspect);
    view.controls.target.copy(center);
    view.camera.position.copy(center).addScaledVector(direction,distance);
    view.controls.update();
  }
}
function update({fit=false}={}) {
  const {info,shape}=current();
  const name=decoderSelect.value, gain=Number(gainControl.value);
  const recovered=shape.recovered[name], metrics=info.decoders[name];
  const errors=recovered.map((p,i)=>Math.hypot(...p.map((x,a)=>x-shape.original[i][a])));
  const displayed=recovered.map((p,i)=>p.map((x,a)=>shape.original[i][a]+gain*(x-shape.original[i][a])));
  draw(views[0],shape.original);
  draw(views[1],displayed,errors);
  q('#description').textContent=info.description;
  q('#corner-rms').textContent=mm(metrics.corner_rmse_m);
  q('#corner-max').textContent=mm(metrics.corner_max_error_m);
  q('#center-max').textContent=`${metric(metrics.center_max_error_m)} m`;
  q('#f-max').textContent=metric(metrics.gradient_max_abs_error);
  q('#boundary-max').textContent=`${metric(metrics.boundary_max_error_m)} m`;
  q('#legend-max').textContent=`0 → ${mm(metrics.corner_max_error_m)}`;
  q('#gain-value').textContent=`${gain}×`;
  q('#gain-note').textContent=gain===1?'':`error displayed at ${gain}×`;
  q('#display-note').textContent=gain===1?'Geometry is shown at its measured coordinates. Colors show unscaled recovery error.':`The reconstructed view displays original + ${gain} × (recovered − original) to make the small error visible. Numeric metrics and colors remain unscaled.`;
  q('#sample-result').textContent=`For this shape, extracting R and U then multiplying them reproduces the sampled F to ${metric(info.polar_F_max_abs_error)} per entry. Reconstructing shared corners from ${name==='centers_axes'?'origins, R and U':'R and U'} plus the fixed end gives ${mm(metrics.corner_rmse_m)} RMS error and ${mm(metrics.corner_max_error_m)} maximum error. ${metrics.corner_max_error_m<1e-10?'This identity case is recovered to roundoff.':'The recovered shape satisfies the stored measurements closely, but the free corners are not exactly recovered.'}`;
  q('#download-state').href=`data/${info.id}.npz`;
  document.querySelectorAll('#results tr').forEach(row=>row.classList.toggle('selected',row.dataset.id===info.id));
  if (fit) resetView();
  window.ROUND_TRIP_VIEW_STATE={case:info.id,decoder:name,gain,metrics};
}
for (const info of report.cases) {
  const metrics=info.decoders.centers_axes;
  const row=document.createElement('tr');
  row.dataset.id=info.id;
  for (const text of [info.label,(metrics.corner_rmse_m*1000).toPrecision(6),(metrics.corner_max_error_m*1000).toPrecision(6),metric(metrics.gradient_max_abs_error),metric(metrics.center_max_error_m),metric(metrics.boundary_max_error_m)]) {
    const td=document.createElement('td');td.textContent=text;row.append(td);
  }
  row.addEventListener('click',()=>{caseSelect.value=info.id;gainControl.value=1;update({fit:true});q('.controls').scrollIntoView({block:'start',behavior:'smooth'});});
  q('#results').append(row);
}
const rank=report.small_grid_rank_audit.centers_axes;
q('#rank-audit').textContent=`Independent small-grid audit: a 2 × 2 × 3 grid with the fixed end has ${rank.free_scalar_corner_unknowns} free scalar corner unknowns and ${rank.rows} center/gradient rows, yet rank ${rank.rank}, leaving ${rank.nullity_per_world_component} unresolved scalar directions per world coordinate. More rows than unknowns does not ensure a unique solution.`;
caseSelect.addEventListener('change',()=>{gainControl.value=1;update({fit:true});});
decoderSelect.addEventListener('change',()=>update());
gainControl.addEventListener('input',()=>update());
q('#reset-view').addEventListener('click',resetView);
update({fit:true});
function animate(){requestAnimationFrame(animate);for(const view of views)view.renderer.render(view.scene,view.camera);}
animate();
