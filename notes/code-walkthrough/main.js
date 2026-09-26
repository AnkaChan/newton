// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
const stages = [
  { title: "Local cell inputs", place: "GPU · frames frozen", source: "input_assembly.py#L185",
    forward: "Gather the eight corners of each cell and form the center deformation F, shape [B, 4000, 3, 3]. Take the closest proper rotation R by SVD (tie cells use the clamped-face reference) and the local axes A = RᵀF. Pack 61 state values per cell: Rᵀ(F_Y − F) (inertial offset), Rᵀ(F − F_start) (physical change), the normalized projected energy gradient, the previous query's normalized gradient and achieved update, six exposed-face and eight fixed-corner flags, the log of the gradient RMS and the history flag. Build 24 edge values for each of 27 neighbor slots and six material conditioning channels.",
    backward: "R is computed under no_grad and detached; the gradient feature and the history blocks are detached too. The local axes and the two deformation-difference blocks stay differentiable in the candidate positions, but the trainer detached the candidate at the query boundary, so no gradient reaches earlier iterations or physical timesteps." },
  { title: "Transformer", place: "GPU · learned weights", source: "network.py#L304",
    forward: "Concatenate the 9 axis values with the 61 state values into 70 node values and encode them to 128 channels; encode the 24 edge values to 64 channels and the six conditioning channels to a 128-channel FiLM signal. One block follows: LayerNorm, FiLM, four-head masked attention over the 27 slots with a learned per-edge score bias and value addition, an output projection, then LayerNorm, FiLM and a SiLU feed-forward residual.",
    backward: "Attention, FiLM, the encoders and the MLPs receive derivatives from the output heads. Masked slots carry exactly zero weight, so padding never contributes. PyTorch accumulates a gradient for each of the 323,214 parameters." },
  { title: "Axis proposal and per-cell step", place: "GPU · local → world", source: "network.py#L352",
    forward: "After a final LayerNorm, a linear head emits nine raw values per cell, bounded to a norm below one by raw / sqrt(1 + |raw|²). A second head emits one sigmoid per cell scaled to a step in (0, 0.05). The local target is A + step × correction. The solver step rotates the difference back to world coordinates: ΔF = R (A_target − A).",
    backward: "The fixed frame rotates the world-increment gradient back into local coordinates. Derivatives continue through the bounded correction and the sigmoid step of every cell into the heads and the transformer. Both heads start at zero, so the first proposals are ΔF = 0 with step 0.025." },
  { title: "Assemble", place: "CPU sparse solve", source: "fusion.py#L39",
    forward: "For each object, solve one sparse least-squares system for the displacement of every free corner so that the Gauss-point gradients of the displacement match the requested world increments, weighted per cell by a stiffness measure times rest cell volume. Prescribed corners are assigned their fixed positions exactly. The result is used as is: no acceptance test, backtracking or step shortening.",
    backward: "The custom backward runs the transposed solve on the same cached PARDISO factor and returns cotangents for the world axis increments, the base positions and the prescribed positions. The factorization itself is fixed and never differentiated." },
  { title: "Physical energy", place: "GPU · eight Gauss points per cell", source: "mixed_physics.py#L389",
    forward: "Evaluate the stable Neo-Hookean density ψ = μ/2 ‖H‖² − μ (s₂ + s₃) + (λ + μ)/2 (tr H + s₂ + s₃)² with H = F − I at the eight Gauss points of every cell, the implicit-Euler inertia Σ m |X − Y|² / (2 dt²), and the metric damping η Σ w ‖FᵀF − F_startᵀF_start‖² / (2 dt). Inverted or collapsed Gauss points give finite values.",
    backward: "Autograd differentiates the total energy with respect to the assembled positions and hands that gradient to the assembly backward. The inertial prediction Y and the physical-start positions are constants for the query." },
  { title: "Loss, Adam and history", place: "GPU · one queried update", source: "train_mixed.py#L700",
    forward: "Before the query, evaluate E_before at the candidate under no_grad and the per-object energy floor. The loss is asinh(E_after / s) + relu((E_after − E_before) / s) with s = max(|E_before|, floor), averaged over the batch. Backpropagate, take one Adam step, then store the query's world axis gradient and achieved ΔF into each trajectory payload and keep the detached assembled positions as the next candidate.",
    backward: "The scalar loss starts backpropagation; only E_after is differentiable. DDP averages the weight gradients across ranks before Adam applies them. The stored history and the next candidate are detached, so the next query begins a fresh graph." }
];
let mode = "forward";
let selected = 0;
const nodeContainer = document.getElementById("flow-nodes");
const detail = document.getElementById("flow-detail");
function renderFlow() {
  nodeContainer.replaceChildren();
  const order = mode === "forward" ? stages.map((_, i) => i) : stages.map((_, i) => i).reverse();
  order.forEach((index, orderIndex) => {
    const stage = stages[index];
    const button = document.createElement("button");
    button.type = "button";
    button.className = "flow-node";
    button.setAttribute("aria-pressed", String(index === selected));
    button.innerHTML = `<small>${orderIndex + 1} · ${stage.place}</small>${stage.title}`;
    button.addEventListener("click", () => { selected = index; renderFlow(); });
    nodeContainer.append(button);
  });
  const stage = stages[selected];
  const [file, anchor] = stage.source.split("#");
  detail.innerHTML = `<h3>${stage.title}</h3><p>${stage[mode]}</p><a href="source/${file}.html#${anchor}" target="_blank" rel="noopener">Open the implementation ↗</a>`;
}
document.querySelectorAll("[data-mode]").forEach(button => button.addEventListener("click", () => {
  mode = button.dataset.mode;
  selected = mode === "forward" ? 0 : stages.length - 1;
  document.querySelectorAll("[data-mode]").forEach(other => other.setAttribute("aria-pressed", String(other === button)));
  renderFlow();
}));
renderFlow();
const toggleCode = document.getElementById("toggle-code");
toggleCode.addEventListener("click", () => {
  const expand = toggleCode.getAttribute("aria-pressed") !== "true";
  document.querySelectorAll("details.code-excerpt").forEach(block => { block.open = expand; });
  toggleCode.setAttribute("aria-pressed", String(expand));
  toggleCode.textContent = expand ? "Collapse code excerpts" : "Expand code excerpts";
});
document.querySelectorAll("[data-copy]").forEach(button => button.addEventListener("click", async () => {
  const lines = button.closest("details").querySelectorAll(".code-text");
  const text = Array.from(lines, line => line.textContent).join("\n");
  try { await navigator.clipboard.writeText(text); button.textContent = "Copied"; }
  catch { button.textContent = "Select code to copy"; }
  setTimeout(() => { button.textContent = "Copy code"; }, 1800);
}));
const navLinks = document.querySelectorAll(".sidebar nav a");
const observer = new IntersectionObserver(entries => {
  for (const entry of entries) if (entry.isIntersecting) {
    navLinks.forEach(link => link.classList.toggle("active", link.hash === `#${entry.target.id}`));
  }
}, { rootMargin: "-5% 0px -65% 0px", threshold: 0 });
document.querySelectorAll("main > section[id],main > header[id]").forEach(section => observer.observe(section));
