// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
const stages = [
  { title: "Local cell inputs", place: "GPU · frame frozen", source: "mixed_physics.py#L275",
    forward: "Gather shared corners, calculate center deformation, extract a fixed rotation frame, and pack local axes, inertial offsets, boundary flags, damping state, and neighbor geometry.",
    backward: "The SVD frame is held fixed. The trainer detached the input state at the query boundary, so gradients do not connect to earlier solver iterations or physical timesteps." },
  { title: "Transformer", place: "GPU · learned weights", source: "network.py#L331",
    forward: "Encode 95 node values and 24 edge values, apply material-conditioned attention over 27 slots, then the feed-forward residual. There is one 128-channel block in this configuration.",
    backward: "Attention, FiLM, the MLPs, and encoders receive derivatives from the output heads. PyTorch accumulates a gradient for every participating network parameter." },
  { title: "Axis proposal", place: "GPU · local → world", source: "mixed_physics.py#L380",
    forward: "Predict nine bounded correction values per cell and one step per object. Add the scaled correction to the local axes. Rotate the target-minus-current axes into world coordinates.",
    backward: "The fixed frame rotates the world-axis gradient back into local coordinates. Derivatives continue through the bounded correction, sigmoid step controller, and network weights." },
  { title: "Assemble & accept", place: "CPU solve · GPU acceptance", source: "fusion.py#L39",
    forward: "Solve for one displacement per shared corner using the cached sparse factor. Enforce prescribed corners exactly. With the saved geometry guard enabled, shorten invalid proposals before energy evaluation.",
    backward: "The detached acceptance scale multiplies the position gradient. A transposed CPU sparse solve transfers that gradient to the proposed world-axis increments. The factorization is fixed; the custom backward supplies the derivative." },
  { title: "Physical energy", place: "GPU · eight samples per cell", source: "mixed_physics.py#L338",
    forward: "Evaluate logarithmic Neo-Hookean elasticity, the implicit-Euler inertia term, and metric damping. The inertial target and physical-start damping anchor stay fixed during the inner solve.",
    backward: "Autograd differentiates the energy with respect to the accepted positions. This derivative contains the inertia residual as well as elastic and damping contributions; it is not fed into the network as an input." },
  { title: "Loss & Adam", place: "GPU · one queried update", source: "train_mixed.py#L645",
    forward: "Normalize energy change using the first candidate's energy and a 1 J floor; add an uphill penalty. Average the query losses, backpropagate, and take one Adam update. Save detached output positions for the pool.",
    backward: "The scalar loss starts backpropagation. Initial and previous energies are constants for this update. DDP averages weight gradients across ranks; Adam uses the accumulated parameter gradients." }
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
