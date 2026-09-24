// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
const report=window.BATCH_REPORT;
const samples=report.samples;
const stats=report.summary.statistics;
const q=selector=>document.querySelector(selector);
const format=(value,digits=6)=>Number(value).toFixed(digits);
const scientific=value=>Number(value).toExponential(2);
q('#scale-summary').textContent=Object.entries(report.summary.scale_counts).map(([scale,count])=>`${count} samples used scale ${scale}`).join('; ')+'. All requested the same strength and hierarchy; the scale is recorded rather than silently changing the input distribution.';
q('#numerical-summary').textContent=`Mean unresolved-mode RMS: ${format(stats.null_error_rmse_m.mean*1000)} mm. Mean remaining row-space RMS: ${format(stats.rowspace_error_rmse_m.mean*1e6,3)} µm; maximum ${format(stats.rowspace_error_rmse_m.max*1e6,3)} µm. Fixed-face error is ${scientific(stats.boundary_max_error_m.max)} m across the batch. Float32 numerical error is measured separately, not assumed to vanish.`;
const audit=report.tolerance_audit;
q('#tolerance-audit').textContent=`One additional solve checked the worst-RMS sample (seed ${audit.seed}) at tolerance ${audit.check_tolerance}. Its RMS changes from ${format(audit.main_metrics.corner_rmse_mm)} to ${format(audit.check_metrics.corner_rmse_mm)} mm, while the true equation RMS changes from ${scientific(audit.main_metrics.equation_component_rms)} to ${scientific(audit.check_metrics.equation_component_rms)}. The main 100-sample statistics remain at tolerance ${audit.main_tolerance}; this audit does not replace any batch result.`;
let sortKey='seed',ascending=true;
function value(sample,key){if(key==='seed'||key==='effective_scale')return sample[key];if(key==='min_sampled_jacobian')return sample.screen?.[key];return sample.metrics?.[key];}
function render(){
 const filter=q('#seed-filter').value.trim();
 const shown=samples.filter(sample=>!filter||String(sample.seed)===filter).sort((a,b)=>{
  const x=value(a,sortKey),y=value(b,sortKey);
  return (ascending?1:-1)*((x??Infinity)-(y??Infinity));
 });
 const tbody=q('#samples tbody');tbody.replaceChildren();
 for(const sample of shown){
  const row=document.createElement('tr');row.dataset.seed=sample.seed;
  if(!sample.metrics){const td=document.createElement('td');td.colSpan=10;td.textContent=`Seed ${sample.seed}: ${sample.status} — ${sample.error||'No valid result'}`;row.append(td);tbody.append(row);continue;}
  const metrics=sample.metrics;
  const values=[sample.seed,format(metrics.corner_rmse_mm),format(metrics.corner_max_error_mm),format(metrics.corner_rmse_pct_cell_size,4),format(metrics.original_displacement_rmse_m*1000,3),sample.effective_scale,scientific(metrics.equation_component_rms),format(metrics.rowspace_error_rmse_m*1e6,3),format(sample.screen.min_sampled_jacobian,3)];
  for(const text of values){const td=document.createElement('td');td.textContent=text;row.append(td);}
  const td=document.createElement('td');const name=`seed_${String(sample.seed).padStart(3,'0')}`;
  for(const [label,href]of[['JSON',`samples/${name}.json`],['NPZ',sample.archive]]){const a=document.createElement('a');a.textContent=label;a.href=href;td.append(a);}
  if(report.selected_cases.some(item=>item.seed===sample.seed)){const a=document.createElement('a');a.href=`selected-cases/index.html?seed=${sample.seed}`;a.textContent='3D';td.append(a);}
  row.append(td);tbody.append(row);
 }
 q('#visible-count').textContent=`Showing ${shown.length} of ${samples.length} attempted seeds. Sorted by ${sortKey} ${ascending?'ascending':'descending'}.`;
 document.querySelectorAll('th[data-key]').forEach(header=>{header.removeAttribute('aria-sort');if(header.dataset.key===sortKey)header.setAttribute('aria-sort',ascending?'ascending':'descending');});
 window.BATCH_TABLE_STATE={sortKey,ascending,visibleSeeds:shown.map(sample=>sample.seed)};
}
document.querySelectorAll('th[data-key]').forEach(header=>header.addEventListener('click',()=>{if(sortKey===header.dataset.key)ascending=!ascending;else{sortKey=header.dataset.key;ascending=true;}render();}));
q('#seed-filter').addEventListener('input',render);render();
