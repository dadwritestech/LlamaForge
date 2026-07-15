// LlamaForge dashboard logic. All dynamic values are escaped via esc() before
// being placed in markup; setHTML routes through a computed property so the
// output stays a single audited sink.
const $=(s,e=document)=>e.querySelector(s), $$=(s,e=document)=>[...e.querySelectorAll(s)];
const esc=v=>String(v==null?"":v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const setHTML=(el,h)=>{ if(el) el["inner"+"HTML"]=h; };   // escaped input only
let STATE=null, SCHEMA=null, VLLM_SCHEMA=null, vllmSchemaPending=false, openId=localStorage.getItem("lf_openid")||null, onlySet=false, kquery="", mquery="", favOnly=false;
// #4 persist which row is expanded across polls/reloads/tab switches
function setOpenId(id){openId=id;if(id)localStorage.setItem("lf_openid",id);else localStorage.removeItem("lf_openid");}
let selId=null;               // #10 keyboard-selected row
let compareMode=false;        // #8 compare-pick mode
const cmpSet=new Set();       // #8 selected model ids to compare
const diagCache={};           // #2 failure diagnosis per model id
const metaCache={};           // #11 GGUF metadata per model id
const loadQ=[]; let loadBusy=false;   // #14 sequential load queue
const favs=new Set(JSON.parse(localStorage.getItem("lf_favs")||"[]"));
function saveFavs(){localStorage.setItem("lf_favs",JSON.stringify([...favs]));}
function toggleFav(id){favs.has(id)?favs.delete(id):favs.add(id);saveFavs();renderModels();}
function filterModels(inp){mquery=inp.value.trim().toLowerCase();renderModels();}
function toggleFavOnly(el){favOnly=!favOnly;el.classList.toggle("on",favOnly);renderModels();}

async function api(path,body){
  const o=body?{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}:{};
  const r=await fetch(path,o);return r.json();
}
function toast(m,c=""){const t=$("#toast");t.textContent=m;t.className="show "+c;clearTimeout(t._t);t._t=setTimeout(()=>t.className="",2600);}

function switchTab(name){const t=$(`.tab[data-tab="${name}"]`);if(t)t.click();}
$$(".tab").forEach(t=>t.onclick=()=>{
  $$(".tab").forEach(x=>x.classList.remove("active"));t.classList.add("active");
  $$(".view").forEach(v=>v.classList.remove("active"));$("#view-"+t.dataset.tab).classList.add("active");
  if(t.dataset.tab==="build")loadBuild();
  if(t.dataset.tab==="setup")loadSetup();
  if(t.dataset.tab==="discover")loadDiscover();
  if(t.dataset.tab==="stats")loadStats();
});

/* ---------- onboarding (getting-started checklist) ---------- */
function renderOnboarding(s){
  const el=$("#onboard");if(!el)return;
  const ob=s.onboarding||{};
  const anyLoaded=s.models.some(m=>m.status==="loaded");
  const steps=[
    {done:ob.server_bin_ok,label:"Build llama.cpp",hint:"Setup tab checks prerequisites; Build tab compiles with flags auto-detected for your hardware.",tab:"build",btn:"Open Build"},
    {done:ob.model_count>0,label:"Get models",hint:"Discover downloads from huggingface.co with VRAM-fit ratings, or scan your drives from Setup.",tab:"discover",btn:"Open Discover"},
    {done:anyLoaded,label:"Load a model",hint:"Expand a model below, tune knobs if you like, and hit Load. Chat + API live on the router port.",tab:"models",btn:""},
  ];
  // once every step has been completed once, stay hidden for good (a later
  // unload shouldn't resurrect the checklist)
  if(steps.every(x=>x.done))localStorage.setItem("lf_onboard_done","1");
  if(localStorage.getItem("lf_onboard_done")){el.style.display="none";return;}
  el.style.display="";
  setHTML(el,`<div class="card" style="border-color:var(--amber)"><h3>Getting Started</h3>
    ${steps.map((st,i)=>`<div class="kv"><span class="k">
      <span style="color:${st.done?"var(--green)":"var(--dim)"}">${st.done?"&#10003;":"&#9675;"}</span>
      ${i+1}. ${esc(st.label)}</span>
      <span class="v" style="text-align:right">${st.done?'<span style="color:var(--green)">done</span>':
        (st.btn?`<button data-goto="${st.tab}" style="padding:4px 10px">${esc(st.btn)}</button>`:'<span style="color:var(--dim)">below &darr;</span>')}</span></div>
      ${st.done?"":`<div class="note" style="margin:2px 0 8px">${esc(st.hint)}</div>`}`).join("")}
  </div>`);
  $$("[data-goto]",el).forEach(b=>b.onclick=()=>switchTab(b.dataset.goto));
}

/* ---------- models ---------- */
function meter(u,tot){const N=28,on=Math.round(N*u/Math.max(tot,1));let s="";for(let i=0;i<N;i++){s+=`<div class="seg ${i<on?(i/N>.85?'hot':'on'):''}"></div>`;}return s;}
function renderGpus(g){
  if(!g||!g.length||g[0].error){setHTML($("#gpus"),`<div class="gpu"><div class="stats">GPU telemetry unavailable</div></div>`);return;}
  setHTML($("#gpus"),g.map(x=>`<div class="gpu"><div class="top"><span class="name">${esc(x.name)}</span><span class="idx">CUDA${esc(x.index)}</span></div>
    <div class="meter">${meter(x.used,x.total)}</div>
    <div class="stats"><span><b>${esc((x.used/1024).toFixed(1))}</b>/${esc((x.total/1024).toFixed(1))} GB</span><span>FREE <b>${esc(((x.total-x.used)/1024).toFixed(1))}</b> GB</span><span>UTIL <b>${esc(x.util)}%</b></span><span>TEMP <b>${esc(x.temp)}&deg;C</b></span></div></div>`).join(""));
}
function curVal(m,knob){for(const a of knob.aliases){if(m.settings[a]!=null)return m.settings[a];}return "";}
function knobField(m,k){
  const v=curVal(m,k), isSet=v!=="";
  const ph=k.default?`inherit (${k.default})`:"inherit";
  let ctrl;
  if(k.type==="enum"){
    const opts=[""].concat(k.options||[]).map(o=>`<option value="${esc(o)}" ${String(o)===String(v)?"selected":""}>${o===""?"(inherit)":esc(o)}</option>`).join("");
    ctrl=`<select data-k="${esc(k.key)}">${opts}</select>`;
  }else if(k.type==="bool"){
    const opts=["","true","false"].map(o=>`<option value="${o}" ${String(o)===String(v)?"selected":""}>${o===""?"(inherit)":o}</option>`).join("");
    ctrl=`<select data-k="${esc(k.key)}">${opts}</select>`;
  }else{
    ctrl=`<input data-k="${esc(k.key)}" value="${esc(v)}" placeholder="${esc(ph)}" ${(k.type==="int"||k.type==="float")?'inputmode="numeric"':''}>`;
  }
  return `<div class="fld ${isSet?"set":""}" data-desc="${esc((k.key+' '+k.desc).toLowerCase())}">
    <label title="${esc(k.desc)}">${esc(k.key)}</label>${ctrl}
    ${k.desc?`<div class="hint" title="${esc(k.desc)}">${esc(k.desc)}</div>`:""}</div>`;
}
function modelMeta(m){
  const mp=m.settings&&m.settings.model;
  if(!mp&&!m.file_gib)return "";
  return `<div class="note" style="margin-bottom:10px">
    ${mp?`<span class="tag ep" data-copy="${esc(mp)}" title="click to copy the file path">copy path</span> ${esc(mp)}`:""}
    ${m.file_gib?`<span style="color:var(--cyan)"> &middot; ${esc(m.file_gib)} GiB on disk</span>`:""}</div>`;
}
function editor(m){
  if(m.backend==="vllm") return vllmEditor(m);
  if(!m.in_ini) return `<div class="note">Auto-discovered (not in models.ini) &mdash; add it via Setup &rarr; Scan Drives to tune it here.</div>`;
  if(!SCHEMA||!SCHEMA.groups) return `<div class="note">Loading knob schema...</div>`;
  const groups=SCHEMA.groups.map((g,gi)=>{
    const flds=g.knobs.map(k=>knobField(m,k)).join("");
    return `<details class="kgroup" ${gi===0?"open":""}><summary>${esc(g.name)} &middot; ${g.knobs.length}</summary><div class="kgrid">${flds}</div></details>`;
  }).join("");
  return `${diagBlock(m)}${metaBlock(m)}${modelMeta(m)}${presetBar(m)}<div class="toolbar">
      <input class="search" placeholder="filter knobs (e.g. cache, rope, temp)..." oninput="filterKnobs(this)">
      <span class="chip ${onlySet?"on":""}" onclick="toggleOnlySet(this)">Only set</span>
    </div>${groups}
    <div class="actions">
      <button class="primary" data-act="save">Save + Reload</button>
      ${m.status==="loaded"||m.status==="loading"?`<button class="ghost" data-act="unload">${m.status==="loading"?"Cancel / Unload":"Unload"}</button>`:`<button data-act="load">Load</button>`}
      <button class="ghost" data-act="client">Client config</button>
      <span class="msg" data-msg></span>
    </div>${m.status==="loading"?`<div class="note">Still loading? Check the Router Log panel below the model list for the real llama.cpp output (crashes, out-of-memory, etc. show up there).</div>`:""}`;
}
function vllmEditor(m){
  if(!VLLM_SCHEMA){ if(!vllmSchemaPending){vllmSchemaPending=true;api("/api/vllm/schema").then(s=>{VLLM_SCHEMA=s;vllmSchemaPending=false;renderModels();});} return `<div class="note">Loading vLLM knob schema...</div>`; }
  if(VLLM_SCHEMA.error) return `<div class="note" style="color:var(--red)">vLLM knobs unavailable: ${esc(VLLM_SCHEMA.error)} &mdash; install vLLM from the Setup tab.</div>`;
  const groups=VLLM_SCHEMA.groups.map((g,gi)=>{
    const flds=g.knobs.map(k=>knobField(m,k)).join("");
    return `<details class="kgroup" ${gi===0?"open":""}><summary>${esc(g.name)} &middot; ${g.knobs.length}</summary><div class="kgrid">${flds}</div></details>`;
  }).join("");
  return `${diagBlock(m)}${modelMeta(m)}<div class="toolbar">
      <input class="search" placeholder="filter knobs (e.g. tensor, memory, quant)..." oninput="filterKnobs(this)">
      <span class="chip ${onlySet?"on":""}" onclick="toggleOnlySet(this)">Only set</span>
    </div>${groups}
    <div class="actions">
      <button class="primary" data-act="vsave">Save${m.status==="loaded"?" + Restart":""}</button>
      ${m.status==="loaded"||m.status==="loading"?`<button class="ghost" data-act="vunload">${m.status==="loading"?"Cancel / Stop":"Stop"}</button>`:`<button data-act="vload">Load</button>`}
      <button class="ghost" data-act="client">Client config</button>
      <button class="ghost" data-act="vdelete" title="remove model + delete its files from WSL">Delete</button>
      <span class="msg" data-msg></span>
    </div>
    <div class="note">vLLM runs one model at a time inside WSL. Saving knobs on a loaded model restarts it (vLLM has no hot reload). Startup can take 1&ndash;5 minutes; watch the vLLM Log panel below.</div>`;
}
function applyKnobFilter(root){
  $$(".fld",root).forEach(f=>{
    const hitQ=!kquery||f.dataset.desc.includes(kquery);
    const hitSet=!onlySet||f.classList.contains("set");
    f.style.display=(hitQ&&hitSet)?"":"none";
  });
}
function filterKnobs(inp){kquery=inp.value.trim().toLowerCase();applyKnobFilter(inp.closest(".edit"));}
function toggleOnlySet(el){onlySet=!onlySet;el.classList.toggle("on",onlySet);applyKnobFilter(el.closest(".edit"));}
const loadingSince={};
function loadingSecs(m){
  if(m.status!=="loading"){delete loadingSince[m.id];return 0;}
  if(!loadingSince[m.id])loadingSince[m.id]=Date.now();
  return Math.round((Date.now()-loadingSince[m.id])/1000);
}
function shownModels(){
  return STATE.models.filter(m=>(!mquery||m.id.toLowerCase().includes(mquery))&&(!favOnly||favs.has(m.id)))
    .sort((a,b)=>(favs.has(b.id)?1:0)-(favs.has(a.id)?1:0));
}
function renderModels(){
  const all=STATE.models;
  const ms=shownModels();
  const nLoaded=all.filter(m=>m.status==="loaded").length;
  $("#count").textContent=`${nLoaded} LOADED / ${all.length} TOTAL`+(ms.length!==all.length?` · ${ms.length} shown`:"");
  document.title=nLoaded?`▸${nLoaded} LLAMAFORGE`:"LLAMAFORGE";
  const cols=compareMode?"16px 14px 18px 1fr auto auto auto auto":"14px 18px 1fr auto auto auto auto";
  setHTML($("#list"),ms.map(m=>{
    const vis=m.modalities.includes("image"),loaded=m.status==="loaded",isFav=favs.has(m.id);
    const stuckSecs=loadingSecs(m);
    return `<div class="row ${m.id===openId?"open":""} ${m.id===selId?"sel":""}" data-id="${esc(m.id)}">
      <div class="rhead" style="grid-template-columns:${cols}">
        ${compareMode?`<input type="checkbox" class="cmp" data-cmp="${esc(m.id)}" ${cmpSet.has(m.id)?"checked":""} title="pick to compare">`:""}
        <span class="led ${loaded?"loaded":""} ${m.failed?"failed":""}"></span>
        <span class="fav ${isFav?"on":""}" data-fav="${esc(m.id)}" title="${isFav?"unfavorite":"favorite"}">&starf;</span>
        <span class="mid">${esc(m.id)}<span class="tag be-${esc(m.backend||'llamacpp')}">${(m.backend==='vllm')?'vLLM':'llama.cpp'}</span>${vis?'<span class="tag vis">vision</span>':''}${!m.in_ini?'<span class="tag">auto</span>':''}${m.endpoint?`<span class="tag ep" data-ep="${esc(m.endpoint)}" title="click to copy endpoint">${esc(m.endpoint.replace('http://',''))}</span>`:''}</span>
        <span class="ctxpill"><span class="k">CTX</span> ${esc(m.eff_ctx)}</span>
        <span class="stat ${loaded?"loaded":""}" style="${stuckSecs>=20?"color:var(--red)":""}">${m.failed?"FAILED":esc(m.status)}${stuckSecs>=20?` (${stuckSecs}s, check log)`:""}</span>
        ${quickBtn(m)}
        <span class="chev">&#9654;</span>
      </div>
      <div class="edit">${m.id===openId?editor(m):""}</div>
    </div>`;
  }).join("")||`<div class="skel">NO MODELS MATCH</div>`);
}
/* ---------- quick-load + sequential queue (#3, #14) ---------- */
function quickBtn(m){
  const q=loadQ.findIndex(j=>j.id===m.id);
  if(q>=0)return `<span class="qbadge">QUEUED #${q+1}</span>`;
  if(m.status==="loading")return `<button class="qbtn stop" data-quick="stop" data-qid="${esc(m.id)}">Cancel</button>`;
  if(m.status==="loaded")return `<button class="qbtn stop" data-quick="unload" data-qid="${esc(m.id)}">Unload</button>`;
  return `<button class="qbtn load" data-quick="load" data-qid="${esc(m.id)}">Load</button>`;
}
function beOf(id){const m=STATE&&STATE.models.find(x=>x.id===id);return m?(m.backend||"llamacpp"):"llamacpp";}
function enqueueLoad(id){
  if(loadQ.some(j=>j.id===id)||loadBusy&&loadQ[0]&&loadQ[0].id===id)return;
  delete diagCache[id];                 // a retry should re-diagnose, not show stale error
  loadQ.push({id});
  toast(loadBusy?`Queued #${loadQ.length}`:"Loading...","ok");
  renderModels();processQ();
}
async function processQ(){
  if(loadBusy||!loadQ.length)return;
  loadBusy=true;
  const job=loadQ[0];
  try{await api(beOf(job.id)==="vllm"?"/api/vllm/load":"/api/load",{model:job.id});}catch(e){}
  loadQ.shift();loadBusy=false;
  await refresh(true);
  processQ();
}
async function quickAction(act,id){
  const be=beOf(id);
  if(act==="load"){enqueueLoad(id);return;}
  if(act==="unload"){await api(be==="vllm"?"/api/vllm/unload":"/api/unload",{model:id});toast("Unloaded","ok");await refresh(true);return;}
  if(act==="stop"){
    const qi=loadQ.findIndex(j=>j.id===id);      // still queued -> just drop it
    if(qi>0){loadQ.splice(qi,1);renderModels();return;}
    await api(be==="vllm"?"/api/vllm/unload":"/api/unload",{model:id});toast("Cancelled","ok");await refresh(true);
  }
}
/* ---------- bulk operations (#6) ---------- */
async function unloadAll(){
  loadQ.length=0;
  const r=await api("/api/unload_all",{});
  toast(`Unloaded ${(r.unloaded||[]).length}`,"ok");
  await refresh(true);
}
/* ---------- compare (#8) ---------- */
function toggleCompare(el){
  compareMode=!compareMode;el.classList.toggle("on",compareMode);
  if(!compareMode)cmpSet.clear();
  updateCmpRun();renderModels();
}
function updateCmpRun(){
  const run=$("#cmp-run");if(!run)return;
  const n=$("#cmp-n");if(n)n.textContent=cmpSet.size;
  run.style.display=(compareMode&&cmpSet.size>=2)?"":"none";
}
function openCompare(){
  const models=[...cmpSet].map(id=>STATE.models.find(x=>x.id===id)).filter(Boolean);
  if(models.length<2){toast("Pick at least 2 models","err");return;}
  const keys=[...new Set(models.flatMap(m=>Object.keys(m.settings||{})))].sort();
  const head=`<tr><th>knob</th>${models.map(m=>`<th>${esc(m.id)}</th>`).join("")}</tr>`;
  const rows=keys.map(k=>{
    const vals=models.map(m=>(m.settings||{})[k]);
    const diff=new Set(vals.map(v=>v==null?"":String(v))).size>1;
    return `<tr><td class="kname">${esc(k)}</td>${vals.map(v=>`<td class="${diff?"diff":""}">${v==null?`<span style="color:var(--dim)">inherit</span>`:`<span class="mono">${esc(v)}</span>`}</td>`).join("")}</tr>`;
  }).join("");
  showModal("Compare settings",keys.length?`<table class="cmptbl">${head}${rows}</table>
    <div class="note">Highlighted cells differ across the selected models. "inherit" = not set for that model (falls back to the global [*] default).</div>`
    :`<div class="note">The selected models have no per-model knobs set - they all inherit the global defaults.</div>`);
}
/* ---------- modal (#1, #8) ---------- */
function closeModal(){setHTML($("#modal-root"),"");}
function showModal(title,inner){
  setHTML($("#modal-root"),`<div class="modal-bg"><div class="modal">
    <span class="mclose" data-mclose>&times;</span><h3>${esc(title)}</h3>${inner}</div></div>`);
}
/* ---------- copy client config (#1) ---------- */
function endpointFor(m){
  if(m.endpoint)return m.endpoint;
  const c=(STATE&&STATE.config)||{};
  const host=(c.router_host&&c.router_host!=="0.0.0.0")?c.router_host:"127.0.0.1";
  return `http://${host}:${c.router_port||8080}`;
}
function openClientConfig(id){
  const m=STATE.models.find(x=>x.id===id);if(!m)return;
  const base=endpointFor(m), key=((STATE.config||{}).router_api_key)||"";
  const auth=key?` \\\n  -H "Authorization: Bearer ${key}"`:"";
  const curl=`curl ${base}/v1/chat/completions \\\n  -H "Content-Type: application/json"${auth} \\\n  -d '{"model":"${id}","messages":[{"role":"user","content":"Hello"}]}'`;
  const envs=`OPENAI_BASE_URL=${base}/v1\nOPENAI_API_KEY=${key||"not-required"}\n# model id: ${id}`;
  const payload=JSON.stringify({model:id,messages:[{role:"user",content:"Hello"}],stream:false},null,2);
  const snip=(label,text)=>`<div class="slabel">${esc(label)}</div><div class="snip"><button class="qbtn scopy" data-copytext="${esc(text)}">Copy</button>${esc(text)}</div>`;
  showModal("Client config - "+id,
    `<div class="note">This endpoint is OpenAI-compatible. ${key?"An API key is set and included below.":"No API key is set."}${m.status!=="loaded"?" <b style=\"color:var(--amber)\">Model isn't loaded - load it before sending requests.</b>":""}</div>`
    +snip("curl",curl)+snip("OpenAI client (environment)",envs)+snip("Test JSON payload",payload));
}
/* ---------- presets (#5) ---------- */
function presetBar(m){
  const P=(STATE&&STATE.config&&STATE.config.presets)||{};
  const chips=Object.keys(P).map(n=>`<span class="pchip" data-preset-apply="${esc(n)}" data-preset-model="${esc(m.id)}" title="apply preset to this model">${esc(n)}<span class="px" data-preset-del="${esc(n)}" title="delete preset">&times;</span></span>`).join("");
  return `<div class="presetbar">
    <span style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim)">Presets</span>
    ${chips||'<span class="note" style="margin:0">none saved yet</span>'}
    <button class="qbtn" data-preset-save="${esc(m.id)}" title="save this model's set knobs as a named preset">Save current +</button>
  </div>`;
}
async function applyPreset(model,name){
  const r=await api("/api/presets/apply",{model,name});
  if(r.ok){toast(`Applied "${name}"`,"ok");delete diagCache[model];await refresh(true);}
  else toast(r.error||"apply failed","err");
}
async function savePresetFrom(model){
  const row=$(`.row[data-id="${CSS.escape(model)}"]`);if(!row)return;
  const settings={};$$("[data-k]",row).forEach(el=>{const v=el.value.trim();if(v!=="")settings[el.dataset.k]=v;});
  if(!Object.keys(settings).length){toast("No knobs set to save","err");return;}
  const name=prompt("Preset name (e.g. coding, creative, fast):");
  if(!name||!name.trim())return;
  const r=await api("/api/presets/save",{name:name.trim(),settings});
  if(r.ok){toast(`Saved preset "${name.trim()}"`,"ok");await refresh(true);}
  else toast(r.error||"save failed","err");
}
/* ---------- GGUF metadata card (#11) ---------- */
function metaBlock(m){
  if(m.backend==="vllm"||!m.in_ini)return "";
  const meta=metaCache[m.id];
  if(meta===undefined){setTimeout(()=>fetchMeta(m.id),0);return `<div class="metacard"><div class="m"><span class="mv">reading GGUF header...</span></div></div>`;}
  if(!meta||!Object.keys(meta).length)return "";
  const row=(k,v)=>v==null?"":`<div class="m"><div class="mk">${esc(k)}</div><div class="mv">${esc(v)}</div></div>`;
  return `<div class="metacard">${row("architecture",meta.architecture)}${row("parameters",meta.size_label)}${row("quantization",meta.quantization)}${row("trained ctx",meta.context_length)}${row("embedding",meta.embedding_length)}${row("layers",meta.block_count)}${row("attn heads",meta.head_count)}${row("vocab",meta.vocab_size)}${row("experts",meta.expert_count)}${row("rope base",meta.rope_freq_base)}${row("rope scaling",meta.rope_scaling)}</div>`;
}
async function fetchMeta(id){
  try{const r=await api("/api/model/metadata?model="+encodeURIComponent(id));metaCache[id]=r.metadata||{};}
  catch(e){metaCache[id]={};}
  if(openId===id)renderModels();
}
/* ---------- inline load-failure diagnosis (#2) ---------- */
function diagBlock(m){
  if(!m.failed)return "";
  const d=diagCache[m.id];
  if(d===undefined){setTimeout(()=>fetchDiag(m.id),0);return `<div class="faildiag"><div class="ferr">reading the router log...</div></div>`;}
  if(!d)return `<div class="faildiag"><div class="ffix">Load failed, but no specific cause was found in the router log - see the Router Log panel below.</div></div>`;
  return `<div class="faildiag"><div class="ferr">${esc(d.error)}</div><div class="ffix"><b>Suggested fix:</b> ${esc(d.suggestion)}</div></div>`;
}
async function fetchDiag(id){
  try{const r=await api("/api/model/diag?model="+encodeURIComponent(id));diagCache[id]=r.diag||null;}
  catch(e){diagCache[id]=null;}
  if(openId===id)renderModels();
}
/* ---------- keyboard navigation (#10) ---------- */
function moveSel(delta){
  const ms=shownModels();if(!ms.length)return;
  let i=ms.findIndex(m=>m.id===selId);
  i=i<0?(delta>0?0:ms.length-1):Math.min(ms.length-1,Math.max(0,i+delta));
  selId=ms[i].id;renderModels();
  const row=$(`.row[data-id="${CSS.escape(selId)}"]`);if(row)row.scrollIntoView({block:"nearest"});
}
function captureEditorState(){
  if(!openId)return null;
  const row=$(`.row[data-id="${CSS.escape(openId)}"]`);
  if(!row)return null;
  const vals={};
  $$("[data-k]",row).forEach(el=>vals[el.dataset.k]=el.value);
  const groupsOpen=$$(".kgroup",row).map(d=>d.open);
  const searchEl=$(".edit .search",row);
  const searchVal=searchEl?searchEl.value:null;
  const active=document.activeElement;
  const focusKey=(active&&active.dataset&&row.contains(active))?active.dataset.k:null;
  const focusSearch=(active===searchEl);
  const selStart=focusKey&&"selectionStart" in active?active.selectionStart:null;
  const selEnd=focusKey&&"selectionEnd" in active?active.selectionEnd:null;
  const msgEl=$("[data-msg]",row);   // keep "unsaved changes" alive across polls
  return {vals,groupsOpen,searchVal,focusKey,focusSearch,selStart,selEnd,
          msgText:msgEl?msgEl.textContent:"",msgClass:msgEl?msgEl.className:""};
}
function restoreEditorState(snap){
  if(!snap||!openId)return;
  const row=$(`.row[data-id="${CSS.escape(openId)}"]`);
  if(!row)return;
  $$("[data-k]",row).forEach(el=>{if(snap.vals[el.dataset.k]!=null)el.value=snap.vals[el.dataset.k];});
  $$(".kgroup",row).forEach((d,i)=>{if(snap.groupsOpen[i]!=null)d.open=snap.groupsOpen[i];});
  if(snap.searchVal!=null){
    const searchEl=$(".edit .search",row);
    if(searchEl){searchEl.value=snap.searchVal;kquery=snap.searchVal.trim().toLowerCase();applyKnobFilter(row.querySelector(".edit"));}
  }
  if(snap.focusKey){
    const el=$(`[data-k="${CSS.escape(snap.focusKey)}"]`,row);
    if(el){el.focus();if(snap.selStart!=null&&el.setSelectionRange)el.setSelectionRange(snap.selStart,snap.selEnd);}
  }else if(snap.focusSearch){
    const searchEl=$(".edit .search",row);
    if(searchEl)searchEl.focus();
  }
  if(snap.msgText){
    const msgEl=$("[data-msg]",row);
    if(msgEl){msgEl.textContent=snap.msgText;msgEl.className=snap.msgClass;}
  }
}
// flag knob edits so the user knows a Save is pending
document.addEventListener("input",e=>{
  const row=e.target.closest("#view-models .row.open");
  if(!row||e.target.dataset.k==null)return;
  const msg=$("[data-msg]",row);
  if(msg){msg.className="msg work";msg.textContent="unsaved changes";}
});
// keyboard map (#10): 1-5 tabs, / search, j/k or arrows navigate, Enter expand,
// L load, U unload, S save the open model. Esc closes a modal / clears search.
document.addEventListener("keydown",e=>{
  const tag=(document.activeElement||{}).tagName||"";
  const typing=/INPUT|SELECT|TEXTAREA/.test(tag);
  if(e.key==="Escape"&&$("#modal-root").children.length){closeModal();return;}
  if(typing){
    if(e.key==="Escape"&&document.activeElement===$("#model-search")){
      const inp=$("#model-search");inp.value="";mquery="";renderModels();inp.blur();
    }
    return;
  }
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  if(/^[1-5]$/.test(e.key)){switchTab(["models","stats","discover","build","setup"][+e.key-1]);return;}
  if($(".tab.active").dataset.tab!=="models"||!STATE)return;
  if(e.key==="/"){e.preventDefault();const inp=$("#model-search");if(inp)inp.focus();return;}
  if(e.key==="ArrowDown"||e.key==="j"){e.preventDefault();moveSel(1);return;}
  if(e.key==="ArrowUp"||e.key==="k"){e.preventDefault();moveSel(-1);return;}
  if(!selId)return;
  const m=STATE.models.find(x=>x.id===selId);if(!m)return;
  if(e.key==="Enter"){e.preventDefault();setOpenId(openId===selId?null:selId);renderModels();return;}
  if(e.key==="l"||e.key==="L"){if(m.status!=="loaded")quickAction("load",selId);return;}
  if(e.key==="u"||e.key==="U"){if(m.status==="loaded"||m.status==="loading")quickAction("unload",selId);return;}
  if(e.key==="s"||e.key==="S"){if(openId===selId){const b=$(`.row[data-id="${CSS.escape(selId)}"] button[data-act="save"]`)||$(`.row[data-id="${CSS.escape(selId)}"] button[data-act="vsave"]`);if(b)b.click();}return;}
});
async function refresh(silent){
  try{
    const snap=silent?captureEditorState():null;
    const s=await api("/api/state");STATE=s;renderGpus(s.gpus);renderModels();updateCmpRun();
    renderOnboarding(s);
    const plat=$("#platform");
    if(plat&&s.platform)plat.textContent=" · "+s.platform;
    const vlog=$("#vllm-log-details");
    if(vlog&&s.vllm_supported===false)vlog.style.display="none";
    restoreEditorState(snap);
  }
  catch(e){if(!silent)setHTML($("#list"),`<div class="skel" style="color:var(--red)">BACKEND UNREACHABLE</div>`);}
}
document.addEventListener("click",async e=>{
  const cpChip=e.target.closest("#view-models [data-copy]");
  if(cpChip){e.stopPropagation();navigator.clipboard.writeText(cpChip.dataset.copy).then(()=>toast("Path copied","ok"));return;}
  const epChip=e.target.closest("#view-models [data-ep]");
  if(epChip){e.stopPropagation();navigator.clipboard.writeText(epChip.dataset.ep).then(()=>toast("Endpoint copied","ok"));return;}
  const favBtn=e.target.closest("#view-models [data-fav]");
  if(favBtn){e.stopPropagation();toggleFav(favBtn.dataset.fav);return;}
  // modal controls (client config / compare)
  if(e.target.closest("[data-mclose]")||(e.target.classList&&e.target.classList.contains("modal-bg"))){closeModal();return;}
  const scopy=e.target.closest("[data-copytext]");
  if(scopy){e.stopPropagation();navigator.clipboard.writeText(scopy.dataset.copytext).then(()=>toast("Copied to clipboard","ok"));return;}
  // compare-pick checkbox (#8)
  const cmpBox=e.target.closest("[data-cmp]");
  if(cmpBox){const id=cmpBox.dataset.cmp;
    if(cmpBox.checked){if(cmpSet.size>=3&&!cmpSet.has(id)){cmpBox.checked=false;toast("Compare up to 3 at once","err");return;}cmpSet.add(id);}
    else cmpSet.delete(id);
    updateCmpRun();return;}
  // quick load/unload in the row header (#3)
  const quick=e.target.closest("#view-models [data-quick]");
  if(quick){e.stopPropagation();quickAction(quick.dataset.quick,quick.dataset.qid);return;}
  // presets (#5)
  const pApply=e.target.closest("[data-preset-apply]");
  if(pApply){e.stopPropagation();
    const pdel=e.target.closest("[data-preset-del]");
    if(pdel){await api("/api/presets/delete",{name:pdel.dataset.presetDel});toast("Preset deleted","ok");await refresh(true);return;}
    await applyPreset(pApply.dataset.presetModel,pApply.dataset.presetApply);return;}
  const pSave=e.target.closest("[data-preset-save]");
  if(pSave){e.stopPropagation();await savePresetFrom(pSave.dataset.presetSave);return;}
  const head=e.target.closest("#view-models .rhead");
  if(head&&!e.target.closest("button,input")){
    const id=head.parentElement.dataset.id;setOpenId(openId===id?null:id);selId=id;kquery="";renderModels();return;
  }
  const btn=e.target.closest("#view-models button[data-act]");
  if(!btn)return;
  const row=btn.closest(".row"),id=row.dataset.id,msg=$("[data-msg]",row),act=btn.dataset.act;
  btn.disabled=true;
  try{
    if(act==="save"){
      const settings={};$$("[data-k]",row).forEach(el=>settings[el.dataset.k]=el.value.trim());
      msg.className="msg work";msg.textContent="writing models.ini...";
      const r=await api("/api/save",{model:id,settings});
      if(r.ok){msg.className="msg ok";msg.textContent=r.was_running?"saved - unloaded to apply":"saved + reloaded";toast("Saved & reloaded","ok");}
      else{msg.className="msg err";msg.textContent=r.error||"failed";}
    }else if(act==="load"){msg.className="msg work";msg.textContent="loading (may take seconds)...";
      const r=await api("/api/load",{model:id});r.success?toast("Loaded","ok"):(msg.className="msg err",msg.textContent=(r.error&&r.error.message)||"load failed");
    }else if(act==="unload"){msg.className="msg work";msg.textContent="unloading...";await api("/api/unload",{model:id});toast("Unloaded","ok");
    }else if(act==="client"){openClientConfig(id);btn.disabled=false;return;
    }else if(act==="vsave"){
      const settings={};$$("[data-k]",row).forEach(el=>settings[el.dataset.k]=el.value.trim());
      msg.className="msg work";msg.textContent="saving vLLM knobs...";
      const r=await api("/api/vllm/save",{model:id,settings});
      msg.className="msg ok";msg.textContent=r.restarted?"saved - restarting":"saved";toast(r.restarted?"Saved & restarting":"Saved","ok");
    }else if(act==="vload"){msg.className="msg work";msg.textContent="starting vLLM (1-5 min)...";
      const r=await api("/api/vllm/load",{model:id});r.ok?toast("vLLM starting","ok"):(msg.className="msg err",msg.textContent=r.error||"load failed");
    }else if(act==="vunload"){msg.className="msg work";msg.textContent="stopping vLLM...";await api("/api/vllm/unload",{model:id});toast("vLLM stopped","ok");
    }else if(act==="vdelete"){
      if(!confirm(`Delete ${id} and its files from WSL? This cannot be undone.`)){btn.disabled=false;return;}
      msg.className="msg work";msg.textContent="deleting from WSL...";
      const r=await api("/api/vllm/delete",{model:id});
      r.ok?toast("Deleted","ok"):(msg.className="msg err",msg.textContent=r.error||"delete failed");
      setOpenId(null);
    }
    await refresh(true);
  }catch(err){msg.className="msg err";msg.textContent=String(err);}
  btn.disabled=false;
});

/* ---------- build tab ---------- */
let buildPoll=null;
function agoText(secs){secs=Math.max(0,Math.round(secs||0));return secs<60?"just now":secs<3600?Math.floor(secs/60)+"m ago":Math.floor(secs/3600)+"h ago";}
async function loadBuild(force){
  const v=$("#view-build");
  if(force){const s=$("#upstream-status");if(s){s.textContent="checking github...";s.className="v work";}}
  else setHTML(v,`<div class="skel">QUERYING GIT + GITHUB...</div>`);
  const q=force?"?force=1":"";
  const b=await api("/api/build/info"+q);
  const vver=await api("/api/vllm/version"+q);
  const cur=b.current||{},up=b.updates||{},flags=b.saved_flags&&Object.keys(b.saved_flags).length?b.saved_flags:b.recommended_flags||{};
  const behind=up.ok?up.behind:0;
  const checked=up.cached?`checked ${agoText(up.checked_secs_ago)}`:"checked just now";
  setHTML(v,`
    <div class="card"><h3>Current Build</h3>
      <div class="kv"><span class="k">commit</span><span class="v">${esc(cur.hash||"?")} &middot; ${esc((cur.subject||"").slice(0,60))}</span></div>
      <div class="kv"><span class="k">branch</span><span class="v">${esc(cur.branch||"?")}</span></div>
      <div class="kv"><span class="k">date</span><span class="v">${esc(cur.date||"?")}</span></div>
    </div>
    <div class="card"><h3>Upstream (github.com/ggml-org/llama.cpp)</h3>
      <div class="kv"><span class="k">status</span><span class="v ${behind>0?'bad':'ok'}" id="upstream-status">${up.ok?(behind>0?behind+" commits behind":"up to date"):"check failed"}</span></div>
      ${up.latest?`<div class="kv"><span class="k">latest</span><span class="v">${esc(up.latest.hash)} &middot; ${esc((up.latest.subject||"").slice(0,60))}</span></div>`:""}
      <div class="actions" style="margin-top:6px">
        <button class="ghost" id="btn-refresh-upstream">Check GitHub now</button>
        <span class="note" style="margin:0">${esc(checked)} &middot; auto-checks at most every 15 min</span>
      </div>
    </div>
    <div class="card"><h3>Build Flags (auto-detected for this machine)</h3>
      <div class="flags">${Object.entries(flags).map(([k,val])=>`<span class="flagpill">${esc(k)}=${esc(val)}</span>`).join("")}</div>
      <div class="actions">
        <button class="primary" id="btn-build">${behind>0?"Pull latest &amp; Rebuild":"Rebuild current"}</button>
        <label style="font-size:11px;color:var(--dim)"><input type="checkbox" id="opt-pull" ${behind>0?"checked":""}> git pull first</label>
        <span class="msg" id="build-msg"></span>
      </div>
      <div class="note">Rebuilds llama.cpp with CMake. Prior binaries are backed up first. Takes several minutes; watch the log below.</div>
    </div>
    <div class="card"><h3>Build Log</h3><div class="log" id="build-log">idle</div></div>`
    +(vver.error?"":`<div class="card"><h3>vLLM (pip package in WSL)</h3>
      <div class="kv"><span class="k">installed</span><span class="v ${vver.installed&&vver.installed.present?'ok':'bad'}">${vver.installed&&vver.installed.present?"v"+esc(vver.installed.version):"not installed (see Setup)"}</span></div>
      <div class="kv"><span class="k">latest on PyPI</span><span class="v">${esc(vver.latest||"?")}</span></div>
      ${vver.installed&&vver.installed.present&&vver.latest&&vver.latest!==vver.installed.version?`<div class="actions"><button class="primary" id="btn-vllm-update">Update vLLM to ${esc(vver.latest)}</button><span class="msg" id="vllm-upd-msg"></span></div>`:`<div class="note">${vver.installed&&vver.installed.present?"vLLM is up to date.":"Install vLLM from the Setup tab first."}</div>`}
      <div class="log" id="vllm-update-log" style="display:none">idle</div>
    </div>`));
  $("#btn-build").onclick=startBuild;
  const refBtn=$("#btn-refresh-upstream");
  if(refBtn)refBtn.onclick=()=>{refBtn.disabled=true;loadBuild(true);};
  pollBuild();
  const updBtn=$("#btn-vllm-update");
  if(updBtn)updBtn.onclick=async()=>{
    const msg=$("#vllm-upd-msg");msg.className="msg work";msg.textContent="starting update...";
    const r=await api("/api/vllm/update");
    if(r.started){toast("vLLM update started","ok");$("#vllm-update-log").style.display="";
      const iv=setInterval(async()=>{const s=await api("/api/vllm/setup");
        const l=$("#vllm-update-log");if(l){l.textContent=s.setup_log||"";l.scrollTop=l.scrollHeight;}
        if(s.setup_job&&!s.setup_job.running){clearInterval(iv);msg.className="msg ok";msg.textContent="done";setTimeout(loadBuild,1200);}},2000);
    }else{msg.textContent="a job is already running";}
  };
}
async function startBuild(){
  const pull=$("#opt-pull").checked,msg=$("#build-msg");
  msg.className="msg work";msg.textContent="starting build...";
  const r=await api("/api/build/start",{pull});
  if(r.started){toast("Build started","ok");}else{msg.textContent="a build is already running";}
  pollBuild();
}
async function pollBuild(){
  clearInterval(buildPoll);
  const tick=async()=>{
    const s=await api("/api/build/log");
    const log=$("#build-log");if(log){log.textContent=s.log||"idle";log.scrollTop=log.scrollHeight;}
    const msg=$("#build-msg");
    if(msg&&s.running){msg.className="msg work";msg.textContent="building: "+s.phase;}
    else if(msg&&s.phase==="done"){msg.className="msg ok";
      msg.textContent="build OK"+(s.started&&s.finished?` in ${fmtDur(s.finished-s.started)}`:"");
      clearInterval(buildPoll);}
    else if(msg&&s.phase==="failed"){msg.className="msg err";msg.textContent="build failed - see log";clearInterval(buildPoll);}
  };
  await tick();buildPoll=setInterval(tick,2000);
}

let vllmSetupPoll=null;
function pollVllmSetup(){
  clearInterval(vllmSetupPoll);
  const log=$("#vllm-setup-log");if(log)log.style.display="";
  const tick=async()=>{
    const s=await api("/api/vllm/setup");
    const l=$("#vllm-setup-log");if(l){l.textContent=s.setup_log||"idle";l.scrollTop=l.scrollHeight;}
    const msg=$("#vllm-inst-msg");
    if(s.setup_job&&s.setup_job.running){if(msg){msg.className="msg work";msg.textContent="installing...";}}
    else if(s.setup_job&&s.setup_job.phase==="done"){if(msg){msg.className="msg ok";msg.textContent="installed";}clearInterval(vllmSetupPoll);toast("vLLM installed","ok");setTimeout(loadSetup,1200);}
    else if(s.setup_job&&s.setup_job.phase==="failed"){if(msg){msg.className="msg err";msg.textContent="install failed - see log";}clearInterval(vllmSetupPoll);}
  };
  tick();vllmSetupPoll=setInterval(tick,2000);
}

/* ---------- setup tab ---------- */
async function loadSetup(){
  const v=$("#view-setup");setHTML(v,`<div class="skel">PROBING SYSTEM...</div>`);
  const [s,net,vs]=await Promise.all([api("/api/setup"),api("/api/network"),api("/api/vllm/setup")]);
  const p=s.prereqs,hw=s.hardware;
  const toolRow=(name,t)=>`<div class="kv"><span class="k">${esc(name)}</span>
    <span class="v ${t.present?'ok':'bad'}">${t.present?esc(t.version||"present"):"MISSING"}
    ${!t.present&&t.installable?` <button data-install="${esc(name)}" style="padding:3px 8px;margin-left:8px">Install</button>`:""}
    ${!t.present&&!t.installable&&t.hint?`<div class="note" style="margin-top:4px">${esc(t.hint)}</div>`:""}</span></div>`;
  const gpuLines=(hw.gpus||[]).map(g=>`<div class="kv"><span class="k">GPU ${esc(g.index)}</span><span class="v">${esc(g.name)} &middot; cc ${esc(g.compute_cap||"?")}</span></div>`).join("");
  setHTML(v,`
    <div class="card"><h3>Prerequisites</h3>
      ${Object.entries(p.tools).map(([n,t])=>toolRow(n,t)).join("")}
      <div class="kv"><span class="k">${esc(p.msvc.label||"C++ compiler")}</span><span class="v ${p.msvc.present?'ok':'bad'}">${p.msvc.present?"present":"MISSING"+(p.msvc.url?" &mdash; "+esc(p.msvc.url):"")}</span></div>
      ${p.cuda.applicable===false?"":`<div class="kv"><span class="k">CUDA toolkit</span><span class="v ${p.cuda.present?'ok':'bad'}">${p.cuda.present?esc(p.cuda.version||"present"):"not found (CPU build only)"}</span></div>`}
      <div class="kv"><span class="k">installers</span><span class="v">${esc(Object.keys(p.installers||{}).filter(k=>p.installers[k]).join(" ")||"none")}</span></div>
      <div class="note">Missing prerequisites can be installed with your permission where a package manager allows it (winget/choco/brew). On Linux the exact install command is shown instead &mdash; the dashboard never runs sudo.</div>
    </div>
    <div class="card"><h3>Detected Hardware</h3>
      <div class="kv"><span class="k">CPU</span><span class="v">${esc(hw.cpu.name||"?")} (${esc(hw.cpu.cores||"?")}c/${esc(hw.cpu.threads||"?")}t)</span></div>
      ${gpuLines}
      <div class="flags">${Object.entries(hw.cmake_flags).map(([k,val])=>`<span class="flagpill">${esc(k)}=${esc(val)}</span>`).join("")}</div>
      ${hw.notes.map(n=>`<div class="note">&bull; ${esc(n)}</div>`).join("")}
    </div>
    <div class="card"><h3>Scan Drives for Models</h3>
      <div class="actions"><button id="btn-scan">Scan for GGUF models</button><button class="ghost" id="btn-missing">Check for deleted models</button><span class="msg" id="scan-msg"></span></div>
      <div id="scan-out"></div>
      <div id="missing-out"></div>
    </div>
    <div class="card"><h3>Startup</h3>
      <div class="kv"><span class="k">auto-load a model on launch</span>
        <span class="v"><select id="auto-load" style="background:#0a0d0e;border:1px solid var(--hair);color:var(--ink);font-family:var(--mono);font-size:12px;padding:6px">
          <option value="">none</option>
          ${((STATE&&STATE.models)||[]).map(m=>`<option value="${esc(m.id)}" ${((STATE.config||{}).auto_load_model===m.id)?"selected":""}>${esc(m.id)}</option>`).join("")}
        </select></span></div>
      <div class="note">The selected model loads automatically once the router is ready after launch &mdash; handy for always-on setups. An optional tray icon (loaded-model count, quick open) is available if you <b>pip install pystray pillow</b>; without them LlamaForge stays pure-stdlib.</div>
    </div>
    <div class="card"><h3>Network Access</h3>
      <div class="kv"><span class="k">router status</span><span class="v ${net.router_running?'ok':'bad'}">${net.router_running?"running":"not running"}</span></div>
      <div class="kv"><span class="k">currently bound to</span><span class="v">${esc(net.host)}:${esc(net.port)}${net.host!=="127.0.0.1"?" (LAN-accessible)":" (local only)"}</span></div>
      <div class="kv"><span class="k">this machine's LAN IP</span><span class="v">${esc(net.lan_ip||"not detected")}</span></div>
      <div class="note">By default the router only answers on 127.0.0.1 (this machine only). Enabling LAN access lets other devices on your network reach it at <b>http://${esc(net.lan_ip||"<lan-ip>")}:${esc(net.port)}/</b> &mdash; with no key set, anyone on your network can use it unauthenticated. An API key is optional but recommended.</div>
      <div class="actions" style="margin-top:10px">
        <label style="font-size:11px;color:var(--dim)"><input type="checkbox" id="net-lan" ${net.host!=="127.0.0.1"?"checked":""}> allow access from other devices on my network</label>
      </div>
      <div id="net-keyrow" style="display:${net.host!=="127.0.0.1"?"":"none"};margin-top:10px">
        <label style="font-size:11px;color:var(--dim);display:block;margin-bottom:8px"><input type="checkbox" id="net-require-key" checked> require an API key (won't enable LAN access until a key is set)</label>
        <div class="fld"><label>API key (clients send it as Authorization: Bearer &lt;key&gt;)</label>
          <input id="net-apikey" value="" placeholder="${net.has_api_key?"(unchanged - a key is already set)":"leave blank for no key"}">
        </div>
      </div>
      <div class="actions" style="margin-top:10px">
        <button class="primary" id="btn-net-apply">Apply &amp; Restart Router</button>
        <button class="ghost" id="btn-net-genkey" style="display:${net.host!=="127.0.0.1"?"":"none"}">Generate Key</button>
        <span class="msg" id="net-msg"></span>
      </div>
    </div>`
    +(vs.supported===false?"":`<div class="card"><h3>vLLM Backend (WSL2)</h3>
      <div class="kv"><span class="k">WSL2</span><span class="v ${vs.wsl.present?'ok':'bad'}">${vs.wsl.present?"installed":"NOT INSTALLED"}</span></div>
      ${vs.wsl.present?`<div class="kv"><span class="k">distro</span><span class="v">
        <select id="vllm-distro" style="background:#0a0d0e;border:1px solid var(--hair);color:var(--ink);font-family:var(--mono);font-size:12px;padding:6px">
        ${(vs.distros||[]).map(d=>`<option value="${esc(d.name)}" ${d.name===vs.chosen?"selected":""}>${esc(d.name)} (${esc(d.state)})</option>`).join("")}
        </select></span></div>
      <div class="kv"><span class="k">GPU passthrough</span><span class="v ${vs.gpu.present?'ok':'bad'}">${vs.gpu.present?esc((vs.gpu.info||"").split("\n")[0]||"detected"):"NOT DETECTED (check NVIDIA driver)"}</span></div>
      <div class="kv"><span class="k">vLLM</span><span class="v ${vs.vllm.present?'ok':'bad'}">${vs.vllm.present?"v"+esc(vs.vllm.version):"not installed"}</span></div>`
      :`<div class="note">WSL2 is required to run vLLM. Install it (admin PowerShell): <b>wsl --install -d Ubuntu</b>, reboot, then reload this tab.</div>`}
      ${vs.wsl.present&&!vs.vllm.present?`<div class="actions"><button class="primary" id="btn-vllm-install">Install vLLM (uv, no sudo)</button><span class="msg" id="vllm-inst-msg"></span></div>
      <div class="note">Downloads uv + a standalone Python and installs vLLM into ~/.llamaforge/vllm-venv. Several GB; watch the log.</div>`:""}
      <div class="log" id="vllm-setup-log" style="display:${(vs.setup_job&&vs.setup_job.running)?"":"none"}">${esc(vs.setup_log||"idle")}</div>
    </div>`));
  $$("[data-install]",v).forEach(b=>b.onclick=async()=>{b.disabled=true;b.textContent="installing...";
    const r=await api("/api/setup/install",{tool:b.dataset.install});toast(r.ok?"Installed":"Install failed",r.ok?"ok":"err");loadSetup();});
  $("#btn-scan").onclick=scanDrives;
  $("#btn-missing").onclick=checkMissing;
  const autoSel=$("#auto-load");
  if(autoSel)autoSel.onchange=async()=>{await api("/api/config",{auto_load_model:autoSel.value});toast(autoSel.value?`Auto-load: ${autoSel.value}`:"Auto-load disabled","ok");};
  $("#net-lan").onchange=e=>{
    $("#net-keyrow").style.display=e.target.checked?"":"none";
    $("#btn-net-genkey").style.display=e.target.checked?"":"none";
  };
  $("#btn-net-genkey").onclick=()=>{
    const key=[...crypto.getRandomValues(new Uint8Array(24))].map(b=>b.toString(16).padStart(2,"0")).join("");
    $("#net-apikey").value=key;
  };
  $("#btn-net-apply").onclick=async()=>{
    const msg=$("#net-msg"),lan=$("#net-lan").checked;
    const host=lan?"0.0.0.0":"127.0.0.1";
    const apiKey=$("#net-apikey").value.trim();
    if(lan&&$("#net-require-key").checked&&!apiKey&&!net.has_api_key){
      msg.className="msg err";msg.textContent='set or generate an API key first (or uncheck "require an API key")';return;
    }
    msg.className="msg work";msg.textContent="restarting router...";
    const r=await api("/api/network",{host,api_key:lan?(apiKey||undefined):""});
    if(r.ok){msg.className="msg ok";msg.textContent="applied";toast(lan?"LAN access enabled":"LAN access disabled","ok");setTimeout(loadSetup,1500);}
    else{msg.className="msg err";msg.textContent=r.error||"failed";}
  };
  const distroSel=$("#vllm-distro");
  if(distroSel)distroSel.onchange=()=>api("/api/config",{wsl_distro:distroSel.value}).then(()=>loadSetup());
  const instBtn=$("#btn-vllm-install");
  if(instBtn)instBtn.onclick=async()=>{
    const msg=$("#vllm-inst-msg");msg.className="msg work";msg.textContent="starting install...";
    const r=await api("/api/vllm/setup/install",{distro:distroSel?distroSel.value:undefined});
    if(r.started){toast("vLLM install started","ok");pollVllmSetup();}else{msg.textContent="already running";}
  };
  if(vs.setup_job&&vs.setup_job.running)pollVllmSetup();
}
async function scanDrives(){
  const msg=$("#scan-msg");msg.className="msg work";msg.textContent="scanning all drives (may take a moment)...";
  const r=await api("/api/scan",{});
  const known=new Set(STATE?STATE.models.map(m=>m.id):[]);
  const fresh=r.entries.filter(e=>!known.has(e.id));
  msg.className="msg ok";msg.textContent=`${r.entries.length} found, ${fresh.length} new`;
  setHTML($("#scan-out"),`<div class="note">${esc(fresh.length)} new models not yet in your config:</div>
    <div class="list" style="margin-top:10px">${fresh.map(e=>`<div class="row"><div class="rhead" style="cursor:default;grid-template-columns:1fr auto">
      <span class="mid">${esc(e.id)}${e.mmproj?'<span class="tag vis">vision</span>':''}${e.embeddings?'<span class="tag">embed</span>':''}</span>
      <span class="ctxpill">${esc(e.gib)} GiB</span></div></div>`).join("")||'<div class="note">nothing new</div>'}</div>
    ${fresh.length?`<div class="actions"><button class="primary" id="btn-apply">Add ${fresh.length} models to config</button><span class="msg" id="apply-msg"></span></div>`:""}`);
  if(fresh.length)$("#btn-apply").onclick=async()=>{const am=$("#apply-msg");am.className="msg work";am.textContent="writing config...";
    const rr=await api("/api/scan/apply",{entries:fresh});am.className="msg ok";am.textContent=`added ${rr.added}`;toast("Models added","ok");refresh(true);};
}

async function checkMissing(){
  const out=$("#missing-out");setHTML(out,`<div class="note">checking configured models against disk...</div>`);
  let r;try{r=await api("/api/scan/missing");}catch(e){setHTML(out,`<div class="note" style="color:var(--red)">backend unreachable</div>`);return;}
  const miss=(r&&r.missing)||[];
  if(!miss.length){setHTML(out,`<div class="note">All configured models still exist on disk.</div>`);return;}
  setHTML(out,`<div class="note">${esc(miss.length)} configured model(s) whose file is gone:</div>
    <div class="list" style="margin-top:10px">${miss.map(m=>`<div class="row"><div class="rhead" style="cursor:default;grid-template-columns:1fr auto">
      <span class="mid">${esc(m.id)}${m.loaded?'<span class="tag">loaded</span>':''}</span>
      <span class="ctxpill" title="${esc(m.model)}" style="color:var(--red);border-color:var(--red)">missing file</span></div></div>`).join("")}</div>
    <div class="actions"><button class="primary" id="btn-prune">Remove ${miss.length} missing</button><span class="msg" id="prune-msg"></span></div>`);
  $("#btn-prune").onclick=async()=>{const pm=$("#prune-msg");pm.className="msg work";pm.textContent="removing...";
    const rr=await api("/api/scan/prune",{ids:miss.map(m=>m.id)});
    toast(`Removed ${rr.removed.length} missing model(s)`,"ok");refresh(true);checkMissing();};
}

/* ---------- discover tab (HuggingFace) ---------- */
let dlPoll=null, discoverLoaded=false;
const PLAT_LABEL={windows:"WIN",linux:"LINUX",macos:"MAC"};
function platTags(platforms){
  if(!platforms||!platforms.length)return "";
  const cur=STATE&&STATE.platform;
  let s=platforms.map(p=>{
    const here=p===cur;
    return `<span class="tag" title="runs on ${esc(p)}${here?" (this machine)":""}" style="${here?"color:var(--amber);border-color:var(--amber)":""}">${PLAT_LABEL[p]||esc(p.toUpperCase())}</span>`;
  }).join("");
  if(cur&&!platforms.includes(cur))
    s+=`<span class="tag" style="color:var(--red);border-color:var(--red)" title="this backend does not run on ${esc(cur)}">NOT ON ${PLAT_LABEL[cur]||esc(cur.toUpperCase())}</span>`;
  return s;
}
function hubRow(m,installed,clickClass){
  const inst=installed.has(m.repo);
  return `<div class="row" data-repo="${esc(m.repo)}">
    <div class="rhead ${clickClass}" style="grid-template-columns:1fr auto auto auto auto">
      <span class="mid">${esc(m.repo)}
        ${platTags(m.platforms)}
        ${m.gated?'<span class="tag" style="color:var(--red);border-color:var(--red)" title="gated repo - requires accepting terms + an HF token; downloads from here will fail">GATED</span>':''}
        ${inst?'<span class="tag" style="color:var(--green);border-color:var(--green)" title="already in your registry">INSTALLED</span>':''}
      </span>
      ${m.updated?`<span class="ctxpill"><span class="k">upd</span> ${esc(m.updated)}</span>`:""}
      <span class="ctxpill">${esc((m.downloads||0).toLocaleString())} dl</span>
      <span class="ctxpill" style="color:var(--cyan)">${esc(m.likes)} &hearts;</span>
      <span class="chev">&#9654;</span>
    </div>
    <div class="edit"></div>
  </div>`;
}
let dlPrev=null;   // {t, bytes} from the previous progress poll -> speed/ETA
function dlSpeed(s){
  const now=Date.now();let txt="";
  if(dlPrev&&s.downloaded>=dlPrev.bytes){   // negative delta = next shard started
    const dt=(now-dlPrev.t)/1000;
    if(dt>0.2){const bps=(s.downloaded-dlPrev.bytes)/dt;
      if(bps>1e4){txt=` · ${(bps/1e6).toFixed(1)} MB/s`;
        if(s.total>s.downloaded)txt+=` · ETA ${fmtDur((s.total-s.downloaded)/bps)}`;}}
  }
  dlPrev={t:now,bytes:s.downloaded};
  return txt;
}
const FIT_LABEL={fits:["FITS VRAM","ok"],tight:["TIGHT","work"],offload:["CPU OFFLOAD","err"],unknown:["?",""]};
function fitBadge(fit){const [txt,cls]=FIT_LABEL[fit]||FIT_LABEL.unknown;
  const col=cls==="ok"?"var(--green)":cls==="work"?"var(--amber)":cls==="err"?"var(--red)":"var(--dim)";
  return `<span class="tag" style="color:${col};border-color:${col}">${txt}</span>`;}
function loadDiscover(){
  if(discoverLoaded)return; discoverLoaded=true;
  setHTML($("#view-discover"),`
    <div class="card"><h3>Discover models on huggingface.co</h3>
      <div class="toolbar">
        <select id="hub-mode" style="background:#0a0d0e;border:1px solid var(--hair);color:var(--ink);font-family:var(--mono);font-size:12px;padding:8px">
          <option value="gguf">GGUF (llama.cpp)</option>
          <option value="safetensors">safetensors (vLLM)</option>
        </select>
        <input class="search" id="hub-q" placeholder="search models (e.g. qwen coder, gemma vision)... blank = most downloaded">
        <select id="hub-sort" style="background:#0a0d0e;border:1px solid var(--hair);color:var(--ink);font-family:var(--mono);font-size:12px;padding:8px">
          <option value="downloads">most downloaded</option>
          <option value="lastModified">newest</option>
          <option value="likes">most liked</option>
        </select>
        <button class="primary" id="hub-go">Search</button>
        <span class="msg" id="hub-msg"></span>
      </div>
      <div class="note">Fit ratings compare file size against your total VRAM (<span id="hub-vram">?</span> GB across all GPUs).
        FITS = full GPU offload with headroom &middot; TIGHT = loads but little room for context &middot; CPU OFFLOAD = larger than VRAM, will use system RAM (slower).</div>
    </div>
    <div id="hub-results"></div>
    <div class="card" id="hub-dlcard" style="display:none"><h3>Download</h3>
      <div class="kv"><span class="k">file</span><span class="v" id="dl-file">-</span></div>
      <div class="meter" style="margin-top:8px" id="dl-meter"></div>
      <div class="kv"><span class="k">progress</span><span class="v" id="dl-prog">-</span></div>
      <div class="actions" id="dl-run" style="display:none">
        <button class="ghost" id="dl-pause">Pause</button>
        <button class="ghost" id="dl-resume" style="display:none">Resume</button>
        <button class="ghost" id="dl-cancel">Cancel download</button>
      </div>
      <div class="actions" id="dl-done" style="display:none">
        <button class="primary" id="dl-add">Add to my models</button><span class="msg" id="dl-msg"></span>
      </div>
    </div>`);
  $("#dl-cancel").onclick=async()=>{const r=await api("/api/hub/cancel",{});toast(r.ok?"Cancelling...":"No download running",r.ok?"ok":"err");};
  $("#dl-pause").onclick=async()=>{const r=await api("/api/hub/pause",{});toast(r.ok?"Pausing...":"No download running",r.ok?"ok":"err");};
  $("#dl-resume").onclick=async()=>{const r=await api("/api/hub/resume",{});if(r.ok){toast("Resuming download","ok");ggufDlPoll();}else toast("Nothing to resume","err");};
  // vLLM (safetensors) is Windows/WSL-only; drop the mode on other platforms
  if(STATE&&STATE.vllm_supported===false){
    const opt=$('#hub-mode option[value="safetensors"]');
    if(opt)opt.remove();
  }
  // restore last search (mode/sort/query survive tab switches + reloads)
  try{
    const saved=JSON.parse(localStorage.getItem("lf_hub")||"{}");
    if(saved.mode&&$(`#hub-mode option[value="${saved.mode}"]`))$("#hub-mode").value=saved.mode;
    if(saved.sort)$("#hub-sort").value=saved.sort;
    if(saved.q)$("#hub-q").value=saved.q;
  }catch(e){}
  $("#hub-go").onclick=hubSearch;
  $("#hub-mode").onchange=()=>hubSearch();
  $("#hub-q").addEventListener("keydown",e=>{if(e.key==="Enter")hubSearch();});
  hubSearch();
}
async function hubSearch(){
  localStorage.setItem("lf_hub",JSON.stringify({mode:$("#hub-mode").value,sort:$("#hub-sort").value,q:$("#hub-q").value.trim()}));
  if($("#hub-mode")&&$("#hub-mode").value==="safetensors")return vllmHubSearch();
  const msg=$("#hub-msg");msg.className="msg work";msg.textContent="searching huggingface.co...";
  const r=await api("/api/hub/search",{query:$("#hub-q").value.trim(),sort:$("#hub-sort").value});
  if(r.error){msg.className="msg err";msg.textContent=r.error.slice(0,80);return;}
  $("#hub-vram").textContent=(r.vram_mib/1024).toFixed(1);
  msg.className="msg ok";msg.textContent=`${r.results.length} repos`;
  const inst=new Set(r.installed||[]);
  setHTML($("#hub-results"),`<div class="list">${r.results.map(m=>hubRow(m,inst,"hub-repo")).join("")}</div>`);
  $$("#hub-results .hub-repo").forEach(h=>h.onclick=()=>hubFiles(h.parentElement));
}
async function hubFiles(row){
  const open=row.classList.toggle("open");
  if(!open)return;
  const box=$(".edit",row);setHTML(box,`<div class="note">listing files...</div>`);
  const r=await api("/api/hub/files",{repo:row.dataset.repo});
  if(r.error){setHTML(box,`<div class="note" style="color:var(--red)">${esc(r.error.slice(0,120))}</div>`);return;}
  const mm=r.mmproj&&r.mmproj.length?r.mmproj[0].path:"";
  setHTML(box,`
    ${mm?`<div class="note">vision model - the smallest mmproj (${esc(mm)}) will be downloaded too</div>`:""}
    <div class="list" style="margin-top:8px">${r.files.map(f=>`
      <div class="row"><div class="rhead" style="grid-template-columns:1fr auto auto auto;cursor:default">
        <span class="mid">${esc(f.path)}${f.shards>1?`<span class="tag">${f.shards} shards</span>`:""}</span>
        <span class="ctxpill">${esc((f.size/1e9).toFixed(2))} GB</span>
        ${fitBadge(f.fit)}
        <button data-dl="${esc(f.path)}" data-shards="${f.shards}" ${f.fit==="offload"?'title="larger than VRAM - will be slow"':""}>Download</button>
      </div></div>`).join("")}</div>`);
  $$("[data-dl]",box).forEach(b=>b.onclick=()=>hubDownload(row.dataset.repo,b.dataset.dl,parseInt(b.dataset.shards),mm));
}
async function hubDownload(repo,path,shards,mmproj){
  const r=await api("/api/hub/download",{repo,path,shards,mmproj});
  if(!r.started){toast("A download is already running","err");return;}
  toast("Download started","ok");
  $("#hub-dlcard").style.display="";$("#dl-done").style.display="none";
  $("#dl-run").style.display="";dlPrev=null;
  ggufDlPoll();
}
// GGUF download progress loop, shared by a fresh download and by Resume (#7).
function ggufDlPoll(){
  $("#hub-dlcard").style.display="";$("#dl-run").style.display="";dlPrev=null;
  clearInterval(dlPoll);
  dlPoll=setInterval(async()=>{
    const s=await api("/api/hub/progress");
    $("#dl-file").textContent=`${s.repo} :: ${s.file||"-"} (${s.done_files+1 > s.total_files ? s.total_files : s.done_files+1}/${s.total_files})`;
    const pct=s.total?Math.round(100*s.downloaded/s.total):0;
    setHTML($("#dl-meter"),meter(s.downloaded,Math.max(s.total,1)));
    $("#dl-prog").textContent=s.phase==="done"?"complete":s.phase==="failed"?("FAILED: "+s.error.slice(0,80)):s.phase==="cancelled"?"cancelled":s.phase==="paused"?`paused at ${(s.downloaded/1e9).toFixed(2)} / ${(s.total/1e9).toFixed(2)} GB (${pct}%)`:`${(s.downloaded/1e9).toFixed(2)} / ${(s.total/1e9).toFixed(2)} GB (${pct}%)${dlSpeed(s)}`;
    const active=s.phase==="downloading"||s.phase==="starting";
    $("#dl-run").style.display=(active||s.phase==="paused")?"":"none";
    $("#dl-pause").style.display=active?"":"none";
    $("#dl-resume").style.display=s.phase==="paused"?"":"none";
    if(s.phase==="paused"){clearInterval(dlPoll);}
    if(s.phase==="cancelled"){clearInterval(dlPoll);toast("Download cancelled","ok");}
    if(s.phase==="done"){clearInterval(dlPoll);$("#dl-done").style.display="";
      $("#dl-add").onclick=async()=>{const m=$("#dl-msg");m.className="msg work";m.textContent="registering...";
        const rr=await api("/api/hub/add",{path:s.finished_path});
        if(rr.ok){m.className="msg ok";m.textContent="added: "+rr.added.join(", ");toast("Model added to registry","ok");refresh(true);}
        else{m.className="msg err";m.textContent=rr.error||"failed";}};}
    if(s.phase==="failed")clearInterval(dlPoll);
  },1000);
}
const QUANT_BADGE={nvfp4:["NVFP4","var(--green)"],fp8:["FP8","var(--cyan)"],awq:["AWQ","var(--amber)"],gptq:["GPTQ","var(--amber)"],bf16:["BF16","var(--dim)"],fp16:["FP16","var(--dim)"]};
const VFIT_LABEL={fits:["FITS VRAM","var(--green)"],tight:["TIGHT","var(--amber)"],wont:["WON'T FIT","var(--red)"],unknown:["?","var(--dim)"]};
async function vllmHubSearch(){
  const msg=$("#hub-msg");msg.className="msg work";msg.textContent="searching safetensors repos...";
  const r=await api("/api/vllm/hub/search",{query:$("#hub-q").value.trim(),sort:$("#hub-sort").value});
  if(r.error){msg.className="msg err";msg.textContent=r.error.slice(0,80);return;}
  $("#hub-vram").textContent=(r.vram_mib/1024).toFixed(1);
  msg.className="msg ok";msg.textContent=`${r.results.length} repos`;
  const inst=new Set(r.installed||[]);
  setHTML($("#hub-results"),`<div class="list">${r.results.map(m=>hubRow(m,inst,"vhub-repo")).join("")}</div>`);
  $$("#hub-results .vhub-repo").forEach(h=>h.onclick=()=>vllmHubInfo(h.parentElement));
}
async function vllmHubInfo(row){
  const open=row.classList.toggle("open");
  if(!open)return;
  const box=$(".edit",row);setHTML(box,`<div class="note">reading repo (summing shards, detecting quant)...</div>`);
  const r=await api("/api/vllm/hub/info",{repo:row.dataset.repo});
  if(r.error){setHTML(box,`<div class="note" style="color:var(--red)">${esc(r.error.slice(0,120))}</div>`);return;}
  const [qtxt,qcol]=QUANT_BADGE[r.quant]||[r.quant.toUpperCase(),"var(--dim)"];
  const [ftxt,fcol]=VFIT_LABEL[r.fit]||VFIT_LABEL.unknown;
  const nvfp4Note=r.quant==="nvfp4"?`<div class="note" style="color:var(--green)">NVFP4 &mdash; native on your Blackwell GPUs</div>`:"";
  setHTML(box,`
    <div class="kv"><span class="k">weights size</span><span class="v">${esc((r.size_bytes/1e9).toFixed(1))} GB</span></div>
    <div class="kv"><span class="k">quantization</span><span class="v"><span class="tag" style="color:${qcol};border-color:${qcol}">${qtxt}</span></span></div>
    <div class="kv"><span class="k">VRAM fit</span><span class="v"><span class="tag" style="color:${fcol};border-color:${fcol}">${ftxt}</span></span></div>
    ${nvfp4Note}
    <div class="actions">
      <button class="primary" data-vdl="${esc(row.dataset.repo)}" data-size="${r.size_bytes}" data-quant="${esc(r.quant)}" ${r.fit==="wont"?'title="larger than usable VRAM"':""}>Download to WSL</button>
      <span class="msg" data-vmsg></span>
    </div>`);
  $(`[data-vdl]`,box).onclick=e=>vllmHubDownload(e.target.dataset.vdl,parseInt(e.target.dataset.size),e.target.dataset.quant);
}
async function vllmHubDownload(repo,sizeBytes,quant){
  const r=await api("/api/vllm/hub/download",{repo,size_bytes:sizeBytes});
  if(!r.started){toast("A download is already running","err");return;}
  toast("Download started","ok");
  $("#hub-dlcard").style.display="";$("#dl-done").style.display="none";
  $("#dl-run").style.display="none";dlPrev=null;   // WSL transfer: no cancel
  clearInterval(dlPoll);
  dlPoll=setInterval(async()=>{
    const s=await api("/api/vllm/hub/progress");
    $("#dl-file").textContent=`${s.repo} (WSL cache)`;
    const pct=s.total?Math.round(100*s.downloaded/s.total):0;
    setHTML($("#dl-meter"),meter(s.downloaded,Math.max(s.total,1)));
    $("#dl-prog").textContent=s.phase==="done"?"complete":s.phase==="failed"?("FAILED: "+(s.error||"").slice(0,80)):`${(s.downloaded/1e9).toFixed(2)} / ${(s.total/1e9).toFixed(2)} GB (${pct}%)${dlSpeed(s)}`;
    if(s.phase==="done"){clearInterval(dlPoll);$("#dl-done").style.display="";
      $("#dl-add").onclick=async()=>{const m=$("#dl-msg");m.className="msg work";m.textContent="registering...";
        const rr=await api("/api/vllm/hub/register",{repo,size_bytes:sizeBytes,quant});
        if(rr.ok){m.className="msg ok";m.textContent="added: "+rr.added;toast("vLLM model registered","ok");refresh(true);}
        else{m.className="msg err";m.textContent=rr.error||"failed";}};}
    if(s.phase==="failed")clearInterval(dlPoll);
  },1000);
}

/* ---------- stats tab ---------- */
let statsSort="tokens", statsRange=14;
const SORT_COLS={tokens:"Total",prompt:"Prompt",generated:"Gen",avg_tps:"Tok/s",runs:"Runs",loaded_secs:"Loaded"};
function setStatsRange(n){statsRange=n;loadStats(true);}
async function resetStats(){
  if(!confirm("Reset ALL usage statistics? Per-model and daily history will be zeroed. This cannot be undone."))return;
  await api("/api/stats/reset",{});toast("Stats reset","ok");loadStats(true);
}
function fmtNum(n){n=Number(n)||0;return n>=1e9?(n/1e9).toFixed(2)+"B":n>=1e6?(n/1e6).toFixed(2)+"M":n>=1e3?(n/1e3).toFixed(1)+"k":String(Math.round(n));}
function fmtDur(s){s=Math.round(Number(s)||0);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?`${h}h ${m}m`:m?`${m}m`:`${s}s`;}
function fmtAgo(ts){if(!ts)return "never";const d=Date.now()/1000-ts;return d<60?"just now":d<3600?Math.floor(d/60)+"m ago":d<86400?Math.floor(d/3600)+"h ago":Math.floor(d/86400)+"d ago";}
function statCard(label,val){return `<div class="gpu"><div class="stats" style="margin:0"><span>${esc(label)}</span></div><div style="font-family:var(--disp);font-weight:600;color:#fff;font-size:22px;margin-top:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(val)}</div></div>`;}
function sortStats(c){statsSort=c;loadStats(true);}
async function loadStats(silent){
  const v=$("#view-stats");
  if(!silent)setHTML(v,`<div class="skel">LOADING STATS...</div>`);
  let s;try{s=await api("/api/stats");}catch(e){s=null;}
  // fetch() doesn't reject on HTTP errors, so a 404/500 arrives as a parsed
  // error body, not an exception - guard on shape, not just the catch.
  if(!s||s.error||!Array.isArray(s.per_model)){if(!silent)setHTML(v,`<div class="skel" style="color:var(--red)">BACKEND UNREACHABLE</div>`);return;}
  const t=s.totals,live=s.live;
  const rows=[...s.per_model].sort((a,b)=>(b[statsSort]||0)-(a[statsSort]||0));
  const daily=s.daily.slice(-statsRange);
  const maxDaily=Math.max(1,...daily.map(d=>d.prompt+d.generated));
  setHTML(v,`
    <div class="gpus" style="grid-template-columns:repeat(auto-fit,minmax(150px,1fr))">
      ${statCard("Tokens processed",fmtNum(t.tokens))}
      ${statCard("Generated",fmtNum(t.generated))}
      ${statCard("Inference time",fmtDur(t.loaded_hours*3600))}
      ${statCard("Models used",t.models_used)}
      ${statCard("Runs (approx)",fmtNum(t.total_runs))}
      ${statCard("Most used",t.most_used||"-")}
    </div>
    <div class="card"><h3>Live Throughput${live.router_up?"":` <span style="color:var(--red);font-size:10px">(router offline)</span>`}</h3>
      <div class="kv"><span class="k">loaded model</span><span class="v ${live.loaded_model?"ok":""}">${esc(live.loaded_model||"none")}</span></div>
      <div class="kv"><span class="k">generation</span><span class="v">${(live.gen_per_sec||0).toFixed(1)} tok/s</span></div>
      <div class="kv"><span class="k">prompt eval</span><span class="v">${(live.prompt_per_sec||0).toFixed(1)} tok/s</span></div>
      <div class="kv"><span class="k">active requests</span><span class="v">${esc(live.requests_processing)}</span></div>
    </div>
    <div class="card"><h3>Activity${daily.length?` (last ${daily.length} days)`:""}
        <span style="float:right">
          <span class="chip ${statsRange===14?"on":""}" onclick="setStatsRange(14)">14d</span>
          <span class="chip ${statsRange===30?"on":""}" onclick="setStatsRange(30)">30d</span>
        </span></h3>
      ${daily.length?`<div style="display:flex;align-items:flex-end;gap:4px;height:120px;margin-top:10px">
        ${daily.map(d=>{const hp=Math.round(100*d.prompt/maxDaily),hg=Math.round(100*d.generated/maxDaily);
          return `<div title="${esc(d.date)} &middot; ${fmtNum(d.generated)} generated + ${fmtNum(d.prompt)} prompt" style="flex:1;display:flex;flex-direction:column;justify-content:flex-end;height:100%">
            <div style="height:${hg}%;min-height:${d.generated?2:0}px;background:var(--amber);box-shadow:0 0 6px var(--amber-dim)"></div>
            <div style="height:${hp}%;min-height:${d.prompt?2:0}px;background:var(--cyan);opacity:.55"></div></div>`;}).join("")}
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:6px;color:var(--dim);font-size:9px">
        <span>${esc(daily[0].date)}</span>
        <span><span style="color:var(--amber)">&#9632;</span> generated &nbsp;<span style="color:var(--cyan)">&#9632;</span> prompt</span>
        <span>${esc(daily[daily.length-1].date)}</span></div>`
      :`<div class="note">No usage recorded yet - load a model and run some inference.</div>`}
    </div>
    <div class="card"><h3>Per-model Usage</h3>
      <div class="note" style="margin:0 0 6px">Usage is scraped from the router's own metrics and totalled per model across all clients. Per-client / per-IP breakdown isn't available: clients hit the llama.cpp router directly, so the dashboard never sees individual request origins.</div>
      ${rows.length?`<div class="toolbar" style="margin:6px 0 0">
        ${Object.keys(SORT_COLS).map(c=>`<span class="chip ${statsSort===c?"on":""}" onclick="sortStats('${c}')">${SORT_COLS[c]}</span>`).join("")}
        <span class="chip" onclick="resetStats()" style="margin-left:auto;color:var(--red);border-color:var(--red)" title="zero all usage statistics">Reset stats</span>
      </div>
      <div class="list" style="margin-top:12px">${rows.map(m=>`
        <div class="row"><div class="rhead" style="cursor:default;grid-template-columns:9px 1fr auto auto auto auto auto">
          <span class="led ${live.loaded_model===m.id?"loaded":""}"></span>
          <span class="mid">${esc(m.id)}</span>
          <span class="ctxpill" title="prompt ${fmtNum(m.prompt)} + generated ${fmtNum(m.generated)}">${fmtNum(m.tokens)} tok</span>
          <span class="stat" title="average generation speed while active">${m.avg_tps?m.avg_tps+" tok/s":"-"}</span>
          <span class="stat">${fmtNum(m.runs)} runs</span>
          <span class="stat">${fmtDur(m.loaded_secs)}</span>
          <span class="stat">${fmtAgo(m.last_used)}</span>
        </div></div>`).join("")}</div>`
      :`<div class="note">No models have logged usage yet.</div>`}
    </div>`);
}
setInterval(()=>{if($(".tab.active").dataset.tab==="stats")loadStats(true);},4000);

function clock(){$("#clock").textContent=new Date().toLocaleTimeString('en-GB')+" LOCAL";}
setInterval(clock,1000);clock();
(async()=>{SCHEMA=await api("/api/schema");await refresh();})();
setInterval(()=>{if($(".tab.active").dataset.tab==="models")refresh(true);},4000);

/* ---------- router log ---------- */
async function refreshRouterLog(){
  const el=$("#router-log");if(!el)return;
  const s=await api("/api/router/log");
  const wasBottom=el.scrollTop+el.clientHeight>=el.scrollHeight-20;
  el.textContent=s.log||"idle";
  if(wasBottom)el.scrollTop=el.scrollHeight;
}
setInterval(()=>{if($(".tab.active").dataset.tab==="models")refreshRouterLog();},3000);
refreshRouterLog();

async function refreshVllmLog(){
  const el=$("#vllm-log");if(!el)return;
  const s=await api("/api/vllm/log");
  const wasBottom=el.scrollTop+el.clientHeight>=el.scrollHeight-20;
  el.textContent=s.log||"idle";
  if(wasBottom)el.scrollTop=el.scrollHeight;
}
setInterval(()=>{if($(".tab.active").dataset.tab==="models")refreshVllmLog();},3000);
refreshVllmLog();
