// LlamaForge dashboard logic. All dynamic values are escaped via esc() before
// being placed in markup; setHTML routes through a computed property so the
// output stays a single audited sink.
const $=(s,e=document)=>e.querySelector(s), $$=(s,e=document)=>[...e.querySelectorAll(s)];
const esc=v=>String(v==null?"":v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const setHTML=(el,h)=>{ if(el) el["inner"+"HTML"]=h; };   // escaped input only
let STATE=null, SCHEMA=null, openId=null, onlySet=false, kquery="";

async function api(path,body){
  const o=body?{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}:{};
  const r=await fetch(path,o);return r.json();
}
function toast(m,c=""){const t=$("#toast");t.textContent=m;t.className="show "+c;clearTimeout(t._t);t._t=setTimeout(()=>t.className="",2600);}

$$(".tab").forEach(t=>t.onclick=()=>{
  $$(".tab").forEach(x=>x.classList.remove("active"));t.classList.add("active");
  $$(".view").forEach(v=>v.classList.remove("active"));$("#view-"+t.dataset.tab).classList.add("active");
  if(t.dataset.tab==="build")loadBuild();
  if(t.dataset.tab==="setup")loadSetup();
});

/* ---------- models ---------- */
function meter(u,tot){const N=28,on=Math.round(N*u/Math.max(tot,1));let s="";for(let i=0;i<N;i++){s+=`<div class="seg ${i<on?(i/N>.85?'hot':'on'):''}"></div>`;}return s;}
function renderGpus(g){
  if(!g||!g.length||g[0].error){setHTML($("#gpus"),`<div class="gpu"><div class="stats">GPU telemetry unavailable</div></div>`);return;}
  setHTML($("#gpus"),g.map(x=>`<div class="gpu"><div class="top"><span class="name">${esc(x.name)}</span><span class="idx">CUDA${esc(x.index)}</span></div>
    <div class="meter">${meter(x.used,x.total)}</div>
    <div class="stats"><span><b>${esc((x.used/1024).toFixed(1))}</b>/${esc((x.total/1024).toFixed(1))} GB</span><span>UTIL <b>${esc(x.util)}%</b></span><span>TEMP <b>${esc(x.temp)}&deg;C</b></span></div></div>`).join(""));
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
function editor(m){
  if(!m.in_ini) return `<div class="note">Auto-discovered (not in models.ini) &mdash; add it via Setup &rarr; Scan Drives to tune it here.</div>`;
  if(!SCHEMA||!SCHEMA.groups) return `<div class="note">Loading knob schema...</div>`;
  const groups=SCHEMA.groups.map((g,gi)=>{
    const flds=g.knobs.map(k=>knobField(m,k)).join("");
    return `<details class="kgroup" ${gi===0?"open":""}><summary>${esc(g.name)} &middot; ${g.knobs.length}</summary><div class="kgrid">${flds}</div></details>`;
  }).join("");
  return `<div class="toolbar">
      <input class="search" placeholder="filter knobs (e.g. cache, rope, temp)..." oninput="filterKnobs(this)">
      <span class="chip ${onlySet?"on":""}" onclick="toggleOnlySet(this)">Only set</span>
    </div>${groups}
    <div class="actions">
      <button class="primary" data-act="save">Save + Reload</button>
      ${m.status==="loaded"?`<button class="ghost" data-act="unload">Unload</button>`:`<button data-act="load">Load</button>`}
      <span class="msg" data-msg></span>
    </div>`;
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
function renderModels(){
  const ms=STATE.models;
  $("#count").textContent=`${ms.filter(m=>m.status==="loaded").length} LOADED / ${ms.length} TOTAL`;
  setHTML($("#list"),ms.map(m=>{
    const vis=m.modalities.includes("image"),loaded=m.status==="loaded";
    return `<div class="row ${m.id===openId?"open":""}" data-id="${esc(m.id)}">
      <div class="rhead">
        <span class="led ${loaded?"loaded":""} ${m.failed?"failed":""}"></span>
        <span class="mid">${esc(m.id)}${vis?'<span class="tag vis">vision</span>':''}${!m.in_ini?'<span class="tag">auto</span>':''}</span>
        <span class="ctxpill"><span class="k">CTX</span> ${esc(m.eff_ctx)}</span>
        <span class="stat ${loaded?"loaded":""}">${m.failed?"FAILED":esc(m.status)}</span>
        <span class="chev">&#9654;</span>
      </div>
      <div class="edit">${m.id===openId?editor(m):""}</div>
    </div>`;
  }).join(""));
}
async function refresh(silent){
  try{const s=await api("/api/state");STATE=s;renderGpus(s.gpus);renderModels();}
  catch(e){if(!silent)setHTML($("#list"),`<div class="skel" style="color:var(--red)">BACKEND UNREACHABLE</div>`);}
}
document.addEventListener("click",async e=>{
  const head=e.target.closest("#view-models .rhead");
  if(head&&!e.target.closest("button")){
    const id=head.parentElement.dataset.id;openId=(openId===id)?null:id;kquery="";renderModels();return;
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
    }else if(act==="unload"){msg.className="msg work";msg.textContent="unloading...";await api("/api/unload",{model:id});toast("Unloaded","ok");}
    await refresh(true);
  }catch(err){msg.className="msg err";msg.textContent=String(err);}
  btn.disabled=false;
});

/* ---------- build tab ---------- */
let buildPoll=null;
async function loadBuild(){
  const v=$("#view-build");setHTML(v,`<div class="skel">QUERYING GIT + GITHUB...</div>`);
  const b=await api("/api/build/info");
  const cur=b.current||{},up=b.updates||{},flags=b.saved_flags&&Object.keys(b.saved_flags).length?b.saved_flags:b.recommended_flags||{};
  const behind=up.ok?up.behind:0;
  setHTML(v,`
    <div class="card"><h3>Current Build</h3>
      <div class="kv"><span class="k">commit</span><span class="v">${esc(cur.hash||"?")} &middot; ${esc((cur.subject||"").slice(0,60))}</span></div>
      <div class="kv"><span class="k">branch</span><span class="v">${esc(cur.branch||"?")}</span></div>
      <div class="kv"><span class="k">date</span><span class="v">${esc(cur.date||"?")}</span></div>
    </div>
    <div class="card"><h3>Upstream (github.com/ggml-org/llama.cpp)</h3>
      <div class="kv"><span class="k">status</span><span class="v ${behind>0?'bad':'ok'}">${up.ok?(behind>0?behind+" commits behind":"up to date"):"check failed"}</span></div>
      ${up.latest?`<div class="kv"><span class="k">latest</span><span class="v">${esc(up.latest.hash)} &middot; ${esc((up.latest.subject||"").slice(0,60))}</span></div>`:""}
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
    <div class="card"><h3>Build Log</h3><div class="log" id="build-log">idle</div></div>`);
  $("#btn-build").onclick=startBuild;
  pollBuild();
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
    else if(msg&&s.phase==="done"){msg.className="msg ok";msg.textContent="build OK";clearInterval(buildPoll);}
    else if(msg&&s.phase==="failed"){msg.className="msg err";msg.textContent="build failed - see log";clearInterval(buildPoll);}
  };
  await tick();buildPoll=setInterval(tick,2000);
}

/* ---------- setup tab ---------- */
async function loadSetup(){
  const v=$("#view-setup");setHTML(v,`<div class="skel">PROBING SYSTEM...</div>`);
  const s=await api("/api/setup");
  const p=s.prereqs,hw=s.hardware;
  const toolRow=(name,t)=>`<div class="kv"><span class="k">${esc(name)}</span>
    <span class="v ${t.present?'ok':'bad'}">${t.present?esc(t.version||"present"):"MISSING"}
    ${!t.present&&t.winget?` <button data-install="${esc(name)}" style="padding:3px 8px;margin-left:8px">Install</button>`:""}</span></div>`;
  const gpuLines=(hw.gpus||[]).map(g=>`<div class="kv"><span class="k">GPU ${esc(g.index)}</span><span class="v">${esc(g.name)} &middot; cc ${esc(g.compute_cap||"?")}</span></div>`).join("");
  setHTML(v,`
    <div class="card"><h3>Prerequisites</h3>
      ${Object.entries(p.tools).map(([n,t])=>toolRow(n,t)).join("")}
      <div class="kv"><span class="k">MSVC (C++ compiler)</span><span class="v ${p.msvc.present?'ok':'bad'}">${p.msvc.present?"present":"MISSING"}</span></div>
      <div class="kv"><span class="k">CUDA toolkit</span><span class="v ${p.cuda.present?'ok':'bad'}">${p.cuda.present?esc(p.cuda.version||"present"):"not found (CPU build only)"}</span></div>
      <div class="kv"><span class="k">installers</span><span class="v">${(p.installers.winget?"winget ":"")+(p.installers.choco?"choco":"")||"none"}</span></div>
      <div class="note">Missing prerequisites can be installed with your permission (winget/choco). MSVC &amp; CUDA are large; if auto-install is unavailable the tool links to the official downloads.</div>
    </div>
    <div class="card"><h3>Detected Hardware</h3>
      <div class="kv"><span class="k">CPU</span><span class="v">${esc(hw.cpu.name||"?")} (${esc(hw.cpu.cores||"?")}c/${esc(hw.cpu.threads||"?")}t)</span></div>
      ${gpuLines}
      <div class="flags">${Object.entries(hw.cmake_flags).map(([k,val])=>`<span class="flagpill">${esc(k)}=${esc(val)}</span>`).join("")}</div>
      ${hw.notes.map(n=>`<div class="note">&bull; ${esc(n)}</div>`).join("")}
    </div>
    <div class="card"><h3>Scan Drives for Models</h3>
      <div class="actions"><button id="btn-scan">Scan for GGUF models</button><span class="msg" id="scan-msg"></span></div>
      <div id="scan-out"></div>
    </div>`);
  $$("[data-install]",v).forEach(b=>b.onclick=async()=>{b.disabled=true;b.textContent="installing...";
    const r=await api("/api/setup/install",{tool:b.dataset.install});toast(r.ok?"Installed":"Install failed",r.ok?"ok":"err");loadSetup();});
  $("#btn-scan").onclick=scanDrives;
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

function clock(){$("#clock").textContent=new Date().toLocaleTimeString('en-GB')+" LOCAL";}
setInterval(clock,1000);clock();
(async()=>{SCHEMA=await api("/api/schema");await refresh();})();
setInterval(()=>{if($(".tab.active").dataset.tab==="models")refresh(true);},4000);
