(() => {
  "use strict";

  const DIRECT_REPORT = "../data/trajectory_dialogues_v2/corpus_report.json";
  const COMPOSITION_REPORT = "../data/t10_compositions_v2/corpus_report.json";
  const opColors = { G: "#5bd9c3", F: "#7d9cff", B: "#cdf36c", M: "#ffd166", R: "#ffad98", P: "#c9a6ff", V: "#ffffff" };
  const classLabels = { all: "ALL", T3: "T3", T4: "T4", T7: "T7", T8: "T8", T10: "T10", T10C: "T1→T10" };
  const scopeText = {
    trajectory_reasoning: "受控轨迹推理。RGB 按真实轨迹逐轮释放，后续问题可以读取此前建立的空间状态。",
    target_view_read_not_prediction: "目标视角关系读取。模型已看到 T10 渲染图，因此这里只训练 frame transform 与 relation read，不把它称为新视角预测。",
    source_to_heldout_perspective_prediction: "严格视角预测。先用 11 张 T1 漫游图建立状态；预测轮不提供新图，下一轮才释放 T10 held-out render 做验证。",
  };
  const state = { entries: [], artifact: null, activeId: null, className: "all", search: "" };
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
  }

  function webPath(value) {
    const path = String(value || "");
    const marker = "/episode3D/";
    const index = path.indexOf(marker);
    if (index >= 0) return `/episode3D/${path.slice(index + marker.length)}`;
    return path;
  }

  function filteredEntries() {
    const needle = state.search.trim().toLowerCase();
    return state.entries.filter((entry) => {
      const classMatch = state.className === "all" || entry.trajectory_class === state.className;
      const text = `${entry.acquisition_id} ${entry.scene || ""} ${entry.trajectory_class}`.toLowerCase();
      return classMatch && (!needle || text.includes(needle));
    });
  }

  function updateRoute(id) {
    const url = new URL(location.href);
    url.searchParams.set("id", id);
    history.replaceState(null, "", url);
  }

  function renderStats(direct, composition) {
    const episodes = (direct.compiled || []).length + (composition.compiled || []).length;
    const rounds = Object.values(direct.round_counts || {}).reduce((sum, value) => sum + Number(value), 0) + (composition.compiled || []).length * 4;
    const viewCounts = { T3: 6, T4: 8, T7: 3, T8: 6, T10: 3 };
    const rgb = Object.entries(direct.compiled_counts || {}).reduce((sum, [key, value]) => sum + (viewCounts[key] || 0) * Number(value), 0) + (composition.compiled || []).length * 12;
    $("#corpus-stats").innerHTML = [[episodes, "EPISODES"], [rounds, "ROUNDS"], [rgb, "RGB"]].map(([value, label]) => `<span><b>${escapeHtml(value)}</b><small>${label}</small></span>`).join("");
  }

  function renderFilters() {
    const counts = { all: state.entries.length };
    state.entries.forEach((entry) => { counts[entry.trajectory_class] = (counts[entry.trajectory_class] || 0) + 1; });
    $("#class-filters").innerHTML = Object.entries(classLabels).map(([key, label]) => `<button type="button" class="${key === state.className ? "active" : ""}" data-class="${escapeHtml(key)}" title="${escapeHtml(counts[key] || 0)} episodes">${escapeHtml(label)}</button>`).join("");
    $$('[data-class]').forEach((button) => button.addEventListener("click", () => {
      state.className = button.dataset.class;
      renderFilters();
      renderList();
    }));
  }

  function renderList() {
    const entries = filteredEntries();
    $("#filter-count").textContent = `${entries.length} / ${state.entries.length} EPISODES`;
    if (!entries.length) {
      $("#episode-list").innerHTML = '<p class="empty">没有符合当前筛选条件的 episode。</p>';
      return;
    }
    $("#episode-list").innerHTML = entries.map((entry) => `<button type="button" class="${entry.acquisition_id === state.activeId ? "active" : ""}" data-id="${escapeHtml(entry.acquisition_id)}">
      <span><b>${escapeHtml(entry.scene || entry.acquisition_id)}</b><em>${escapeHtml(classLabels[entry.trajectory_class] || entry.trajectory_class)}</em></span>
      <code>${escapeHtml(entry.acquisition_id)}</code>
      <small>${escapeHtml(entry.round_count || 4)} ROUNDS · ${entry.trajectory_class === "T10C" ? "PREDICT → VERIFY" : "PASSED"}</small>
    </button>`).join("");
    $$('[data-id]').forEach((button) => button.addEventListener("click", () => loadArtifact(button.dataset.id)));
  }

  function imagePath(artifact, viewId) {
    return webPath(artifact.rgb_paths?.[viewId] || "");
  }

  function renderArtifact(artifact) {
    state.artifact = artifact;
    state.activeId = artifact.acquisition_id;
    updateRoute(artifact.acquisition_id);
    renderList();
    $("#episode-status").textContent = artifact.task_scope === "source_to_heldout_perspective_prediction" ? "PREDICT BEFORE RENDER" : "ALL COMPILER GATES PASS";
    $("#episode-title").textContent = (artifact.catalog_scene || artifact.acquisition_id).replaceAll("_", " ");
    $("#episode-id").textContent = artifact.acquisition_id;
    $("#episode-meta").innerHTML = [[artifact.trajectory_class, "CLASS"], [artifact.rounds.length, "ROUNDS"], [artifact.view_ids.length, "RGB"], [artifact.task_scope === "target_view_read_not_prediction" ? "READ" : artifact.task_scope === "source_to_heldout_perspective_prediction" ? "PREDICT" : "EPISODIC", "SCOPE"]].map(([value, label]) => `<span><b>${escapeHtml(value)}</b><small>${label}</small></span>`).join("");
    $("#scope-card").innerHTML = `<b>TASK SCOPE</b><p>${escapeHtml(scopeText[artifact.task_scope] || artifact.task_scope)}</p>`;

    let released = 0;
    $("#round-list").innerHTML = artifact.rounds.map((round) => {
      const views = round.new_view_ids || [];
      released += views.length;
      const images = views.length ? views.map((viewId) => {
        const src = imagePath(artifact, viewId);
        return `<figure><button class="image-button" type="button" data-src="${escapeHtml(src)}" data-caption="${escapeHtml(artifact.acquisition_id)} · ${escapeHtml(viewId)}"><img loading="lazy" src="${escapeHtml(src)}" alt="${escapeHtml(viewId)} RGB"></button><figcaption>${escapeHtml(viewId)}</figcaption></figure>`;
      }).join("") : '<div class="state-only"><b>0 NEW RGB</b><span>本轮只读取此前对话中的共享空间状态</span></div>';
      const abilities = [round.capability?.primary, ...(round.capability?.supporting || [])].filter(Boolean);
      const sentences = (round.answer_sentences || []).map((sentence) => `<span class="${escapeHtml(sentence.role)}"><b>${escapeHtml(sentence.role)}</b>${escapeHtml(sentence.text)}</span>`).join("");
      const dag = (round.program?.nodes || []).map((node) => `<span style="--op:${escapeHtml(opColors[node.operation] || "#fff")}"><b>${escapeHtml(node.operation)}</b><small>${escapeHtml(node.node_id)}</small><code>${escapeHtml(node.output_type)}</code></span>`).join("<i>→</i>");
      const certificate = { answer_key: round.answer_key, diagnostics: round.diagnostics, new_view_ids: round.new_view_ids, evidence_view_ids: round.evidence_view_ids, claims: round.claim_sheet?.claims };
      return `<article class="round-card">
        <header><div class="round-number"><b>${String(round.round_index).padStart(2, "0")}</b><small>ROUND</small></div><div><h2>${escapeHtml(round.turn_id)}</h2><code>${escapeHtml(round.program?.semantic_signature)}</code></div><div class="ability-tags">${abilities.map((ability) => `<span>${escapeHtml(ability)}</span>`).join("")}</div></header>
        <div class="release-line"><b>本轮新增 ${views.length} 图</b><span>累计 ${released}/${artifact.view_ids.length} · evidence ${(round.evidence_view_ids || []).length}</span></div>
        <div class="new-images">${images}</div>
        <div class="qa-pair"><section><small>USER · CURRENT PREFIX ONLY</small><p>${escapeHtml(round.question_zh)}</p></section><section><small>ASSISTANT · SFT TARGET</small><p>${escapeHtml(round.answer_zh)}</p></section></div>
        <div class="reasoning-strip">${sentences}</div>
        <div class="hidden-review"><details><summary>展开 DAG、claim 与几何 certificate</summary><div class="dag">${dag}</div><pre class="certificate">${escapeHtml(JSON.stringify(certificate, null, 2))}</pre></details></div>
      </article>`;
    }).join("");
    $$(".image-button").forEach((button) => button.addEventListener("click", () => {
      $("#dialog-image").src = button.dataset.src;
      $("#dialog-caption").textContent = button.dataset.caption;
      $("#image-dialog").showModal();
    }));
    scrollTo({ top: 0, behavior: "smooth" });
  }

  async function loadArtifact(id) {
    const entry = state.entries.find((item) => item.acquisition_id === id);
    if (!entry) return;
    $("#episode-status").textContent = "LOADING JSON";
    const response = await fetch(webPath(entry.dialogue_path));
    if (!response.ok) throw new Error(`dialogue fetch failed: ${response.status}`);
    renderArtifact(await response.json());
  }

  async function init() {
    try {
      const [directResponse, compositionResponse] = await Promise.all([fetch(DIRECT_REPORT), fetch(COMPOSITION_REPORT)]);
      if (!directResponse.ok) throw new Error(`direct corpus report failed: ${directResponse.status}`);
      const direct = await directResponse.json();
      const composition = compositionResponse.ok ? await compositionResponse.json() : { compiled: [] };
      state.entries = [
        ...(direct.compiled || []),
        ...(composition.compiled || []).map((entry) => ({ ...entry, trajectory_class: "T10C", scene: entry.acquisition_id.replace(/_t1_to_t10_seed\d+$/, ""), task_scope: "source_to_heldout_perspective_prediction", round_count: 4 })),
      ];
      renderStats(direct, composition);
      renderFilters();
      renderList();
      const requested = new URL(location.href).searchParams.get("id");
      const initial = state.entries.find((entry) => entry.acquisition_id === requested)
        || state.entries.find((entry) => entry.acquisition_id === "Merom_1_int_t8_seed17")
        || state.entries[0];
      if (initial) await loadArtifact(initial.acquisition_id);
      $("#episode-search").addEventListener("input", (event) => { state.search = event.target.value; renderList(); });
      $("#close-dialog").addEventListener("click", () => $("#image-dialog").close());
    } catch (error) {
      $("#episode-status").textContent = "LOAD FAILED";
      $("#episode-title").textContent = error.message || String(error);
      $("#round-list").innerHTML = `<p class="empty">请确认 HTTP server 能读取 data/trajectory_dialogues_v2。</p>`;
    }
  }

  init();
})();
