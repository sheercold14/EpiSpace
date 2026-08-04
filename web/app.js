(() => {
  "use strict";

  const state = { data: null, trajectoryData: null, releaseData: null, releaseDetail: null, releaseLoadingId: null, releaseEpisodeId: null, releaseViewIndex: 0, releaseQuestionIndex: 0, releaseSplit: "all", releaseClass: "all", releaseSearch: "", qaPilotData: null, qaEpisodeIndex: 0, rsintDialogueData: null, compiledDialogueReport: null, compositionDialogueReport: null, compiledDialogueEntries: [], compiledDialogueArtifact: null, compiledDialogueId: null, compiledDialogueClass: "all", compiledDialogueSearch: "", sceneDialogueReport: null, sceneDialogueScene: null, sceneDialogueDubbed: null, sceneDialogueSceneId: null, sceneDialogueMode: "compare", sceneDialogueSearch: "", sceneDialogueSplit: "all", sceneDialogueCoverage: "all", trajectoryIndex: 0, trajectoryOverride: null, trajectoryLoadingId: null, trajectoryViewIndex: 0, trajectoryChannel: "rgb", collectionClass: "T1", collectionStatus: "all", viewIndex: 0, queryIndex: 5, pilotIndex: 0, channel: "rgb", filter: "all", variantIndex: 0 };
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const relationLabels = { left_of: "左侧", right_of: "右侧", in_front_of: "前方", behind: "后方", above: "上方", below: "下方" };
  const opColors = { G: "#55d6be", F: "#7799ff", B: "#c9f36a", M: "#ffd166", R: "#ffad98", P: "#c9a6ff", V: "#ffffff" };
  const channelLabels = { rgb: "RGB · MODEL VISIBLE", depth: "DEPTH · SUPERVISION", instance: "INSTANCE · SUPERVISION", semantic: "SEMANTIC · DERIVED" };

  function escapeHtml(value) {
    return String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
  }

  function text(selector, value) {
    const element = $(selector);
    if (element) element.textContent = String(value);
  }

  function currentView() { return state.data.views[state.viewIndex]; }
  function currentQuery() { return state.data.queries[state.queryIndex]; }
  function currentPilot() { return state.data.pilot_study.examples[state.pilotIndex]; }
  function currentTrajectory() { return state.trajectoryOverride || state.trajectoryData.trajectories[state.trajectoryIndex]; }
  function currentTrajectoryView() { return currentTrajectory().views[state.trajectoryViewIndex]; }

  function formatBytes(value) {
    const units = ["B", "KB", "MB", "GB"];
    let amount = Number(value);
    let unit = 0;
    while (amount >= 1024 && unit < units.length - 1) { amount /= 1024; unit += 1; }
    return `${amount >= 10 || unit === 0 ? amount.toFixed(0) : amount.toFixed(1)} ${units[unit]}`;
  }

  function shortHash(value) {
    const string = String(value || "");
    return string.length > 20 ? `${string.slice(0, 12)}…${string.slice(-8)}` : string;
  }

  function firstDefined(...values) {
    return values.find((value) => value !== undefined && value !== null && value !== "");
  }

  function firstArray(...values) {
    return values.find((value) => Array.isArray(value) && value.length) || values.find(Array.isArray) || [];
  }

  function qaPilotEpisodes() {
    const pilot = state.qaPilotData || {};
    return firstArray(pilot.episodes, pilot.batches, pilot.pilot?.episodes, pilot.data?.episodes);
  }

  function qaEpisodeQuestions(episode) {
    return firstArray(episode?.qa_records, episode?.questions, episode?.qa_pairs, episode?.items, episode?.examples, episode?.plan?.questions, episode?.generated_qa);
  }

  function qaEpisodeViews(episode) {
    return firstArray(episode?.views, episode?.observations, episode?.trajectory?.views, episode?.exposure?.views, episode?.model_input?.views);
  }

  function qaMediaUrl(value) {
    const path = String(value || "");
    if (!path) return "";
    if (/^(https?:|data:|blob:|\/\/)/.test(path)) return path;
    const projectMarker = "/episode3D/";
    const projectIndex = path.indexOf(projectMarker);
    if (projectIndex >= 0) return `/episode3D/${path.slice(projectIndex + projectMarker.length)}`;
    if (path.startsWith("web/")) return path.slice(4);
    if (/^data\/(epispace|collection|trajectory|render|raw|processed)/.test(path)) return `../${path}`;
    return path;
  }

  function qaViewId(view, index) {
    return firstDefined(view?.view_id, view?.id, view?.frame_id, view?.name, typeof view === "string" && !/[/.]/.test(view) ? view : null, `view-${String(index).padStart(3, "0")}`);
  }

  function qaViewImage(view) {
    if (typeof view === "string" && /[/.]/.test(view)) return qaMediaUrl(view);
    return qaMediaUrl(firstDefined(view?.rgb, view?.rgb_url, view?.image, view?.image_url, view?.rgb_path, view?.file_path, view?.media?.rgb, view?.paths?.rgb, view?.observation?.rgb));
  }

  function qaQuestionText(item, kind) {
    if (kind === "old") return firstDefined(item?.old_question_zh, item?.old_question, item?.original_question, item?.source_question, item?.question_before, item?.source?.question_zh, item?.source?.question, item?.original?.question_zh, item?.original?.question, "未提供原始问法");
    return firstDefined(item?.new_question_zh, item?.natural_question_zh, item?.generated_question, item?.rewritten_question, item?.question_after, item?.generated?.question_zh, item?.output?.question, item?.question, "未生成自然语言问题");
  }

  function qaAnswerText(item) {
    const answer = firstDefined(item?.surface_answer_zh, item?.answer_surface_zh, item?.new_answer_zh, item?.generated_answer, item?.generated?.answer?.surface_answer_zh, item?.response, item?.output?.answer, item?.answer);
    return typeof answer === "object" ? JSON.stringify(answer, null, 2) : firstDefined(answer, "未生成监督答案");
  }

  function qaCapabilities(item) {
    const raw = firstArray(item?.ability_tags, item?.capabilities, item?.abilities, item?.blueprint?.capability_tags, item?.sense_nova_abilities);
    if (raw.length) return raw.map((value) => typeof value === "object" ? firstDefined(value.id, value.name, value.label, JSON.stringify(value)) : value);
    if (item?.capability && typeof item.capability === "object") return [item.capability.primary, ...firstArray(item.capability.supporting)].filter(Boolean);
    const fallback = firstDefined(item?.ability, item?.capability, item?.sense_nova_ability, item?.task_type);
    return fallback ? [fallback] : [];
  }

  function qaDisposition(item) {
    return String(firstDefined(item?.disposition, item?.train_disposition, item?.split_role, item?.curriculum_role, item?.export?.disposition, "unspecified"));
  }

  function qaOptimizerEligible(value) {
    return firstDefined(value?.optimizer_eligible, value?.optimizer_decision?.optimizer_eligible, value?.training?.optimizer_eligible, value?.export?.optimizer_eligible, value?.optimizer?.eligible);
  }

  function qaValidationState(item) {
    const report = firstDefined(item?.validation, item?.validation_report, item?.validators, item?.verification, {});
    const explicit = firstDefined(report?.passed, report?.valid, report?.ok, item?.validation_passed);
    const nestedErrors = [report?.question, report?.answer, report?.critic].flatMap((value) => firstArray(value?.errors));
    const errors = [...firstArray(report?.errors, report?.failures, report?.issues, item?.validation_errors), ...nestedErrors];
    return { report, passed: explicit === undefined ? errors.length === 0 : Boolean(explicit), errors };
  }

  function compactObject(value) {
    if (!value || typeof value !== "object") return value;
    return Object.fromEntries(Object.entries(value).filter(([, item]) => item !== undefined && item !== null && item !== ""));
  }

  function renderQaPilotUnavailable(error) {
    const message = String(error?.message || error || "artifact unavailable");
    state.qaPilotData = null;
    $("#qa-pilot-authority")?.classList.add("unavailable");
    text("#qa-pilot-status", "WAITING FOR ARTIFACT");
    text("#qa-pilot-title", "QA 生成管线尚未产出可浏览文件");
    text("#qa-pilot-boundary", "这是非阻塞的开发视图，不影响页面其他数据资产。");
    $("#qa-pilot-provenance").innerHTML = `<small>LOAD RESULT</small><code>${escapeHtml(message)}</code>`;
    $("#qa-pilot-metrics").innerHTML = "";
    $("#qa-pilot-browser").hidden = true;
    $("#qa-pilot-empty").hidden = false;
  }

  function renderRsintDialogueUnavailable(error) {
    const root = $("#rsint-flagship");
    if (!root) return;
    root.classList.add("unavailable");
    text("#rsint-flagship-title", "Rs_int 增量对话 artifact 尚未就绪");
    text("#rsint-flagship-subtitle", String(error?.message || error || "artifact unavailable"));
    $("#rsint-flagship-score").innerHTML = "<b>0/7</b><small>VERIFIED ROUNDS</small>";
    $("#rsint-flagship-metrics").innerHTML = "";
    $("#rsint-round-timeline").innerHTML = '<p class="rsint-empty">运行 <code>scripts/build_rsint_llm_episode.py</code> 后，这里会按真实释放顺序展示全部7轮。</p>';
    $("#rsint-export-links").innerHTML = "";
  }

  function renderRsintDialogue() {
    const payload = state.rsintDialogueData;
    const artifact = payload?.artifact;
    if (!artifact || !Array.isArray(artifact.rounds) || !Array.isArray(artifact.observations)) throw new Error("invalid Rs_int dialogue payload");
    const root = $("#rsint-flagship");
    root.classList.remove("unavailable");
    const observations = Object.fromEntries(artifact.observations.map((item) => [item.view_id, item]));
    const passed = artifact.rounds.filter((turn) => turn.validation?.passed).length;
    text("#rsint-flagship-title", payload.title || artifact.episode_id);
    text("#rsint-flagship-subtitle", payload.subtitle || "严格增量多轮 episode");
    $("#rsint-flagship-score").innerHTML = `<b>${passed}/${artifact.rounds.length}</b><small>VERIFIED ROUNDS</small>`;
    const calls = artifact.generation_summary || {};
    const metrics = [
      [artifact.observations.length, "ordered RGB"],
      [artifact.rounds.length, "dialogue rounds"],
      [artifact.capability_coverage?.covered?.length || 0, "SenseNova abilities"],
      [(calls.question_editor_calls || 0) + (calls.answer_narrator_calls || 0) + (calls.critic_calls || 0), "subagent calls"],
      [calls.cache_hits || 0, "replay cache hits"],
    ];
    $("#rsint-flagship-metrics").innerHTML = metrics.map(([value, label]) => `<div><b>${escapeHtml(value)}</b><span>${escapeHtml(label)}</span></div>`).join("");

    $("#rsint-round-timeline").innerHTML = artifact.rounds.map((turn) => {
      const generated = turn.generated || {};
      const question = generated.question?.question_zh || "问题未生成";
      const answer = generated.answer?.surface_answer_zh || "回答未生成";
      const capabilities = [turn.capability?.primary, ...(turn.capability?.supporting || [])].filter(Boolean);
      const newViews = (turn.new_view_ids || []).map((viewId) => observations[viewId]).filter(Boolean);
      const images = newViews.length ? newViews.map((view) => `<figure><button class="rsint-image-button" data-image="${escapeHtml(qaMediaUrl(view.rgb))}" data-caption="图${escapeHtml(view.display_index)} · ${escapeHtml(view.view_id)}"><img src="${escapeHtml(qaMediaUrl(view.rgb))}" alt="${escapeHtml(view.view_id)} RGB"></button><figcaption><b>图${escapeHtml(view.display_index)}</b><span>${escapeHtml(view.view_id)}</span></figcaption></figure>`).join("") : '<div class="rsint-state-only"><b>0 NEW RGB</b><span>读取前6轮共享状态</span></div>';
      const claimRows = (turn.claim_sheet?.claims || []).map((claim) => `<li><code>${escapeHtml(claim.node_id)} · ${escapeHtml(claim.claim_id)}</code><span>${escapeHtml(claim.statement_zh)}</span></li>`).join("");
      const dag = (turn.program?.nodes || []).map((node) => `<span style="--op:${escapeHtml(opColors[node.operation] || '#fff')}"><b>${escapeHtml(node.operation)}</b><small>${escapeHtml(node.node_id)}</small></span>`).join("<i>→</i>");
      const sentenceRows = (generated.answer?.sentences || []).map((sentence) => `<li><b>${escapeHtml(sentence.role)}</b><span>${escapeHtml(sentence.text)}</span><code>${escapeHtml((sentence.claim_ids || []).join(" · "))}</code></li>`).join("");
      const provenance = turn.generation_provenance || {};
      const provenanceRows = ["question", "answer", "critic"].map((stage) => {
        const item = provenance[stage] || {};
        return `<div><b>${stage.toUpperCase()}</b><span>${escapeHtml(item.role || "—")} · ${escapeHtml(item.model || "—")}</span><code title="${escapeHtml(item.request_hash || "")}">${escapeHtml(shortHash(item.request_hash || "—"))}</code></div>`;
      }).join("");
      return `<article class="rsint-round-card ${turn.validation?.passed ? "passed" : "failed"}">
        <header><div class="rsint-round-index"><b>${String(turn.round_index).padStart(2, "0")}</b><span>ROUND</span></div><div><small>${escapeHtml(_ROUTE_LABELS[turn.round_index] || "INCREMENTAL READ")}</small><h3>${escapeHtml(turn.turn_id)}</h3><code>${escapeHtml(turn.program?.semantic_signature || "")}</code></div><div class="rsint-round-badges">${capabilities.map((value) => `<span>${escapeHtml(value)}</span>`).join("")}<em>${turn.validation?.passed ? "3-STAGE PASS" : "REJECTED"}</em></div></header>
        <div class="rsint-new-view-label"><b>本轮新增 ${newViews.length} 图</b><span>当前累计 ${(turn.available_view_ids || []).length}/11，未来图不进入上下文</span></div>
        <div class="rsint-new-views">${images}</div>
        <div class="rsint-dialogue-pair"><section><small>USER · ANSWER-BLIND QUESTION</small><p>${escapeHtml(question)}</p></section><section><small>ASSISTANT · CLAIM-GROUNDED VOICEOVER</small><p>${escapeHtml(answer)}</p></section></div>
        <div class="rsint-round-details">
          <details><summary>逐句证据绑定 <span>${(generated.answer?.sentences || []).length} sentences</span></summary><ol class="rsint-sentence-list">${sentenceRows}</ol></details>
          <details><summary>执行 DAG 与 node claims <span>${turn.claim_sheet?.claims?.length || 0} claims</span></summary><div class="rsint-dag">${dag}</div><ol class="rsint-claim-list">${claimRows}</ol></details>
          <details><summary>三角色 provenance <span>${escapeHtml((turn.answer_skill_ids || []).join(" + "))}</span></summary><div class="rsint-provenance">${provenanceRows}</div></details>
        </div>
      </article>`;
    }).join("");
    const exportLabels = ["episode · all turns", "isolated · prefix matched", "state-op · auxiliary"];
    $("#rsint-export-links").innerHTML = (payload.training_exports || []).map((value, index) => `<a href="${escapeHtml(value)}"><b>${escapeHtml(exportLabels[index] || "training export")}</b><code>${escapeHtml(value)}</code></a>`).join("");
    $$(".rsint-image-button").forEach((button) => button.addEventListener("click", () => {
      $("#dialog-image").src = button.dataset.image;
      text("#dialog-caption", button.dataset.caption);
      $("#image-dialog").showModal();
    }));
  }

  const _ROUTE_LABELS = {
    1: "WRITE · VISUAL INVENTORY",
    2: "READ · CO-VISIBILITY + METRIC",
    3: "UPDATE · TEMPORAL MEMORY",
    4: "COMPOSE · NON-CO-VISIBLE MAP",
    5: "TRANSFORM · OBJECT PERSPECTIVE",
    6: "VERIFY · LOOP + CALIBRATION",
    7: "COMMIT · STATE SUFFICIENCY",
  };

  async function loadRsintDialogue() {
    const response = await fetch("data/rsint_dialogue_pilot.v2.json");
    if (!response.ok) throw new Error(`Rs_int dialogue fetch failed: ${response.status}`);
    const payload = await response.json();
    if (payload.schema_version !== "epispace.rsint_dialogue_web.v1") throw new Error("unsupported Rs_int dialogue schema");
    state.rsintDialogueData = payload;
    renderRsintDialogue();
  }

  const COMPILED_DIALOGUE_BASE = "../data/trajectory_dialogues_v2";
  const COMPOSITION_DIALOGUE_BASE = "../data/t10_compositions_v2";
  const compiledClassLabels = {
    all: ["ALL", "全部受控轨迹"],
    T3: ["T3", "原地转向"],
    T4: ["T4", "物体环绕"],
    T7: ["T7", "高度干预"],
    T8: ["T8", "遮挡显露"],
    T10: ["T10", "目标视角读取"],
    T10C: ["T1→T10", "严格视角预测"],
  };
  const compiledScopeLabels = {
    trajectory_reasoning: "受控轨迹推理：当前 episode 的图像按时间递增释放，问题共享同一状态。",
    target_view_read_not_prediction: "目标视角读取：模型已经看见 T10 渲染图；用于 frame relation read，不宣称新视角预测。",
    source_to_heldout_perspective_prediction: "严格视角预测：先用 11 张 T1 漫游图建立状态，在不展示目标图时回答；下一轮才释放 T10 渲染图验证。",
  };

  function compiledFilteredEntries() {
    const needle = state.compiledDialogueSearch.trim().toLowerCase();
    return state.compiledDialogueEntries.filter((entry) => {
      const matchesClass = state.compiledDialogueClass === "all" || entry.trajectory_class === state.compiledDialogueClass;
      const matchesSearch = !needle || `${entry.acquisition_id} ${entry.scene || ""} ${entry.trajectory_class} ${entry.task_scope || ""}`.toLowerCase().includes(needle);
      return matchesClass && matchesSearch;
    });
  }

  function compiledArtifactUrl(entry) {
    return qaMediaUrl(entry.dialogue_path);
  }

  function compiledImageUrl(artifact, viewId) {
    return qaMediaUrl(artifact.rgb_paths?.[viewId] || "");
  }

  function renderCompiledClassStrip() {
    const counts = Object.fromEntries(Object.keys(compiledClassLabels).map((key) => [key, 0]));
    counts.all = state.compiledDialogueEntries.length;
    state.compiledDialogueEntries.forEach((entry) => { counts[entry.trajectory_class] = (counts[entry.trajectory_class] || 0) + 1; });
    $("#compiled-class-strip").innerHTML = Object.entries(compiledClassLabels).map(([key, [tag, label]]) => `<button type="button" class="${key === state.compiledDialogueClass ? "active" : ""}" data-compiled-class="${escapeHtml(key)}"><b>${escapeHtml(tag)}</b><span>${escapeHtml(label)}</span><small>${escapeHtml(counts[key] || 0)} EPISODES</small></button>`).join("");
    $$('[data-compiled-class]').forEach((button) => button.addEventListener("click", () => {
      state.compiledDialogueClass = button.dataset.compiledClass;
      renderCompiledClassStrip();
      renderCompiledList();
    }));
  }

  function renderCompiledMetrics() {
    const report = state.compiledDialogueReport || {};
    const requested = Object.values(report.requested || {}).reduce((sum, value) => sum + Number(value), 0);
    const rounds = Object.values(report.round_counts || {}).reduce((sum, value) => sum + Number(value), 0);
    const classViews = { T3: 6, T4: 8, T7: 3, T8: 6, T10: 3 };
    const images = Object.entries(report.compiled_counts || {}).reduce((sum, [key, value]) => sum + (classViews[key] || 0) * Number(value), 0);
    const rejected = (report.rejected || []).length;
    const strict = (state.compositionDialogueReport?.compiled || []).length;
    const metrics = [[requested, "episode SFT"], [rounds, "dialogue rounds"], [images, "ordered RGB"], [rejected, "compiler rejects"], [strict, "T1→T10 strict"]];
    $("#compiled-corpus-metrics").innerHTML = metrics.map(([value, label]) => `<div><b>${escapeHtml(value)}</b><span>${escapeHtml(label)}</span></div>`).join("");
  }

  function renderCompiledList() {
    const entries = compiledFilteredEntries();
    text("#compiled-dialogue-filter-count", `${entries.length} / ${state.compiledDialogueEntries.length} EPISODES`);
    if (!entries.length) {
      $("#compiled-dialogue-list").innerHTML = '<p class="compiled-empty">没有符合当前筛选条件的 episode。</p>';
      return;
    }
    $("#compiled-dialogue-list").innerHTML = entries.map((entry) => {
      const label = compiledClassLabels[entry.trajectory_class]?.[0] || entry.trajectory_class;
      const strict = entry.trajectory_class === "T10C";
      return `<button type="button" class="${entry.acquisition_id === state.compiledDialogueId ? "active" : ""}" data-compiled-id="${escapeHtml(entry.acquisition_id)}">
        <span><b>${escapeHtml((entry.scene || entry.acquisition_id).replaceAll("_", " "))}</b><em>${escapeHtml(label)}</em></span>
        <code>${escapeHtml(entry.acquisition_id)}</code>
        <small>${escapeHtml(entry.round_count || 4)} ROUNDS · ${strict ? "PREDICT THEN VERIFY" : "PASSED DAG + LANGUAGE GATES"}</small>
      </button>`;
    }).join("");
    $$('[data-compiled-id]').forEach((button) => button.addEventListener("click", () => { void loadCompiledArtifact(button.dataset.compiledId); }));
  }

  function renderCompiledArtifact(artifact) {
    state.compiledDialogueArtifact = artifact;
    state.compiledDialogueId = artifact.acquisition_id;
    renderCompiledList();
    text("#compiled-dialogue-status", artifact.task_scope === "source_to_heldout_perspective_prediction" ? "PREDICT → HELD-OUT VERIFY" : "TYPED DAG · ALL GATES PASS");
    text("#compiled-dialogue-id", artifact.acquisition_id);
    text("#compiled-dialogue-title", artifact.acquisition_id.replaceAll("_", " "));
    $("#compiled-dialogue-meta").innerHTML = [
      [artifact.trajectory_class, "CLASS"],
      [(artifact.rounds || []).length, "ROUNDS"],
      [(artifact.view_ids || []).length, "RGB"],
      [artifact.task_scope === "target_view_read_not_prediction" ? "READ" : artifact.task_scope === "source_to_heldout_perspective_prediction" ? "PREDICT" : "EPISODIC", "SCOPE"],
    ].map(([value, label]) => `<span><b>${escapeHtml(value)}</b><small>${escapeHtml(label)}</small></span>`).join("");
    $("#compiled-dialogue-scope").innerHTML = `<b>TASK SCOPE</b><span>${escapeHtml(compiledScopeLabels[artifact.task_scope] || artifact.task_scope)}</span>`;

    let releasedCount = 0;
    $("#compiled-rounds").innerHTML = (artifact.rounds || []).map((round) => {
      const newViews = round.new_view_ids || [];
      releasedCount += newViews.length;
      const capabilities = [round.capability?.primary, ...(round.capability?.supporting || [])].filter(Boolean);
      const images = newViews.length ? newViews.map((viewId) => {
        const url = compiledImageUrl(artifact, viewId);
        return `<figure><button type="button" class="compiled-image-button" data-image="${escapeHtml(url)}" data-caption="${escapeHtml(artifact.acquisition_id)} · ${escapeHtml(viewId)}"><img loading="lazy" src="${escapeHtml(url)}" alt="${escapeHtml(viewId)} RGB"></button><figcaption>${escapeHtml(viewId)}</figcaption></figure>`;
      }).join("") : '<div class="compiled-state-read"><b>0 NEW RGB</b><span>只读取此前提交的共享空间状态</span></div>';
      const sentences = (round.answer_sentences || []).map((sentence) => `<span class="${sentence.role === "transform" ? "transform" : sentence.role === "conclusion" ? "conclusion" : ""}"><b>${escapeHtml(sentence.role)}</b>${escapeHtml(sentence.text)}</span>`).join("");
      const dag = (round.program?.nodes || []).map((node) => `<span style="--op:${escapeHtml(opColors[node.operation] || "#fff")}"><b>${escapeHtml(node.operation)}</b><small>${escapeHtml(node.node_id)}</small><code>${escapeHtml(node.output_type || "")}</code></span>`).join("<i>→</i>");
      const debug = { answer_key: round.answer_key, diagnostics: round.diagnostics, evidence_view_ids: round.evidence_view_ids, claims: round.claim_sheet?.claims };
      return `<article class="compiled-round-card">
        <header><div class="compiled-round-index"><b>${String(round.round_index).padStart(2, "0")}</b><small>ROUND</small></div><div><h3>${escapeHtml(round.turn_id)}</h3><code>${escapeHtml(round.program?.semantic_signature || "")}</code></div><div class="compiled-round-tags">${capabilities.map((value) => `<span>${escapeHtml(value)}</span>`).join("")}</div></header>
        <div class="compiled-release"><b>本轮新增 ${newViews.length} 图</b><span>累计释放 ${releasedCount}/${artifact.view_ids.length} · evidence ${escapeHtml((round.evidence_view_ids || []).length)}</span></div>
        <div class="compiled-images">${images}</div>
        <div class="compiled-qa"><section><small>USER · CURRENT PREFIX ONLY</small><p>${escapeHtml(round.question_zh)}</p></section><section><small>ASSISTANT · SUPERVISED TARGET</small><p>${escapeHtml(round.answer_zh)}</p></section></div>
        <div class="compiled-sentences">${sentences}</div>
        <div class="compiled-hidden"><details><summary>隐藏监督 · typed DAG / claim / certificate</summary><div class="compiled-dag">${dag}</div><pre class="compiled-debug">${escapeHtml(JSON.stringify(debug, null, 2))}</pre></details></div>
      </article>`;
    }).join("");
    $$(".compiled-image-button").forEach((button) => button.addEventListener("click", () => {
      $("#dialog-image").src = button.dataset.image;
      text("#dialog-caption", button.dataset.caption);
      $("#image-dialog").showModal();
    }));
  }

  async function loadCompiledArtifact(acquisitionId) {
    const entry = state.compiledDialogueEntries.find((item) => item.acquisition_id === acquisitionId);
    if (!entry) return;
    text("#compiled-dialogue-status", "LOADING DIALOGUE JSON");
    const response = await fetch(compiledArtifactUrl(entry));
    if (!response.ok) throw new Error(`compiled dialogue fetch failed: ${response.status}`);
    const artifact = await response.json();
    renderCompiledArtifact(artifact);
  }

  function bindCompiledControls() {
    const search = $("#compiled-dialogue-search");
    search.addEventListener("input", () => {
      state.compiledDialogueSearch = search.value;
      renderCompiledList();
    });
  }

  function renderCompiledUnavailable(error) {
    text("#compiled-dialogue-status", "CORPUS UNAVAILABLE");
    text("#compiled-dialogue-id", String(error?.message || error));
    $("#compiled-rounds").innerHTML = `<p class="compiled-empty">无法读取全量 trajectory dialogue corpus；旧版 T1 配音对照仍可独立浏览。</p>`;
  }

  async function loadCompiledCorpus() {
    const [directResponse, compositionResponse] = await Promise.all([
      fetch(`${COMPILED_DIALOGUE_BASE}/corpus_report.json`),
      fetch(`${COMPOSITION_DIALOGUE_BASE}/corpus_report.json`),
    ]);
    if (!directResponse.ok) throw new Error(`trajectory dialogue report failed: ${directResponse.status}`);
    state.compiledDialogueReport = await directResponse.json();
    state.compositionDialogueReport = compositionResponse.ok ? await compositionResponse.json() : { compiled: [], rejected: [] };
    const direct = (state.compiledDialogueReport.compiled || []).map((entry) => ({ ...entry, task_scope: entry.task_scope || "trajectory_reasoning" }));
    const composition = (state.compositionDialogueReport.compiled || []).map((entry) => ({ ...entry, trajectory_class: "T10C", scene: entry.acquisition_id.replace(/_t1_to_t10_seed\d+$/, ""), task_scope: "source_to_heldout_perspective_prediction", round_count: 4 }));
    state.compiledDialogueEntries = [...direct, ...composition];
    renderCompiledMetrics();
    renderCompiledClassStrip();
    renderCompiledList();
    const defaultEntry = state.compiledDialogueEntries.find((entry) => entry.trajectory_class === "T8") || state.compiledDialogueEntries[0];
    if (defaultEntry) await loadCompiledArtifact(defaultEntry.acquisition_id);
  }

  const SCENE_DIALOGUE_BASE = "../data/scene_dialogues_v1";
  const sceneDialogueCache = new Map();
  const sceneDialogueDubbedCache = new Map();
  const DUBBED_SCENE_IDS = new Set(["Merom_0_int_seed17"]);
  const sceneRelationLabels = { behind: "后方", left_of: "左侧", in_front_of: "前方", right_of: "右侧" };

  function sceneDialogueEntries() {
    return Array.isArray(state.sceneDialogueReport?.compiled) ? state.sceneDialogueReport.compiled : [];
  }

  function sceneDialogueFilteredEntries() {
    const needle = state.sceneDialogueSearch.trim().toLowerCase();
    return sceneDialogueEntries().filter((entry) => {
      const matchesSearch = !needle || `${entry.acquisition_id} ${entry.split} ${(entry.rounds || []).join(" ")}`.toLowerCase().includes(needle);
      const matchesSplit = state.sceneDialogueSplit === "all" || entry.split === state.sceneDialogueSplit;
      const count = (entry.rounds || []).length;
      const matchesCoverage = state.sceneDialogueCoverage === "all" || (state.sceneDialogueCoverage === "7" ? count === 7 : count < 7);
      return matchesSearch && matchesSplit && matchesCoverage;
    });
  }

  function sceneDialogueImageUrl(sceneId, viewId) {
    return `${SCENE_DIALOGUE_BASE}/rgb/${encodeURIComponent(sceneId)}/${encodeURIComponent(viewId)}.png`;
  }

  function sceneViewNumber(viewId) {
    const match = String(viewId || "").match(/(\d+)$/);
    return match ? Number(match[1]) + 1 : viewId;
  }

  function renderSceneDialogueCatalog() {
    const report = state.sceneDialogueReport;
    if (!report) return;
    const compiled = sceneDialogueEntries();
    const rejected = report.rejected && typeof report.rejected === "object" ? Object.entries(report.rejected) : [];
    const rounds = compiled.reduce((total, entry) => total + (entry.rounds || []).length, 0);
    const metrics = [
      [compiled.length + rejected.length, "attempted scenes"],
      [compiled.length, "compiled episodes"],
      [rejected.length, "rejected scenes"],
      [rounds, "dialogue rounds"],
      [compiled.length * 11, "ordered RGB"],
      [`${DUBBED_SCENE_IDS.size}/${compiled.length}`, "LLM dubbed pilot"],
    ];
    $("#scene-corpus-metrics").innerHTML = metrics.map(([value, label]) => `<div><b>${escapeHtml(value)}</b><span>${escapeHtml(label)}</span></div>`).join("");
    text("#metric-views", compiled.length + rejected.length);
    text("#metric-queries", compiled.length);
    text("#metric-variants", rounds);
    text("#metric-psnr", `${DUBBED_SCENE_IDS.size}/${compiled.length}`);

    const directions = Object.entries(report.direction_balance || {});
    const directionMax = Math.max(1, ...directions.map(([, count]) => Number(count)));
    $("#scene-direction-balance").innerHTML = directions.map(([relation, count]) => `<div><span>${escapeHtml(sceneRelationLabels[relation] || relation)}<small>${escapeHtml(relation)}</small></span><i><b style="width:${Math.max(4, Number(count) / directionMax * 100)}%"></b></i><strong>${escapeHtml(count)}</strong></div>`).join("");
    $("#scene-rejection-list").innerHTML = rejected.map(([sceneId, reason]) => `<div><code>${escapeHtml(sceneId)}</code><p>${escapeHtml(reason)}</p></div>`).join("");
    renderSceneDialogueList();
  }

  function renderSceneDialogueList() {
    const entries = sceneDialogueFilteredEntries();
    text("#scene-dialogue-filter-count", `${entries.length} / ${sceneDialogueEntries().length} COMPILED SCENES`);
    if (!entries.length) {
      $("#scene-dialogue-list").innerHTML = '<p class="scene-dialogue-empty">没有符合当前筛选条件的场景。</p>';
      return;
    }
    $("#scene-dialogue-list").innerHTML = entries.map((entry) => {
      const rounds = entry.rounds || [];
      const skipped = entry.skipped || [];
      return `<button type="button" data-scene-dialogue-id="${escapeHtml(entry.acquisition_id)}" class="${entry.acquisition_id === state.sceneDialogueSceneId ? "active" : ""}">
        <span><b>${escapeHtml(entry.acquisition_id.replace(/_seed\d+$/, ""))}</b><em>${escapeHtml(entry.split || "unsplit")}</em></span>
        <code>${escapeHtml(entry.acquisition_id)}</code>
        <small>${rounds.length}/7 ROUNDS · ${skipped.length ? `${skipped.length} SKIPPED` : "FULL COVERAGE"}${DUBBED_SCENE_IDS.has(entry.acquisition_id) ? " · LLM DUBBED" : ""}</small>
        <i style="--coverage:${Math.min(100, rounds.length / 7 * 100)}%"></i>
      </button>`;
    }).join("");
    $$("[data-scene-dialogue-id]").forEach((button) => button.addEventListener("click", () => { void loadSceneDialogueScene(button.dataset.sceneDialogueId); }));
  }

  function renderSceneDialogueUnavailable(error) {
    text("#scene-dialogue-status", "CORPUS UNAVAILABLE");
    text("#scene-dialogue-id", String(error?.message || error));
    $("#scene-round-timeline").innerHTML = `<p class="scene-dialogue-empty">无法读取 <code>${escapeHtml(SCENE_DIALOGUE_BASE)}/corpus_report.json</code>。其他页面不受影响。</p>`;
  }

  function sceneSurfaceCard(turn, version) {
    const isDubbed = version === "dubbed";
    const dubbing = isDubbed ? turn.dubbing || {} : {};
    const versionLabel = isDubbed ? "STAGE B · LLM DUBBING" : "STAGE A · GEOMETRY TEMPLATE";
    const boundary = isDubbed ? "只改语言表面 · 当前为 pilot artifact" : "确定性编译 · 当前 SFT 导出来源";
    const metadata = isDubbed ? `<div class="scene-dubbing-meta"><span>${escapeHtml(dubbing.strategy || "strategy unspecified")}</span><span>${escapeHtml(dubbing.style || "style unspecified")}</span><span>${escapeHtml(dubbing.attempts || 0)} ATTEMPT</span><em class="${dubbing.fallback ? "fallback" : "accepted"}">${dubbing.fallback ? "TEMPLATE FALLBACK" : "LOCAL SURFACE GATE PASS"}</em></div>` : "";
    return `<article class="scene-surface-card ${version}">
      <header><div><small>${versionLabel}</small><b>${boundary}</b></div>${isDubbed ? '<span>CLAUDE CLI</span>' : '<span>DETERMINISTIC</span>'}</header>
      <section><small>USER QUESTION</small><p>${escapeHtml(turn.question_zh)}</p></section>
      <section><small>ASSISTANT ANSWER</small><p>${escapeHtml(turn.answer_zh)}</p></section>
      ${metadata}
    </article>`;
  }

  function renderSceneVersionSwitch(hasDubbed) {
    const effectiveMode = hasDubbed ? state.sceneDialogueMode : "template";
    $$('[data-scene-version]').forEach((button) => {
      const requiresDubbed = button.dataset.sceneVersion !== "template";
      button.disabled = requiresDubbed && !hasDubbed;
      button.classList.toggle("active", button.dataset.sceneVersion === effectiveMode);
    });
    text(
      "#scene-dubbing-availability",
      hasDubbed
        ? "该场景已有 Claude 配音 pilot；可逐轮比较模板与配音，几何真值完全共享。"
        : "该场景尚未配音；当前只展示确定性模板，不把缺失版本伪装成完成。",
    );
  }

  function renderSceneDialogueScene(scene, dubbed = null) {
    const sceneId = scene.acquisition_id;
    state.sceneDialogueScene = scene;
    state.sceneDialogueDubbed = dubbed;
    state.sceneDialogueSceneId = sceneId;
    renderSceneDialogueList();
    renderSceneVersionSwitch(Boolean(dubbed));
    text("#scene-dialogue-status", dubbed ? "TRUTH COMPILED · LLM DUBBED PILOT" : "TRUTH COMPILED · TEMPLATE ONLY");
    text("#scene-dialogue-id", sceneId);
    text("#scene-dialogue-title", sceneId.replaceAll("_", " "));
    $("#scene-dialogue-meta").innerHTML = [
      [scene.split || "—", "SPLIT"],
      [(scene.rounds || []).length, "ROUNDS"],
      [(scene.view_ids || []).length, "RGB"],
      [(scene.skipped_rounds || []).length, "SKIPPED"],
      [dubbed ? "YES" : "NO", "DUBBED"],
    ].map(([value, label]) => `<span><b>${escapeHtml(value)}</b><small>${escapeHtml(label)}</small></span>`).join("");
    $("#scene-frame-contract").innerHTML = `<b>FRAME CONTRACT</b><p>${escapeHtml(scene.frame_contract?.surface_zh || "未提供场景方向约定。")}</p><code>${escapeHtml(scene.frame_contract?.frame_id || "frame unspecified")}</code>`;
    if ((scene.view_ids || []).length) $("#hero-image").src = sceneDialogueImageUrl(sceneId, scene.view_ids[0]);

    const dubbedByTurn = new Map((dubbed?.rounds || []).map((turn) => [turn.turn_id, turn]));
    const effectiveMode = dubbed ? state.sceneDialogueMode : "template";
    const accumulated = new Set();
    $("#scene-round-timeline").innerHTML = (scene.rounds || []).map((turn) => {
      const dubbedTurn = dubbedByTurn.get(turn.turn_id);
      const newViews = turn.new_view_ids || [];
      newViews.forEach((viewId) => accumulated.add(viewId));
      const capabilities = [turn.capability?.primary, ...(turn.capability?.supporting || [])].filter(Boolean);
      const images = newViews.length ? newViews.map((viewId) => `<figure><button type="button" class="scene-dialogue-image" data-image="${escapeHtml(sceneDialogueImageUrl(sceneId, viewId))}" data-caption="${escapeHtml(sceneId)} · 图${escapeHtml(sceneViewNumber(viewId))} · ${escapeHtml(viewId)}"><img loading="lazy" src="${escapeHtml(sceneDialogueImageUrl(sceneId, viewId))}" alt="${escapeHtml(sceneId)} ${escapeHtml(viewId)} RGB"></button><figcaption><b>图${escapeHtml(sceneViewNumber(viewId))}</b><span>${escapeHtml(viewId)}</span></figcaption></figure>`).join("") : '<div class="scene-state-read"><b>0 NEW RGB</b><span>本轮只读取先前对话中的共享空间状态</span></div>';
      const answerSentences = (turn.answer_sentences || []).length ? turn.answer_sentences : [{ role: "conclusion", text: turn.answer_zh }];
      const nodes = (turn.program?.nodes || []).map((node) => `<span style="--op:${escapeHtml(opColors[node.operation] || "#fff")}"><b>${escapeHtml(node.operation)}</b><small>${escapeHtml(node.node_id)}</small><code>${escapeHtml(node.output_type || "")}</code></span>`).join("<i>→</i>");
      const claims = (turn.claim_sheet?.claims || []).map((claim) => `<li><div><b>${escapeHtml(claim.reasoning_role || claim.kind || "fact")}</b><code>${escapeHtml(claim.claim_id)} · ${escapeHtml(claim.node_id)}</code></div><p>${escapeHtml(claim.statement_zh)}</p><small>${escapeHtml((claim.evidence_view_ids || []).join(" · ") || "global truth")}</small></li>`).join("");
      const crossView = turn.diagnostics && Object.prototype.hasOwnProperty.call(turn.diagnostics, "cross_view") ? (turn.diagnostics.cross_view ? "CROSS-VIEW" : "SINGLE-VIEW") : null;
      const diagnosticPayload = { answer_key: turn.answer_key || {}, diagnostics: turn.diagnostics || {}, new_view_ids: newViews, evidence_view_ids: turn.evidence_view_ids || [], dubbing: dubbedTurn?.dubbing || null };
      let surfaces = sceneSurfaceCard(turn, "template");
      if (effectiveMode === "dubbed" && dubbedTurn) surfaces = sceneSurfaceCard(dubbedTurn, "dubbed");
      if (effectiveMode === "compare" && dubbedTurn) surfaces = sceneSurfaceCard(turn, "template") + sceneSurfaceCard(dubbedTurn, "dubbed");
      return `<article class="scene-round-card">
        <header><div class="scene-round-number"><b>${String(turn.round_index).padStart(2, "0")}</b><span>ROUND</span></div><div><small>${escapeHtml(turn.capability?.sense_nova_subtask || "spatial read")}</small><h3>${escapeHtml(turn.turn_id)}</h3><code>${escapeHtml(turn.program?.semantic_signature || "")}</code></div><div class="scene-round-tags">${capabilities.map((value) => `<span>${escapeHtml(value)}</span>`).join("")}${crossView ? `<em class="${crossView === "CROSS-VIEW" ? "cross" : "single"}">${crossView}</em>` : ""}</div></header>
        <div class="scene-view-release"><b>本轮新增 ${newViews.length} 张 RGB</b><span>累计 ${accumulated.size}/${(scene.view_ids || []).length} · 未来图严格不可见</span></div>
        <div class="scene-new-views">${images}</div>
        <div class="scene-surface-grid ${effectiveMode === "compare" && dubbedTurn ? "compare" : "single"}">${surfaces}</div>
        <div class="scene-hidden-supervision">
          <details><summary>真值剧本的 cue → transform → conclusion 结构 <span>${answerSentences.length} sentences</span></summary><div class="scene-role-sequence">${answerSentences.map((sentence) => `<span class="role-${escapeHtml(sentence.role)}"><b>${escapeHtml(sentence.role === "evidence" ? "cue" : sentence.role)}</b>${escapeHtml(sentence.text)}</span>`).join("<i>→</i>")}</div></details>
          <details><summary>Typed operation graph · 不进入主输入 <span>${(turn.program?.nodes || []).length} nodes</span></summary><div class="scene-program-dag">${nodes}</div></details>
          <details><summary>Geometry claim sheet · 编译证书 <span>${(turn.claim_sheet?.claims || []).length} claims</span></summary><ol class="scene-claim-sheet">${claims}</ol></details>
          <details><summary>Answer key + diagnostics + dubbing gate <span>${escapeHtml((turn.evidence_view_ids || []).length)} evidence views</span></summary><pre>${escapeHtml(JSON.stringify(diagnosticPayload, null, 2))}</pre></details>
        </div>
      </article>`;
    }).join("") || '<p class="scene-dialogue-empty">该场景没有通过认证的对话轮次。</p>';

    $$(".scene-dialogue-image").forEach((button) => button.addEventListener("click", () => {
      $("#dialog-image").src = button.dataset.image;
      text("#dialog-caption", button.dataset.caption);
      $("#image-dialog").showModal();
    }));
  }

  async function loadSceneDialogueScene(sceneId) {
    if (!sceneId) return;
    state.sceneDialogueSceneId = sceneId;
    renderSceneDialogueList();
    text("#scene-dialogue-status", "LOADING SCENE");
    text("#scene-dialogue-id", sceneId);
    try {
      let scene = sceneDialogueCache.get(sceneId);
      if (!scene) {
        const response = await fetch(`${SCENE_DIALOGUE_BASE}/scenes/${encodeURIComponent(sceneId)}.dialogue.json`);
        if (!response.ok) throw new Error(`scene dialogue fetch failed: ${response.status}`);
        scene = await response.json();
        if (scene.schema_version !== "epispace.scene_dialogue_truth.v1") throw new Error("unsupported scene dialogue schema");
        sceneDialogueCache.set(sceneId, scene);
      }
      let dubbed = sceneDialogueDubbedCache.get(sceneId);
      if (dubbed === undefined) {
        const response = await fetch(`${SCENE_DIALOGUE_BASE}/dubbed/${encodeURIComponent(sceneId)}.dubbed.json`);
        if (response.ok) {
          dubbed = await response.json();
        } else if (response.status === 404) {
          dubbed = null;
        } else {
          throw new Error(`scene dubbing fetch failed: ${response.status}`);
        }
        sceneDialogueDubbedCache.set(sceneId, dubbed);
      }
      if (state.sceneDialogueSceneId === sceneId) renderSceneDialogueScene(scene, dubbed);
    } catch (error) {
      renderSceneDialogueUnavailable(error);
    }
  }

  async function loadSceneDialogueCorpus() {
    const response = await fetch(`${SCENE_DIALOGUE_BASE}/corpus_report.json`);
    if (!response.ok) throw new Error(`scene corpus report fetch failed: ${response.status}`);
    const report = await response.json();
    if (!Array.isArray(report.compiled) || !report.rejected) throw new Error("invalid scene corpus report");
    state.sceneDialogueReport = report;
    renderSceneDialogueCatalog();
    const defaultEntry = report.compiled.find((entry) => entry.acquisition_id === "Merom_0_int_seed17") || report.compiled[0];
    if (defaultEntry) await loadSceneDialogueScene(defaultEntry.acquisition_id);
  }

  function bindSceneDialogueControls() {
    $("#scene-dialogue-search")?.addEventListener("input", (event) => { state.sceneDialogueSearch = event.target.value; renderSceneDialogueList(); });
    $("#scene-dialogue-split")?.addEventListener("change", (event) => { state.sceneDialogueSplit = event.target.value; renderSceneDialogueList(); });
    $("#scene-dialogue-coverage")?.addEventListener("change", (event) => { state.sceneDialogueCoverage = event.target.value; renderSceneDialogueList(); });
    $$("[data-scene-version]").forEach((button) => button.addEventListener("click", () => {
      if (button.disabled) return;
      state.sceneDialogueMode = button.dataset.sceneVersion;
      if (state.sceneDialogueScene) renderSceneDialogueScene(state.sceneDialogueScene, state.sceneDialogueDubbed);
    }));
  }

  function renderQaPilotSummary() {
    const pilot = state.qaPilotData;
    const episodes = qaPilotEpisodes();
    const questions = episodes.flatMap(qaEpisodeQuestions);
    const trainQuestions = questions.filter((item) => /train|sft/i.test(qaDisposition(item)) && !/held|eval|reject/i.test(qaDisposition(item)));
    const heldoutQuestions = questions.filter((item) => /held|eval|reject/i.test(qaDisposition(item)));
    const shortfallQuestions = questions.filter((item) => /shortfall|development_only/i.test(qaDisposition(item)));
    const eligibleEpisodes = episodes.filter((episode) => qaOptimizerEligible(episode) === true);
    const status = String(firstDefined(pilot.status, pilot.release_status, pilot.authority?.status, "development_only"));
    const firstRecord = questions[0] || {};
    const firstStage = firstDefined(firstRecord.generation_provenance?.question, firstRecord.generation_provenance?.answer, {});
    const provenance = compactObject(firstDefined(pilot.provenance, pilot.backend, pilot.generation?.provenance, firstStage, {}));
    const provider = firstDefined(provenance.provider, provenance.backend, provenance.engine, pilot.backend?.provider, "unknown backend");
    const model = firstDefined(provenance.model, provenance.model_id, pilot.backend?.model, "model unspecified");
    const requestHash = firstDefined(provenance.request_hash, provenance.prompt_hash, provenance.manifest_sha256, pilot.manifest_sha256, "hash unavailable");
    $("#qa-pilot-authority")?.classList.remove("unavailable");
    $("#qa-pilot-authority")?.classList.toggle("development", !/verified|release/i.test(status));
    text("#qa-pilot-status", status.replaceAll("_", " ").toUpperCase());
    text("#qa-pilot-title", firstDefined(pilot.title_zh, pilot.title, pilot.name, pilot.dataset_id, "Claim-grounded episodic QA pilot"));
    text("#qa-pilot-boundary", firstDefined(pilot.claim_boundary_zh, pilot.authority?.claim_boundary_zh, pilot.source_semantic_visual_audit?.interpretation_zh, "开发态 pilot：通过确定性 QA 门不等于通过正式 RGB 语义复核。"));
    $("#qa-pilot-provenance").innerHTML = `<small>LANGUAGE BACKEND</small><b>${escapeHtml(provider)} · ${escapeHtml(model)}</b><small>REQUEST / MANIFEST</small><code title="${escapeHtml(requestHash)}">${escapeHtml(shortHash(requestHash))}</code>`;
    const metrics = [
      [episodes.length, "episode batches"],
      [questions.length, "claim-grounded QA"],
      [trainQuestions.length, "train disposition"],
      [heldoutQuestions.length, "held-out / eval"],
      [shortfallQuestions.length, "batch shortfall"],
      [eligibleEpisodes.length, "optimizer eligible"],
    ];
    $("#qa-pilot-metrics").innerHTML = metrics.map(([value, label]) => `<div><b>${escapeHtml(value)}</b><span>${escapeHtml(label)}</span></div>`).join("");
  }

  function renderQaEpisodeList() {
    const episodes = qaPilotEpisodes();
    $("#qa-episode-list").innerHTML = episodes.map((episode, index) => {
      const identifier = firstDefined(episode.episode_id, episode.batch_id, episode.batch_plan?.batch_id, episode.batch_plan?.episode_id, episode.id, `episode-${index + 1}`);
      const scene = firstDefined(episode.scene_id, episode.scene, episode.batch_plan?.scene_id, episode.trajectory_class, episode.batch_plan?.trajectory_class, "scene unknown");
      const eligible = qaOptimizerEligible(episode);
      const count = qaEpisodeQuestions(episode).length;
      return `<button type="button" class="${index === state.qaEpisodeIndex ? "active" : ""}" data-qa-episode-index="${index}"><span><b>${escapeHtml(firstDefined(episode.trajectory_class, episode.batch_plan?.trajectory_class, episode.task_class, `E${index + 1}`))}</b><em class="${eligible === true ? "eligible" : eligible === false ? "blocked" : "unknown"}">${eligible === true ? "OPTIMIZE" : eligible === false ? "NO STEP" : "UNSET"}</em></span><code>${escapeHtml(identifier)}</code><small>${escapeHtml(scene)} · ${count} QA</small></button>`;
    }).join("");
    $$("#qa-episode-list button").forEach((button) => button.addEventListener("click", () => {
      state.qaEpisodeIndex = Number(button.dataset.qaEpisodeIndex);
      renderQaPilot();
    }));
  }

  function renderQaTrajectory(episode) {
    const views = qaEpisodeViews(episode);
    text("#qa-trajectory-count", `${views.length} ORDERED VIEWS`);
    if (!views.length) {
      $("#qa-trajectory-strip").innerHTML = `<p class="qa-missing">该 artifact 没有内嵌 view 列表；请在生成 manifest 中写入按输入顺序排列的 RGB 路径。</p>`;
      return;
    }
    $("#qa-trajectory-strip").innerHTML = views.map((view, index) => {
      const viewId = qaViewId(view, index);
      const source = qaViewImage(view);
      return `<figure data-qa-view-index="${index}" class="${source ? "" : "missing"}">${source ? `<button type="button" aria-label="放大 ${escapeHtml(viewId)}"><img loading="lazy" src="${escapeHtml(source)}" alt="${escapeHtml(viewId)}"></button>` : `<div>NO RGB PATH</div>`}<figcaption><b>${index + 1}</b><span>${escapeHtml(viewId)}</span></figcaption></figure>`;
    }).join("");
    $$("#qa-trajectory-strip figure button").forEach((button) => button.addEventListener("click", () => {
      const figure = button.closest("figure");
      const view = views[Number(figure.dataset.qaViewIndex)];
      const source = qaViewImage(view);
      if (!source || !$("#image-dialog")) return;
      $("#dialog-image").src = source;
      text("#dialog-caption", `${qaViewId(view, Number(figure.dataset.qaViewIndex))} · QA PILOT MODEL INPUT`);
      $("#image-dialog").showModal();
    }));
  }

  function renderQaClaims(item) {
    const claimSheet = firstDefined(item.claim_sheet, item.claims, item.supervision?.claim_sheet, {});
    const claims = Array.isArray(claimSheet) ? claimSheet : firstArray(claimSheet.claims, claimSheet.items, claimSheet.required_claims);
    if (!claims.length) return `<p class="qa-missing">没有内嵌 claim sheet。</p>`;
    return `<ol>${claims.map((claim) => {
      const id = firstDefined(claim?.claim_id, claim?.id, "claim");
      const content = firstDefined(claim?.surface_zh, claim?.statement_zh, claim?.statement, claim?.text, claim?.value, claim);
      return `<li><code>${escapeHtml(id)}</code><span>${escapeHtml(typeof content === "object" ? JSON.stringify(content) : content)}</span></li>`;
    }).join("")}</ol>`;
  }

  function renderQaCertificate(item) {
    const compilerProof = compactObject({
      program: item?.program,
      answer_contract: item?.claim_sheet?.answer_contract,
      required_claim_ids: item?.claim_sheet?.required_claim_ids,
      source_episode_ir_sha256: item?.claim_sheet?.source_episode_ir_sha256,
      source_bundle: item?.claim_sheet?.source_bundle,
    });
    const certificate = firstDefined(item.certificate, item.geometry_certificate, item.claim_sheet?.certificate, item.source?.certificate, item.supervision?.certificate, Object.keys(compilerProof).length ? compilerProof : null);
    if (!certificate || (Array.isArray(certificate) && !certificate.length)) return `<p class="qa-missing">没有内嵌 certificate。</p>`;
    const values = Array.isArray(certificate) ? certificate : [certificate];
    return values.map((entry) => `<pre>${escapeHtml(typeof entry === "string" ? entry : JSON.stringify(entry, null, 2))}</pre>`).join("");
  }

  function renderQaValidation(item) {
    const validation = qaValidationState(item);
    const report = validation.report || {};
    const checks = Array.isArray(report) ? report : firstArray(report.checks, report.validators, report.results);
    const nestedChecks = ["question", "answer", "critic"].flatMap((stage) => Object.entries(report?.[stage]?.checks || {}).filter(([, passed]) => passed).map(([name]) => `${stage}.${name}`));
    const labels = checks.length ? checks.map((check) => typeof check === "string" ? check : firstDefined(check.name, check.check, check.id, JSON.stringify(check))) : [...Object.entries(report).filter(([, value]) => value === true).map(([key]) => key), ...nestedChecks];
    return `<div class="qa-validation-summary ${validation.passed ? "pass" : "fail"}"><b>${validation.passed ? "VALIDATION PASS" : "VALIDATION FAIL"}</b><span>${validation.errors.length ? validation.errors.map((error) => typeof error === "string" ? error : JSON.stringify(error)).join(" · ") : "no reported errors"}</span></div>${labels.length ? `<div class="qa-validation-checks">${labels.map((label) => `<span>✓ ${escapeHtml(label)}</span>`).join("")}</div>` : ""}`;
  }

  function renderQaProvenance(item) {
    if (item?.generation_provenance && typeof item.generation_provenance === "object") return `<pre>${escapeHtml(JSON.stringify(item.generation_provenance, null, 2))}</pre>`;
    const provenance = compactObject(firstDefined(item.provenance, item.generation?.provenance, item.backend, {}));
    const requestHash = firstDefined(provenance.request_hash, provenance.prompt_hash, item.request_hash, item.generation_request_hash);
    const payload = compactObject({
      provider: firstDefined(provenance.provider, provenance.backend, item.provider),
      model: firstDefined(provenance.model, provenance.model_id, item.model),
      stage: firstDefined(provenance.stage, item.generation_stage),
      request_hash: requestHash,
      cache_hit: firstDefined(provenance.cache_hit, item.cache_hit),
      generated_at: firstDefined(provenance.generated_at, item.generated_at),
    });
    return Object.keys(payload).length ? `<pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre>` : `<p class="qa-missing">未记录 item-level API provenance；请至少保存 provider、model 与 request hash。</p>`;
  }

  function renderQaComparisons(episode) {
    const questions = qaEpisodeQuestions(episode);
    text("#qa-question-count", `${questions.length} QUESTIONS`);
    $("#qa-comparison-list").innerHTML = questions.map((item, index) => {
      const identifier = firstDefined(item.fact_id, item.question_id, item.query_id, item.id, `q-${index + 1}`);
      const disposition = qaDisposition(item);
      const directEligibility = qaOptimizerEligible(item);
      const eligible = directEligibility === undefined ? qaOptimizerEligible(episode) === true && disposition === "train_candidate" : directEligibility;
      const capabilities = qaCapabilities(item);
      const validation = qaValidationState(item);
      const oldQuestion = qaQuestionText(item, "old");
      const oldAnswer = firstDefined(item?.original?.answer_zh, item?.old_answer_zh, item?.source?.answer_zh);
      const newQuestion = qaQuestionText(item, "new");
      const answer = qaAnswerText(item);
      return `<article class="qa-comparison-card ${validation.passed ? "verified" : "rejected"}">
        <header><div><span>Q${index + 1}</span><code>${escapeHtml(identifier)}</code></div><div class="qa-card-badges"><em class="${/held|eval|reject/i.test(disposition) ? "heldout" : "train"}">${escapeHtml(disposition.replaceAll("_", " "))}</em><em class="${eligible === true ? "eligible" : eligible === false ? "blocked" : "unknown"}">${eligible === true ? "LOSS ON" : eligible === false ? "NO LOSS" : "LOSS UNSET"}</em></div></header>
        <div class="qa-card-capabilities">${capabilities.length ? capabilities.map((ability) => `<span>${escapeHtml(ability)}</span>`).join("") : "<span>ABILITY UNSET</span>"}</div>
        <div class="qa-old-new"><section class="old"><small>OLD · SOURCE TEMPLATE</small><p>${escapeHtml(oldQuestion)}</p>${oldAnswer ? `<em>SOURCE TARGET · ${escapeHtml(oldAnswer)}</em>` : ""}</section><i>→</i><section class="new"><small>NEW · ANSWER-BLIND NATURALIZATION</small><p>${escapeHtml(newQuestion)}</p></section></div>
        <section class="qa-surface-answer"><small>SHORT SFT / EVAL TARGET</small><p>${escapeHtml(answer)}</p></section>
        <div class="qa-card-details">
          <details><summary>Claim sheet <span>回答允许引用的事实</span></summary><div class="qa-claim-list">${renderQaClaims(item)}</div></details>
          <details><summary>Geometry certificate <span>仅编译与审计可见</span></summary><div class="qa-certificate">${renderQaCertificate(item)}</div></details>
          <details><summary>Validation <span>${validation.passed ? "PASS" : "FAIL"}</span></summary><div>${renderQaValidation(item)}</div></details>
          <details><summary>Codex / API provenance <span>可复现请求</span></summary><div class="qa-item-provenance">${renderQaProvenance(item)}</div></details>
        </div>
      </article>`;
    }).join("") || `<p class="qa-missing">该 episode 没有可展示的 QA items。</p>`;
  }

  function renderQaPilot() {
    const episodes = qaPilotEpisodes();
    if (!episodes.length) {
      renderQaPilotUnavailable(new Error("artifact loaded, but no episodes/batches were found"));
      return;
    }
    state.qaEpisodeIndex = Math.min(state.qaEpisodeIndex, episodes.length - 1);
    const episode = episodes[state.qaEpisodeIndex];
    const plan = episode.batch_plan || {};
    const identifier = firstDefined(episode.episode_id, episode.batch_id, plan.batch_id, plan.episode_id, episode.id, `episode-${state.qaEpisodeIndex + 1}`);
    const eligible = qaOptimizerEligible(episode);
    const questions = qaEpisodeQuestions(episode);
    const abilities = [...new Set([...firstArray(plan.capabilities), ...questions.flatMap(qaCapabilities)])];
    renderQaPilotSummary();
    renderQaEpisodeList();
    text("#qa-episode-class", firstDefined(episode.trajectory_class, plan.trajectory_class, episode.task_class, episode.split, plan.split, "EPISODE BATCH"));
    text("#qa-episode-id", identifier);
    text("#qa-episode-title", firstDefined(episode.title_zh, episode.scene_id, plan.scene_id, episode.scene, "共享证据 QA episode"));
    const failedChecks = Object.entries(episode.optimizer_decision?.checks || {}).filter(([, passed]) => !passed).map(([name]) => name);
    const exclusionReasons = [...firstArray(episode.ineligibility_reasons, episode.optimizer?.reasons, episode.exclusion_reasons), ...failedChecks];
    $("#qa-episode-disposition").innerHTML = `<span class="${eligible === true ? "eligible" : eligible === false ? "blocked" : "unknown"}">${eligible === true ? "OPTIMIZER ELIGIBLE" : eligible === false ? "EXCLUDED FROM OPTIMIZER" : "ELIGIBILITY UNSET"}</span>${exclusionReasons.map((reason) => `<small>${escapeHtml(reason)}</small>`).join("")}`;
    $("#qa-episode-capabilities").innerHTML = abilities.length ? abilities.map((ability) => `<span>${escapeHtml(ability)}</span>`).join("") : `<span>CAPABILITIES UNSET</span>`;
    renderQaTrajectory(episode);
    renderQaComparisons(episode);
    $("#qa-pilot-browser").hidden = false;
    $("#qa-pilot-empty").hidden = true;
  }

  async function loadQaGenerationPilot() {
    const response = await fetch("data/qa_generation_pilot.v1.json");
    if (!response.ok) throw new Error(`QA pilot fetch failed: ${response.status}`);
    const pilot = await response.json();
    if (!pilot || typeof pilot !== "object") throw new Error("QA pilot root must be a JSON object");
    state.qaPilotData = pilot;
    renderQaPilot();
  }

  function releaseIsCandidate() {
    return state.releaseData?.status === "machine_verified_candidate";
  }

  function filteredReleaseEpisodes() {
    const needle = state.releaseSearch.trim().toLowerCase();
    return state.releaseData.episodes.filter((episode) => {
      if (state.releaseSplit !== "all" && episode.split !== state.releaseSplit) return false;
      if (state.releaseClass !== "all" && episode.trajectory_class !== state.releaseClass) return false;
      if (!needle) return true;
      return [episode.episode_id, episode.scene_id, episode.trajectory_class, ...Object.keys(episode.programs), ...Object.keys(episode.task_types)].join(" ").toLowerCase().includes(needle);
    });
  }

  function renderReleaseSummary() {
    const release = state.releaseData;
    const funnel = release.funnel;
    const candidate = releaseIsCandidate();
    const semanticAudit = release.quality.semantic_visual_audit;
    const auditFailed = candidate && semanticAudit.status === "failed";
    $("#release .release-authority")?.classList.toggle("candidate", candidate);
    $("#release .release-authority")?.classList.toggle("rework", auditFailed);
    $("#release")?.classList.toggle("candidate-mode", candidate);
    $("#release")?.classList.toggle("audit-failed", auditFailed);
    text("#release-status", auditFailed ? "REWORK REQUIRED · AUDIT FAILED" : candidate ? "MACHINE-VERIFIED CANDIDATE" : "VERIFIED RELEASE");
    text("#release-dataset-id", release.provenance.dataset_id);
    text("#release-chapter", auditFailed ? "03 · CANDIDATE FAILED · REWORK REQUIRED" : candidate ? "03 · MACHINE CANDIDATE · AUDIT PENDING" : "03 · VERIFIED DATA RELEASE");
    text("#release-heading", auditFailed ? "自动规则不是终点，\nRGB 复核已经打回。" : candidate ? "机器门已经闭合，\n语义画质仍待独立复核。" : "不是一页统计数字，\n而是一条可复核的数据链。");
    text("#release-description", candidate ? release.authority.claim_boundary_zh : "此区读取通过 final_release_index 全量复验、并与 manifest 和每个 artifact SHA-256 闭合的 release。轨迹是采集证据；这里是模型实际可消费的数据资产。");
    text("#release-authority-note", auditFailed ? "NOT A FORMAL RELEASE · FAILED / REWORK_REQUIRED" : candidate ? "NOT A FORMAL RELEASE · SEMANTIC RGB AUDIT PENDING" : "FINAL RELEASE INDEX · FAIL CLOSED");
    text("#release-manifest-label", candidate ? "PIPELINE CONFIG SHA-256" : "MANIFEST SHA-256");
    text("#release-index-label", candidate ? "EPISODE IR SHA-256" : "FINAL INDEX SHA-256");
    text("#release-manifest-sha", shortHash(candidate ? release.provenance.pipeline_config_sha256 : release.provenance.release_manifest_sha256));
    text("#release-index-sha", shortHash(candidate ? release.provenance.episode_ir_sha256 : release.provenance.final_release_index_sha256));
    text("#release-browser-title", `浏览全部 ${release.episodes.length} 条编译 Episode`);
    text("#release-browser-description", auditFailed ? "这里保留全部机器候选用于定位问题，不代表可以训练或发布。独立 RGB 复核已经发现 major / unreviewable 样本，必须修正规则并重新编译、重新抽审。" : candidate ? "这里投影全部可编译 Episode；featured 只决定首屏顺序，不删数据。问题、答案与隐藏 certificate 均绑定同一 Episode IR，但语义 RGB 复核尚未完成。" : "目录只加载摘要；打开条目后才读取对应 detail。绿色是模型可见面，蓝色是监督 target，深色 typed graph 与 certificate 是生成器证明，永不进入主输入。");
    const metrics = [
      [funnel.compiled_episodes, "episode IR"],
      [funnel.compiled_views, "ordered RGB views"],
      [funnel.compiled_questions, "geometry-backed QA"],
      [release.families.count, "consistency families"],
      [candidate ? Object.keys(release.coverage.programs).length : release.exports.benchmark_composition_records, candidate ? "typed programs" : "held-out composition"],
      [candidate ? funnel.featured_episodes : release.exports.episode_supervised_facts, candidate ? "featured candidates" : "paired train facts"],
    ];
    $("#release-metrics").innerHTML = metrics.map(([value, label]) => `<div><b>${escapeHtml(value)}</b><span>${escapeHtml(label)}</span></div>`).join("");
    const funnelSteps = candidate ? [
      [1, "config", "hash-bound policy"],
      [1, "IR", "hash-bound corpus"],
      [funnel.compiled_episodes, "episodes", "all compiled rows"],
      [funnel.compiled_views, "RGB", "media-closed views"],
      [funnel.compiled_questions, "facts", "passing certificates"],
      [funnel.featured_episodes, "featured", "deterministic preview"],
    ] : [
      [funnel.planned_jobs, "planned", "simulator jobs"],
      [funnel.strict_passed_bundles, "strict pass", "source closure"],
      [funnel.write_source_bundles, "write", "compiled sources"],
      [funnel.compiled_episodes, "IR", "episodes"],
      [funnel.compiled_questions, "facts", "executable QA"],
      [release.exports.episode_sft_records, "SFT", "episode records"],
    ];
    $("#release-funnel").innerHTML = funnelSteps.map(([value, stage, label], index) => `${index ? "<i>→</i>" : ""}<div><small>${escapeHtml(stage)}</small><b>${escapeHtml(value)}</b><span>${escapeHtml(label)}</span></div>`).join("");

    $("#release-splits").innerHTML = ["train", "val", "test"].map((split) => `<div><b>${escapeHtml(split.toUpperCase())}</b><span>${escapeHtml(release.splits.scenes[split])} scenes</span><strong>${escapeHtml(release.splits.episodes[split])} episodes</strong></div>`).join("");
    $("#release-families").innerHTML = Object.entries(release.families.shapes).map(([shape, count]) => `<div><b>${escapeHtml(count)}</b><span>${escapeHtml(shape.replaceAll("+", " ↔ "))}</span></div>`).join("") || `<div><b>0</b><span>no paired family in candidate</span></div>`;
    const contract = release.composition_contract;
    $("#release-composition").innerHTML = `<div class="release-atoms"><span>TRAIN</span>${contract.train_atoms.map((atom) => `<b>${escapeHtml(atom)}</b>`).join("")}</div><div class="release-heldout">${contract.heldout_program_ids.map((program) => `<code>${escapeHtml(program)}</code>`).join("")}<small>${candidate ? "benchmark export pending" : `${escapeHtml(contract.heldout_benchmark_records)} benchmark records`}</small></div>`;

    const verification = release.verification;
    text("#release-semantic-title", auditFailed ? `${semanticAudit.sample_size} 项独立 RGB 复核：${semanticAudit.failed_item_count} 项触发失败` : candidate ? "独立 RGB 语义复核尚未完成" : `${semanticAudit.sample_size}/${semanticAudit.sample_size} 项独立 RGB 复核闭合`);
    text("#release-semantic-boundary", semanticAudit.reviewer_label_zh);
    if (candidate) {
      if (auditFailed) {
        const semanticLabels = { pass: "pass", minor_issue: "minor", major_issue: "major", unreviewable: "unreviewable" };
        $("#release-semantic-counts").innerHTML = Object.entries(semanticLabels).map(([key, label]) => `<div class="${escapeHtml(key)}"><b>${escapeHtml(semanticAudit.counts[key])}</b><span>${escapeHtml(label)}</span></div>`).join("");
        text("#release-semantic-seed", semanticAudit.seed);
        text("#release-semantic-packet", semanticAudit.packet_id);
        $("#release-semantic-links").innerHTML = `<a href="${escapeHtml(semanticAudit.packet_url)}" target="_blank" rel="noopener">PACKET</a><a href="${escapeHtml(semanticAudit.result_url)}" target="_blank" rel="noopener">FAILED RESULT</a>`;
        text("#release-gate-title", `SEMANTIC RGB GATE FAILED · ${semanticAudit.failed_item_count} failed · ${semanticAudit.counts.minor_issue} minor`);
        $("#release-gates").innerHTML = `${verification.candidate_checks.map((gate) => `<span title="${escapeHtml(gate.name)}">✓ ${escapeHtml(gate.name.replaceAll("_", " "))}</span>`).join("")}<span class="failed">✕ FAILED · REWORK REQUIRED</span>`;
      } else {
        $("#release-semantic-counts").innerHTML = `<div><b>—</b><span>pending</span></div>`;
        text("#release-semantic-seed", "PENDING");
        text("#release-semantic-packet", "NOT ISSUED");
        $("#release-semantic-links").innerHTML = `<span>NO RESULT CLAIMED</span>`;
        text("#release-gate-title", `${verification.candidate_checks.length}/${verification.candidate_checks.length} automatic checks pass · semantic gate pending`);
        $("#release-gates").innerHTML = `${verification.candidate_checks.map((gate) => `<span title="${escapeHtml(gate.name)}">✓ ${escapeHtml(gate.name.replaceAll("_", " "))}</span>`).join("")}<span class="notice">PENDING · INDEPENDENT SEMANTIC RGB AUDIT</span>`;
      }
    } else {
      const semanticLabels = { pass: "pass", minor_issue: "minor", major_issue: "major", unreviewable: "unreviewable" };
      $("#release-semantic-counts").innerHTML = Object.entries(semanticLabels).map(([key, label]) => `<div><b>${escapeHtml(semanticAudit.counts[key])}</b><span>${escapeHtml(label)}</span></div>`).join("");
      text("#release-semantic-seed", semanticAudit.seed);
      text("#release-semantic-packet", semanticAudit.packet_id);
      $("#release-semantic-links").innerHTML = `<a href="${escapeHtml(semanticAudit.packet_url)}" target="_blank" rel="noopener">PACKET</a><a href="${escapeHtml(semanticAudit.result_url)}" target="_blank" rel="noopener">RESULT</a>`;
      text("#release-gate-title", `${verification.corpus_gates.filter((gate) => gate.passed).length}/${verification.corpus_gates.length} corpus gates + semantic RGB gate pass`);
      $("#release-gates").innerHTML = `${verification.corpus_gates.map((gate) => `<span title="${escapeHtml(gate.name)}">✓ ${escapeHtml(gate.name.replaceAll("_", " "))}</span>`).join("")}<span class="notice" title="${escapeHtml(verification.corpus_audit.warnings.join(" · "))}">AUDIT WARN · MODEL SCORE INTENTIONALLY ABSENT</span>`;
    }
    $("#release-artifacts").innerHTML = release.artifacts.map((artifact) => `<a href="${escapeHtml(artifact.url)}" target="_blank" rel="noopener"><span><b>${escapeHtml(artifact.name)}</b><small>${artifact.records === null || artifact.records === undefined ? "index" : `${escapeHtml(artifact.records)} records`} · ${formatBytes(artifact.bytes)}</small></span><code>${escapeHtml(shortHash(artifact.sha256))}</code></a>`).join("");
    text("#release-license", release.license_boundary);
  }

  function renderReleaseFilters() {
    const splitValues = ["all", "train", "val", "test"];
    const classValues = ["all", ...new Set(state.releaseData.episodes.map((episode) => episode.trajectory_class))];
    $("#release-split-filters").innerHTML = splitValues.map((value) => `<button type="button" class="${value === state.releaseSplit ? "active" : ""}" data-release-split="${escapeHtml(value)}">${escapeHtml(value.toUpperCase())}</button>`).join("");
    $("#release-class-filters").innerHTML = classValues.map((value) => `<button type="button" class="${value === state.releaseClass ? "active" : ""}" data-release-class="${escapeHtml(value)}">${escapeHtml(value.toUpperCase())}</button>`).join("");
    $$('[data-release-split]').forEach((button) => button.addEventListener("click", () => {
      state.releaseSplit = button.dataset.releaseSplit;
      renderReleaseFilters();
      renderReleaseEpisodeList(true);
    }));
    $$('[data-release-class]').forEach((button) => button.addEventListener("click", () => {
      state.releaseClass = button.dataset.releaseClass;
      renderReleaseFilters();
      renderReleaseEpisodeList(true);
    }));
  }

  function renderReleaseEpisodeList(selectFirst = false) {
    const episodes = filteredReleaseEpisodes();
    text("#release-browser-count", `${episodes.length} / ${state.releaseData.episodes.length} EPISODES`);
    $("#release-episode-list").innerHTML = episodes.length ? episodes.map((episode) => {
      const selected = episode.episode_id === state.releaseEpisodeId;
      const programs = Object.keys(episode.programs).slice(0, 2).map((item) => item.replace(".v1", "")).join(" · ");
      const featured = episode.featured ? "FEATURED · " : "";
      const reviewed = episode.fully_sample_reviewed_pass ? `${episode.sample_reviewed_fact_count}/${episode.sample_reviewed_fact_total} FACT PASS · ` : "";
      return `<button type="button" class="${selected ? "active" : ""}" data-release-episode="${escapeHtml(episode.episode_id)}"><img src="${escapeHtml(episode.thumbnail_url)}" alt="" loading="lazy" decoding="async"><span><em>${escapeHtml(featured)}${escapeHtml(reviewed)}${escapeHtml(episode.trajectory_class)} · ${escapeHtml(episode.split)}</em><b>${escapeHtml(episode.episode_id.slice(0, 13))}…</b><small>${escapeHtml(episode.view_count)} views · ${escapeHtml(episode.question_count)} QA</small><code>${escapeHtml(programs)}</code></span></button>`;
    }).join("") : `<p class="release-empty">没有符合筛选条件的 Episode。</p>`;
    $$('[data-release-episode]').forEach((button) => button.addEventListener("click", () => {
      const entry = state.releaseData.episodes.find((episode) => episode.episode_id === button.dataset.releaseEpisode);
      if (entry) void loadReleaseDetail(entry, { updateRoute: true });
    }));
    if (selectFirst && episodes.length && !episodes.some((episode) => episode.episode_id === state.releaseEpisodeId)) void loadReleaseDetail(episodes[0]);
  }

  async function loadReleaseDetail(entry, { updateRoute = false } = {}) {
    if (!entry || state.releaseLoadingId) return;
    state.releaseLoadingId = entry.episode_id;
    text("#release-detail-badge", releaseIsCandidate() ? "LOADING CANDIDATE DETAIL" : "LOADING VERIFIED DETAIL");
    try {
      const response = await fetch(entry.detail_url);
      if (!response.ok) throw new Error(`release detail fetch failed: ${response.status}`);
      const detail = await response.json();
      const candidate = releaseIsCandidate();
      const expectedSchema = candidate ? "epispace.web_candidate_detail.v1" : "epispace.web_release_detail.v1";
      if (detail.schema_version !== expectedSchema) throw new Error("unsupported release detail schema");
      if (detail.identity.episode_id !== entry.episode_id) throw new Error("release detail identity mismatch");
      const binding = candidate ? detail.candidate_binding : detail.release_binding;
      if (candidate && binding.pipeline_config_sha256 !== state.releaseData.provenance.pipeline_config_sha256) throw new Error("candidate detail is bound to a stale config");
      if (!candidate && binding.release_manifest_sha256 !== state.releaseData.provenance.release_manifest_sha256) throw new Error("release detail is bound to a stale manifest");
      if (binding.episode_ir_sha256 !== state.releaseData.provenance.episode_ir_sha256) throw new Error("release detail is bound to a stale episode IR");
      if (candidate && (detail.status !== "machine_verified_candidate" || detail.authority.semantic_visual_audit !== state.releaseData.authority.semantic_visual_audit || detail.authority.disposition !== state.releaseData.authority.disposition || detail.authority.formal_release !== false)) throw new Error("candidate detail overstates or disagrees with its authority");
      state.releaseDetail = detail;
      state.releaseEpisodeId = entry.episode_id;
      state.releaseViewIndex = 0;
      state.releaseQuestionIndex = 0;
      if (updateRoute) history.pushState(null, "", `#release/${encodeURIComponent(entry.episode_id)}`);
      renderReleaseEpisodeList();
      renderReleaseDetail();
    } catch (error) {
      text("#release-detail-badge", "DETAIL REJECTED");
      text("#release-detail-id", error.message || error);
    } finally {
      state.releaseLoadingId = null;
    }
  }

  function renderReleaseDetail() {
    const detail = state.releaseDetail;
    if (!detail) return;
    const observations = detail.model_visible.observations;
    const questions = detail.model_visible.questions;
    const view = observations[state.releaseViewIndex];
    const question = questions[state.releaseQuestionIndex];
    const target = detail.supervision_target.answers[state.releaseQuestionIndex];
    const proof = detail.compiler_proof.questions[state.releaseQuestionIndex];
    const candidateAuditFailed = releaseIsCandidate() && state.releaseData.authority.semantic_visual_audit === "failed";
    const catalogEntry = state.releaseData.episodes.find((item) => item.episode_id === detail.identity.episode_id);
    text("#release-detail-badge", candidateAuditFailed ? "CANDIDATE · AUDIT FAILED · REWORK" : releaseIsCandidate() ? "HASH-BOUND CANDIDATE · AUDIT PENDING" : "HASH-BOUND DETAIL");
    text("#release-detail-id", detail.identity.episode_id);
    $("#release-detail-meta").innerHTML = `<span>${escapeHtml(detail.identity.trajectory_class)}</span><span>${escapeHtml(detail.identity.split)}</span><span>${escapeHtml(observations.length)} views</span><span>${escapeHtml(questions.length)} QA</span>`;
    const reviewScope = $("#release-review-scope");
    if (catalogEntry?.fully_sample_reviewed_pass) {
      reviewScope.hidden = false;
      reviewScope.textContent = `本轮抽样逐事实复核 ${catalogEntry.sample_reviewed_fact_count}/${catalogEntry.sample_reviewed_fact_total} pass（仍非整库正式 release）`;
    } else {
      reviewScope.hidden = true;
      reviewScope.textContent = "";
    }
    $("#release-image").src = view.rgb_url;
    $("#release-image").alt = `${detail.identity.episode_id} ${view.view_id}`;
    text("#release-view-label", `${view.view_id} · ${view.role}`);
    $("#release-view-strip").innerHTML = observations.map((item, index) => `<button type="button" class="${index === state.releaseViewIndex ? "active" : ""}" data-release-view="${index}"><img src="${escapeHtml(item.rgb_url)}" alt=""><span>${escapeHtml(item.view_id.replace("view-", "v"))}</span></button>`).join("");
    $$('[data-release-view]').forEach((button) => button.addEventListener("click", () => {
      state.releaseViewIndex = Number(button.dataset.releaseView);
      renderReleaseDetail();
    }));
    $("#release-question-tabs").innerHTML = questions.map((item, index) => `<button type="button" class="${index === state.releaseQuestionIndex ? "active" : ""}" data-release-question="${index}"><b>Q${escapeHtml(item.order)}</b><span>${escapeHtml(detail.supervision_target.answers[index].task_type.replaceAll("_", " "))}</span></button>`).join("");
    $$('[data-release-question]').forEach((button) => button.addEventListener("click", () => {
      state.releaseQuestionIndex = Number(button.dataset.releaseQuestion);
      renderReleaseDetail();
    }));
    text("#release-question", question.question_zh);
    text("#release-answer", target.answer_zh);
    $("#release-target-meta").innerHTML = `<span>${escapeHtml(target.answer_status)}</span><span>${escapeHtml(target.family_variant)}</span><code>${escapeHtml(target.fact_id)}</code>`;
    text("#release-signature", proof.program.semantic_signature);
    $("#release-dag").innerHTML = proof.program.nodes.map((node, index) => `${index ? "<i>→</i>" : ""}<div><b>${escapeHtml(node.operation)}</b><span>${escapeHtml(node.output_type)}</span></div>`).join("");
    $("#release-certificate").innerHTML = proof.certificate.checks.map((check) => `<span>✓ ${escapeHtml(check.name.replaceAll("_", " "))}</span>`).join("");
  }

  async function bindRelease() {
    renderReleaseSummary();
    renderReleaseFilters();
    renderReleaseEpisodeList();
    $("#release-search").addEventListener("input", (event) => {
      state.releaseSearch = event.target.value;
      renderReleaseEpisodeList(true);
    });
    if (!await applyReleaseRoute()) {
      const featuredId = state.releaseData.featured_episode_ids?.[0];
      const initial = state.releaseData.episodes.find((episode) => episode.episode_id === featuredId) || state.releaseData.episodes[0];
      if (initial) await loadReleaseDetail(initial);
    }
  }

  async function applyReleaseRoute() {
    const match = location.hash.match(/^#release\/(.+)$/);
    if (!match || !state.releaseData) return false;
    const episodeId = decodeURIComponent(match[1]);
    const entry = state.releaseData.episodes.find((episode) => episode.episode_id === episodeId);
    if (!entry) return false;
    await loadReleaseDetail(entry);
    return true;
  }

  function bindSummary() {
    text("#metric-views", state.data.views.length);
    text("#metric-queries", state.data.queries.length);
    text("#metric-variants", state.data.family_variants.length);
    text("#metric-psnr", Number(state.data.source.loop_rgb_psnr_db).toFixed(2));
    $("#hero-image").src = state.data.views[0].media.rgb;
    const abilities = [
      ["G", "Grounding"], ["F", "Perspective"], ["B", "Reconstruction"],
      ["M", "Metric"], ["R", "Relation"], ["P/V", "Predict + Verify"],
    ];
    $("#ability-strip").innerHTML = abilities.map(([op, label]) => `<div><b>${escapeHtml(label)}</b><span>${escapeHtml(op)} · TYPED OP</span></div>`).join("");
  }

  function renderTrajectorySummary() {
    const summary = state.trajectoryData.summary;
    const metrics = [
      [summary.acquisition_job_count, "acquired episodes"],
      [summary.passed_episode_count, "machine passed"],
      [summary.needs_review_episode_count, "needs review"],
      [summary.failed_episode_count, "rejected attempts"],
      [`${(summary.strict_pass_rate * 100).toFixed(1)}%`, "strict pass rate"],
      [summary.certified_intervention_pair_count, "certified T9 pairs"],
    ];
    $("#trajectory-summary").innerHTML = metrics.map(([value, label]) => `<div><b>${escapeHtml(value)}</b><span>${escapeHtml(label)}</span></div>`).join("");
    $("#trajectory-coverage").innerHTML = state.trajectoryData.coverage.map((item) => `<article class="${escapeHtml(item.status)}"><b>${escapeHtml(item.class)}</b><span>${escapeHtml(item.name)}</span><small>${escapeHtml(item.status.replaceAll("_", " "))} · N=${escapeHtml(item.count)}</small><div class="coverage-counts"><i>${escapeHtml(item.passed)} pass</i>${item.needs_review ? `<i>${escapeHtml(item.needs_review)} review</i>` : ""}${item.failed ? `<i>${escapeHtml(item.failed)} reject</i>` : ""}</div></article>`).join("");
  }

  function renderCollectionMatrix() {
    const summary = state.trajectoryData.summary;
    const total = summary.acquisition_job_count;
    const statusItems = [
      ["passed", summary.passed_episode_count, "通过"],
      ["needs_review", summary.needs_review_episode_count, "待复核"],
      ["failed", summary.failed_episode_count, "拒绝"],
    ];
    $("#collection-status-bar").innerHTML = `<div class="status-track">${statusItems.map(([status, value]) => `<i class="${status}" style="--share:${value / total}"></i>`).join("")}</div><div class="status-legend">${statusItems.map(([status, value, label]) => `<span class="${status}"><i></i><b>${escapeHtml(value)}</b>${escapeHtml(label)}</span>`).join("")}</div>`;
    text("#collection-accounting-note", `${total} 条采集 episode 中，${summary.passed_episode_count} 条可进入高置信训练候选；${summary.needs_review_episode_count} 条保持隔离等待人工复核，${summary.failed_episode_count} 条仅保留失败 provenance。另有 ${summary.intervention_pair_count} 个 T9 干预对。`);
    const rows = state.trajectoryData.collections.map((item) => {
      const rate = item.total ? item.passed / item.total : 0;
      return `<div class="collection-row"><div><b>${escapeHtml(item.trajectory_class)}</b><span>${escapeHtml(item.name_zh)}</span></div><div class="mini-track"><i style="--rate:${rate}"></i></div><strong>${(rate * 100).toFixed(1)}%</strong><small><em>${escapeHtml(item.passed)} P</em><em>${escapeHtml(item.needs_review)} R</em><em>${escapeHtml(item.failed)} F</em></small></div>`;
    });
    $("#collection-matrix").innerHTML = `<header><span>CLASS</span><span>STRICT YIELD</span><span>RATE</span><span>P / R / F</span></header>${rows.join("")}`;
  }

  function renderCollectionJobs() {
    const jobs = state.trajectoryData.collection_jobs.filter((job) => (state.collectionClass === "all" || job.trajectory_class === state.collectionClass) && (state.collectionStatus === "all" || job.status === state.collectionStatus));
    const classes = ["all", ...new Set(state.trajectoryData.collection_jobs.map((job) => job.trajectory_class))];
    const statuses = ["all", "passed", "needs_review", "failed"];
    $("#collection-class-filters").innerHTML = classes.map((value) => `<button type="button" class="${state.collectionClass === value ? "active" : ""}" data-collection-class="${escapeHtml(value)}">${escapeHtml(value.toUpperCase())}</button>`).join("");
    $("#collection-status-filters").innerHTML = statuses.map((value) => `<button type="button" class="${state.collectionStatus === value ? "active" : ""}" data-collection-status="${escapeHtml(value)}">${escapeHtml(value.replaceAll("_", " ").toUpperCase())}</button>`).join("");
    const browsable = jobs.filter((job) => job.detail_url).length;
    const classLabel = state.collectionClass === "all" ? "全部类别" : `${state.collectionClass} · ${jobs[0]?.trajectory_class === state.collectionClass ? state.trajectoryData.collections.find((item) => item.trajectory_class === state.collectionClass)?.name_zh || "" : ""}`;
    text("#collection-browser-title", `${classLabel}场景画廊`);
    text("#collection-job-count", `${jobs.length} JOBS · ${browsable} 可播放`);
    $("#collection-job-list").innerHTML = jobs.map((job) => {
      const isOpen = state.trajectoryOverride?.trajectory_id === job.job_id;
      const preview = job.thumbnail_url
        ? `<img src="${escapeHtml(job.thumbnail_url)}" alt="${escapeHtml(job.scene)} 首帧" loading="lazy" decoding="async">`
        : `<div class="job-no-render"><b>NO BUNDLE</b><span>采集失败，仅保留 provenance</span></div>`;
      return `<article class="${escapeHtml(job.status)} ${isOpen ? "open" : ""}"><div class="job-thumb">${preview}<span class="job-class">${escapeHtml(job.trajectory_class)}</span><em>${escapeHtml(job.view_count)} views</em></div><div class="job-card-body"><div class="job-main"><b>${escapeHtml(job.scene)}</b><code>${escapeHtml(job.job_id)}</code></div><div class="job-meta"><span>${escapeHtml(job.split)}</span><span>${escapeHtml(job.classification)}</span></div><div class="job-outcome"><strong>${escapeHtml(job.status.replaceAll("_", " "))}</strong><small title="${escapeHtml(job.status_detail || job.reason_code)}">${escapeHtml(job.reason_code.replaceAll("_", " "))}</small></div><div class="job-links">${job.detail_url ? `<button type="button" data-open-job="${escapeHtml(job.job_id)}">${state.trajectoryLoadingId === job.job_id ? "LOADING…" : isOpen ? "OPENED" : "OPEN TRAJECTORY"}</button>` : `<span>无完整渲染</span>`}${job.preview_url ? `<a href="${escapeHtml(job.preview_url)}" target="_blank" rel="noopener">PREVIEW ↗</a>` : ""}${job.quality_url ? `<a href="${escapeHtml(job.quality_url)}" target="_blank" rel="noopener">GT JSON ↗</a>` : ""}</div></div></article>`;
    }).join("");
    $$("#collection-class-filters button").forEach((button) => button.addEventListener("click", () => {
      state.collectionClass = button.dataset.collectionClass;
      if (state.collectionClass !== "all") {
        const representativeIndex = state.trajectoryData.trajectories.findIndex((item) => item.trajectory_class === state.collectionClass);
        if (representativeIndex >= 0) {
          state.trajectoryIndex = representativeIndex;
          state.trajectoryOverride = null;
          state.trajectoryViewIndex = 0;
          renderTrajectory();
        }
      }
      const nextHash = state.collectionClass === "all" ? "#trajectories" : `#trajectories/${encodeURIComponent(state.collectionClass)}`;
      history.pushState(null, "", nextHash);
      renderCollectionJobs();
    }));
    $$("#collection-status-filters button").forEach((button) => button.addEventListener("click", () => { state.collectionStatus = button.dataset.collectionStatus; renderCollectionJobs(); }));
    $$('[data-open-job]').forEach((button) => button.addEventListener("click", () => {
      const job = state.trajectoryData.collection_jobs.find((item) => item.job_id === button.dataset.openJob);
      if (job) loadTrajectoryDetail(job);
    }));
  }

  async function loadTrajectoryDetail(job, { updateRoute = true, scroll = true } = {}) {
    if (!job?.detail_url || state.trajectoryLoadingId) return;
    if (state.trajectoryOverride?.trajectory_id === job.job_id) {
      if (scroll) $(".trajectory-workbench").scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }
    state.trajectoryLoadingId = job.job_id;
    renderCollectionJobs();
    try {
      const response = await fetch(job.detail_url);
      if (!response.ok) throw new Error(`trajectory detail fetch failed: ${response.status}`);
      state.trajectoryOverride = await response.json();
      state.trajectoryViewIndex = 0;
      state.trajectoryChannel = "rgb";
      state.collectionClass = job.trajectory_class;
      renderTrajectory();
      if (updateRoute) history.pushState(null, "", `#trajectories/${encodeURIComponent(job.trajectory_class)}/${encodeURIComponent(job.job_id)}`);
      if (scroll) $(".trajectory-workbench").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      text("#trajectory-status", "DETAIL LOAD FAILED");
      text("#trajectory-id", error.message || error);
    } finally {
      state.trajectoryLoadingId = null;
      renderCollectionJobs();
    }
  }

  function renderTrajectorySelector() {
    $("#trajectory-selector").innerHTML = `<p>REPRESENTATIVE ENTRIES</p>${state.trajectoryData.trajectories.map((trajectory, index) => `<button type="button" class="${!state.trajectoryOverride && index === state.trajectoryIndex ? "active" : ""}" data-trajectory-index="${index}"><span><strong>${escapeHtml(trajectory.trajectory_class)}</strong><em>${escapeHtml(trajectory.status.toUpperCase())}</em></span><b>${escapeHtml(trajectory.name_zh)}</b><small>${trajectory.view_count} views · seed ${trajectory.seed}</small></button>`).join("")}`;
    $$("#trajectory-selector button").forEach((button) => button.addEventListener("click", () => {
      state.trajectoryIndex = Number(button.dataset.trajectoryIndex);
      state.trajectoryOverride = null;
      state.trajectoryViewIndex = 0;
      history.pushState(null, "", "#trajectories");
      renderTrajectory();
    }));
  }

  function displayValue(value) {
    if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
    if (Array.isArray(value)) return value.some((item) => typeof item === "object") ? `${value.length} records` : value.join(" · ");
    if (typeof value === "boolean") return value ? "yes" : "no";
    if (typeof value === "object" && value !== null) return JSON.stringify(value);
    if (value === null || value === undefined || value === "") return "not recorded";
    return String(value);
  }

  function renderTrajectoryMethod() {
    const trajectory = currentTrajectory();
    text("#trajectory-name", `${trajectory.trajectory_class} · ${trajectory.name_zh}`);
    text("#trajectory-hypothesis", trajectory.capability_hypothesis);
    text("#trajectory-axes", trajectory.axes);
    const selection = trajectory.selection || {};
    const selectionRows = {
      T1: [["scene", trajectory.scene], ["view count", trajectory.view_count], ["path", "outbound + revisit + exact loop"], ["camera height", "1.5 m"], ["question policy", "coverage first; compile later"]],
      T3: [["station", selection.station_id], ["room", selection.room_instance], ["wall clearance", `${displayValue(selection.measured_wall_clearance_m)} m`], ["room-centroid offset", `${displayValue(selection.centroid_distance_m)} m`], ["policy", selection.selection_policy]],
      T4: [["focus", `${displayValue(selection.focus_category)} · ${displayValue(selection.focus_entity_id)}`], ["room", selection.room_instance], ["orbit radius", `${displayValue(selection.actual_radius_range_m || selection.desired_radius_m)} m`], ["arc coverage", `${displayValue(selection.arc_coverage_deg || 360)}° · ${selection.complete_orbit === false ? "partial" : "complete"}`], ["adaptive radius", displayValue(selection.adaptive_radius_used)], ["orientation QA", selection.orientation_questions_allowed], ["policy", selection.selection_policy]],
      T7: [["station", selection.station_id], ["room", selection.room_instance], ["height sequence", `${displayValue(selection.height_sequence_m)} m`], ["pitch sequence", `${displayValue(selection.pitch_sequence_deg)}°`], ["ceiling bottom", `${displayValue(selection.measured_ceiling_bottom_m)} m`]],
      T8: [["target", `${displayValue(selection.target_category)} · ${displayValue(selection.target_entity_id)}`], ["occluder", `${displayValue(selection.occluder_category)} · ${displayValue(selection.occluder_entity_id)}`], ["room", selection.room_instance], ["geodesic", `${displayValue(selection.selected_geodesic_distance_m)} m`], ["candidates", selection.candidate_count]],
      T10: [["anchor pairs", displayValue(selection.anchor_pairs)], ["candidates", selection.candidate_count], ["held out", selection.held_out], ["policy", selection.selection_policy], ["scene", trajectory.scene]],
    };
    const rows = selectionRows[trajectory.trajectory_class] || [["scene", trajectory.scene], ["view count", trajectory.view_count]];
    $("#trajectory-selection").innerHTML = rows.map(([label, value]) => `<div><span>${escapeHtml(label)}</span><b>${escapeHtml(displayValue(value))}</b></div>`).join("");

    let gates;
    if (trajectory.trajectory_class === "T3") {
      const gate = trajectory.quality.gates.T3;
      gates = [
        ["zero parallax", gate.checks.zero_parallax, `${gate.maximum_translational_baseline_m} m`],
        ["panorama entities ≥8", gate.checks.panorama_core_entities_at_least_8, gate.panorama_core_entity_count],
        ["rear-only entities ≥2", gate.checks.rear_only_core_entities_at_least_2, gate.rear_only_core_entity_count],
        ["depth question ban", gate.checks.depth_questions_forbidden, "enforced"],
        ["yaw/view alignment", gate.checks.yaw_sequence_matches_views, gate.yaw_sequence_deg.join("° · ") + "°"],
      ];
    } else if (trajectory.trajectory_class === "T1") {
      const closure = trajectory.quality.loop_closure || {};
      gates = [
        ["artifact integrity", trajectory.quality.integrity_status === "pass", trajectory.quality.integrity_status],
        ["visual audit", trajectory.quality.visual_status === "pass", trajectory.quality.visual_status],
        ["depth loop exact", closure.depth_exact, closure.depth_exact ? "exact" : "fail"],
        ["instance loop exact", closure.instance_exact, closure.instance_exact ? "exact" : "fail"],
        ["RGB loop PSNR", Number(closure.rgb_psnr_db) >= 30, `${closure.rgb_psnr_db} dB`],
      ];
    } else {
      const gate = trajectory.quality.gates[trajectory.trajectory_class];
      const checks = Object.entries(gate.checks).slice(0, 4).map(([label, pass]) => [label.replaceAll("_", " "), Boolean(pass), pass ? "pass" : "fail"]);
      const evidence = {
        T4: ["180-degree pair exists", (gate.opposite_view_index_pairs?.length || 0) >= 1, `${gate.opposite_view_index_pairs?.length || 0} pairs · ${gate.arc_coverage_deg || 360}°`],
        T7: ["cross-height anchors", gate.cross_height_anchor_instance_count >= 2, gate.cross_height_anchor_instance_count],
        T8: ["decisive reveal", Boolean(gate.decisive_view), gate.decisive_view],
        T10: ["visible target anchors", gate.anchors?.every((item) => item.facing_visible), `${gate.anchors?.length || 0} anchors`],
      }[trajectory.trajectory_class];
      gates = [...checks, evidence];
    }
    $("#trajectory-gates").innerHTML = gates.map(([label, pass, value]) => `<div><span>${escapeHtml(label)}</span><b class="${pass ? "pass" : "na"}">${pass ? "✓ " : "— "}${escapeHtml(value)}</b></div>`).join("");
  }

  function renderTrajectoryStrip() {
    const trajectory = currentTrajectory();
    $("#trajectory-strip").innerHTML = trajectory.views.map((view, index) => `<button type="button" class="${index === state.trajectoryViewIndex ? "active" : ""}" data-trajectory-view="${index}"><img src="${escapeHtml(view.media.rgb)}" alt=""><span>${escapeHtml(view.view_id.replace("view-", "v"))} · ${view.yaw_deg.toFixed(0)}°</span></button>`).join("");
    $$("#trajectory-strip button").forEach((button) => button.addEventListener("click", () => {
      state.trajectoryViewIndex = Number(button.dataset.trajectoryView);
      renderTrajectoryView();
    }));
  }

  function renderTrajectoryView() {
    const trajectory = currentTrajectory();
    const view = currentTrajectoryView();
    $("#trajectory-image").src = view.media[state.trajectoryChannel];
    $("#trajectory-image").alt = `${trajectory.trajectory_id} ${view.view_id} ${state.trajectoryChannel}`;
    text("#trajectory-status", `${trajectory.status.toUpperCase()} · ${trajectory.quality.integrity_status.toUpperCase()} · ${trajectory.quality.visual_status.toUpperCase()} · ${trajectory.quality.trajectory_status.toUpperCase()}`);
    text("#trajectory-id", `${trajectory.trajectory_id} · ${trajectory.scene}`);
    text("#trajectory-view-label", `${view.view_id} · ${view.role} · yaw ${view.yaw_deg.toFixed(1)}°`);
    text("#trajectory-channel-label", state.trajectoryChannel === "rgb" ? "MODEL VISIBLE" : "SUPERVISION ONLY");
    $("#trajectory-compass").innerHTML = `<i style="--yaw:${view.yaw_deg}deg"></i><span>YAW ${view.yaw_deg.toFixed(0)}°</span>`;
    const position = view.position_m.map((value) => Number(value).toFixed(2)).join(", ");
    const stats = [
      ["camera xyz", position],
      ["yaw", `${view.yaw_deg.toFixed(1)}°`],
      ["visible instances", view.visible_instance_count],
      ["core entities", view.visible_core_entity_count],
      ["valid depth", `${(view.valid_depth_fraction * 100).toFixed(1)}%`],
    ];
    $("#trajectory-view-stats").innerHTML = stats.map(([label, value]) => `<div>${escapeHtml(label)}<b>${escapeHtml(value)}</b></div>`).join("");
    $$("#trajectory-channels button").forEach((button) => button.classList.toggle("active", button.dataset.trajectoryChannel === state.trajectoryChannel));
    renderTrajectoryStrip();
  }

  function renderDerivedTrajectories() {
    const derived = state.trajectoryData.derived;
    const staticAudit = state.trajectoryData.static_sweep_t6_audit;
    const t2 = derived.T2;
    const t5 = derived.T5;
    const t2Missing = t2.coverage_gaps.length;
    const cards = [
      ["T2 · SPARSE SUBSAMPLE", "稀疏视图与顺序扰动", "从同一条 T1 抽取 2–5 帧；共视图连通决定可答，真正零共视才标 unknown。", [[t2.selected_subsets.length, "selected subsets"], [t2Missing, "honest gaps"]]],
      ["T5 · TWO-VIEW PAIR", "双视图注册难度阶梯", "每对保存共同核心实体数、相机平移、朝向差与 high/low/zero 档。", [[t5.selected_pairs.length, "selected pairs"], [t5.sources.length, "source classes"]]],
      ["T6 · BRIDGE AUDIT", "旧数据只筛查，不冒充认证", "46 个旧 T1 缺有序房间转移与门洞事件；因此即使 9 条覆盖≥3房间，也不签发 bridge_eligible。", [[staticAudit.three_room_coverage_count, "≥3-room screen"], [staticAudit.bridge_certified_count, "certified"]]],
    ];
    $("#derived-trajectory-grid").innerHTML = cards.map(([tag, title, copy, metrics]) => `<article><span>${escapeHtml(tag)}</span><h3>${escapeHtml(title)}</h3><p>${escapeHtml(copy)}</p><div class="derived-metrics">${metrics.map(([value, label]) => `<div><b>${escapeHtml(value)}</b><small>${escapeHtml(label)}</small></div>`).join("")}</div></article>`).join("");
    $("#trajectory-boundary").innerHTML = `<span>EVIDENCE BOUNDARY</span><p>${escapeHtml(staticAudit.conclusion)} T4/T7/T8/T10 已完成 184 个场景任务，但只有 machine-passed 条目进入训练候选；待复核与失败不会被总数掩盖。</p>`;
  }

  function renderTrajectory() {
    renderTrajectorySelector();
    renderTrajectoryMethod();
    renderTrajectoryView();
  }

  function bindTrajectory() {
    renderTrajectorySummary();
    renderCollectionMatrix();
    renderCollectionJobs();
    renderDerivedTrajectories();
    renderTrajectory();
  }

  async function applyTrajectoryRoute({ scroll = false } = {}) {
    const parts = location.hash.split("/").map((part) => decodeURIComponent(part));
    if (parts[0] !== "#trajectories") return;
    const trajectoryClass = parts[1];
    const jobId = parts[2];
    if (!trajectoryClass) {
      state.collectionClass = "all";
      renderCollectionJobs();
      return;
    }
    if (!state.trajectoryData.collection_jobs.some((job) => job.trajectory_class === trajectoryClass)) return;
    state.collectionClass = trajectoryClass;
    const representativeIndex = state.trajectoryData.trajectories.findIndex((item) => item.trajectory_class === trajectoryClass);
    if (representativeIndex >= 0) state.trajectoryIndex = representativeIndex;
    if (!jobId) {
      state.trajectoryOverride = null;
      state.trajectoryViewIndex = 0;
      renderCollectionJobs();
      renderTrajectory();
      return;
    }
    const job = state.trajectoryData.collection_jobs.find((item) => item.trajectory_class === trajectoryClass && item.job_id === jobId);
    if (job?.detail_url) await loadTrajectoryDetail(job, { updateRoute: false, scroll });
  }

  function renderTimeline() {
    const container = $("#view-timeline");
    container.innerHTML = "";
    state.data.views.forEach((view, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = index === state.viewIndex ? "active" : "";
      button.setAttribute("aria-label", `查看 ${view.view_id}`);
      button.innerHTML = `<img src="${escapeHtml(view.media.rgb)}" alt=""><span>${escapeHtml(view.view_id.replace("view-", "v"))} · S=${view.belief_entity_count}</span>`;
      button.addEventListener("click", () => { state.viewIndex = index; renderView(); });
      container.append(button);
    });
  }

  function renderStateCurve() {
    const values = state.data.views.map((view) => view.belief_entity_count);
    const width = 280, height = 95, pad = 9;
    const max = Math.max(...values);
    const points = values.map((value, index) => {
      const x = pad + index * (width - pad * 2) / (values.length - 1);
      const y = height - pad - value / max * (height - pad * 2);
      return [x, y];
    });
    const path = points.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
    const dots = points.map(([x, y], index) => `<circle cx="${x}" cy="${y}" r="${index === state.viewIndex ? 4 : 2.2}" fill="${index === state.viewIndex ? "#ff6b4a" : "#c9f36a"}"/>`).join("");
    $("#state-curve").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="belief growth"><path d="M${pad},${height-pad} H${width-pad}" stroke="rgba(255,255,255,.13)"/><path d="${path}" fill="none" stroke="#c9f36a" stroke-width="2"/>${dots}</svg>`;
  }

  function positionBbox() {
    const box = $("#bbox");
    const query = currentQuery();
    const view = currentView();
    const answer = query?.answer || {};
    const bbox = answer.bbox_norm_xyxy;
    const show = Boolean(bbox && view.view_id === "view-000" && ["rgb", "instance"].includes(state.channel));
    box.classList.toggle("visible", show);
    if (!show) return;
    const stage = $(".sensor-stage").getBoundingClientRect();
    const image = $("#sensor-image");
    const ratio = image.naturalWidth && image.naturalHeight ? image.naturalWidth / image.naturalHeight : 1;
    let contentWidth = stage.width;
    let contentHeight = contentWidth / ratio;
    if (contentHeight > stage.height) { contentHeight = stage.height; contentWidth = contentHeight * ratio; }
    const offsetX = (stage.width - contentWidth) / 2;
    const offsetY = (stage.height - contentHeight) / 2;
    box.style.left = `${offsetX + bbox[0] * contentWidth}px`;
    box.style.top = `${offsetY + bbox[1] * contentHeight}px`;
    box.style.width = `${(bbox[2] - bbox[0]) * contentWidth}px`;
    box.style.height = `${(bbox[3] - bbox[1]) * contentHeight}px`;
    text("#bbox-label", "fridge · GT instance");
  }

  function renderView() {
    const view = currentView();
    const image = $("#sensor-image");
    image.src = view.media[state.channel];
    image.alt = `${view.view_id} ${state.channel}`;
    image.onload = positionBbox;
    text("#view-label", `${view.view_id} · ${view.role}`);
    text("#channel-label", channelLabels[state.channel]);
    text("#belief-count", view.belief_entity_count);
    $("#view-stats").innerHTML = [
      ["camera xyz", view.world_from_camera.translation_m.map((v) => Number(v).toFixed(2)).join(", ")],
      ["visible now", view.visible_entity_count],
      ["belief total", view.belief_entity_count],
      ["frame", view.world_from_camera.child_frame],
    ].map(([label, value]) => `<div>${escapeHtml(label)}<b>${escapeHtml(value)}</b></div>`).join("");
    const delta = view.state_delta;
    $("#delta-cards").innerHTML = [[delta.added, "added"], [delta.reobserved, "reobserved"], [delta.left_current_view, "left view"]].map(([value, label]) => `<div><b>${value}</b><span>${label}</span></div>`).join("");
    $$("#channel-tabs button").forEach((button) => button.classList.toggle("active", button.dataset.channel === state.channel));
    renderTimeline();
    renderStateCurve();
    renderMap();
    requestAnimationFrame(positionBbox);
  }

  function mapPoint(x, y, bounds, width, height, pad) {
    const sx = pad + (x - bounds.minX) / Math.max(.1, bounds.maxX - bounds.minX) * (width - 2 * pad);
    const sy = height - pad - (y - bounds.minY) / Math.max(.1, bounds.maxY - bounds.minY) * (height - 2 * pad);
    return [sx, sy];
  }

  function renderMap() {
    const width = 760, height = 330, pad = 38;
    const cameras = state.data.views.map((view) => view.position_canonical_m);
    const entities = Object.values(state.data.entities);
    const all = [...cameras, ...entities.map((entity) => entity.position_canonical_m)];
    const bounds = { minX: Math.min(...all.map((p) => p[0])) - .25, maxX: Math.max(...all.map((p) => p[0])) + .25, minY: Math.min(...all.map((p) => p[1])) - .25, maxY: Math.max(...all.map((p) => p[1])) + .25 };
    const queryIds = new Set(currentQuery()?.evidence_entity_ids || []);
    const path = cameras.map((point, index) => {
      const [x, y] = mapPoint(point[0], point[1], bounds, width, height, pad);
      return `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    const cameraDots = cameras.map((point, index) => {
      const [x, y] = mapPoint(point[0], point[1], bounds, width, height, pad);
      const active = index === state.viewIndex;
      return `<circle cx="${x}" cy="${y}" r="${active ? 7 : 3.5}" fill="${active ? "#ff6b4a" : "#55d6be"}" stroke="#fffefa" stroke-width="${active ? 2 : 0}"/>`;
    }).join("");
    const entityDots = entities.map((entity, index) => {
      const [x, y] = mapPoint(entity.position_canonical_m[0], entity.position_canonical_m[1], bounds, width, height, pad);
      const selected = queryIds.has(entity.entity_id);
      return `<g><circle cx="${x}" cy="${y}" r="${selected ? 8 : 4}" fill="${selected ? (index % 2 ? "#ff6b4a" : "#c9f36a") : "#9aa5b3"}" opacity="${selected ? 1 : .55}"/><text x="${x + 9}" y="${y - 7}" font-size="${selected ? 11 : 8}" font-family="ui-monospace" fill="#122033">${escapeHtml(entity.label)}</text></g>`;
    }).join("");
    $("#scene-map").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="canonical scene map"><defs><pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M32 0H0V32" fill="none" stroke="rgba(18,32,51,.08)"/></pattern></defs><rect width="100%" height="100%" fill="url(#grid)"/><path d="${path}" fill="none" stroke="#55d6be" stroke-width="2" stroke-dasharray="5 5" opacity=".75"/>${cameraDots}${entityDots}<text x="20" y="23" font-size="9" font-family="ui-monospace" fill="#667488">+Y ↑ · +X → · meter</text></svg>`;
  }

  function filterKind(query) {
    if (query.status === "unknown") return "unknown";
    return query.curriculum_role.startsWith("held-out") ? "heldout" : "seen";
  }

  function filteredQueryIndices() {
    return state.data.queries.map((query, index) => ({ query, index })).filter(({ query }) => state.filter === "all" || filterKind(query) === state.filter);
  }

  function renderFilters() {
    const options = [["all", "全部"], ["seen", "Seen atoms"], ["heldout", "Held-out signatures"], ["unknown", "Unknown"]];
    $("#query-filters").innerHTML = options.map(([value, label]) => `<button type="button" class="${state.filter === value ? "active" : ""}" data-filter="${value}">${label}</button>`).join("");
    $$("#query-filters button").forEach((button) => button.addEventListener("click", () => {
      state.filter = button.dataset.filter;
      const first = filteredQueryIndices()[0];
      if (first) state.queryIndex = first.index;
      renderQueries();
    }));
  }

  function renderQueryList() {
    const list = $("#query-list");
    list.innerHTML = "";
    filteredQueryIndices().forEach(({ query, index }, position) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = index === state.queryIndex ? "active" : "";
      button.innerHTML = `<span class="query-index">${String(position + 1).padStart(2, "0")}</span><span><b>${escapeHtml(query.sense_nova_ability)}</b><small>${escapeHtml(query.curriculum_role)} · ${escapeHtml(query.query_id)}</small></span>`;
      button.addEventListener("click", () => {
        state.queryIndex = index;
        const evidence = query.evidence_view_ids[0];
        const viewIndex = state.data.views.findIndex((view) => view.view_id === evidence);
        if (viewIndex >= 0) state.viewIndex = viewIndex;
        renderQueries();
        renderView();
      });
      list.append(button);
    });
  }

  function answerForDisplay(answer) {
    if (answer && answer.relation && relationLabels[answer.relation]) return { ...answer, relation_label: relationLabels[answer.relation] };
    return answer;
  }

  function renderQueryDetail() {
    const query = currentQuery();
    text("#ability-label", query.sense_nova_ability);
    text("#role-label", query.curriculum_role.toUpperCase());
    text("#question-text", query.question);
    text("#program-signature", query.program_signature);
    text("#answer-json", JSON.stringify(answerForDisplay(query.answer), null, 2));
    $("#dag-flow").innerHTML = query.operation_graph.map((node, index) => `${index ? '<span class="dag-arrow">→</span>' : ""}<div class="dag-node"><b style="background:${opColors[node.op] || "#fff"}">${escapeHtml(node.op)}</b><span>${escapeHtml(node.node_id)}</span><small>${escapeHtml(node.output_type)}</small></div>`).join("");
    $("#certificate-list").innerHTML = query.certificate.map((check) => `<div><span>${escapeHtml(check.check)}</span><b class="${check.pass ? "pass" : "fail"}">${check.pass ? "PASS" : "FALSE"}</b></div>`).join("");
    renderTokens();
    renderMap();
    positionBbox();
  }

  function renderTokens() {
    const policy = currentQuery().loss_policy;
    const groups = [["input", policy.input_tokens], ["target", policy.target_spans], ["masked", policy.masked_out]];
    $("#token-box").innerHTML = groups.flatMap(([kind, values]) => values.map((value) => `<span class="token ${kind}">${kind.toUpperCase()} · ${escapeHtml(value)}</span>`)).join("");
  }

  function renderQueries() {
    renderFilters();
    renderQueryList();
    renderQueryDetail();
  }

  function dispositionLabel(value) {
    const labels = { train: "TRAIN · INCLUDED IN SFT", eval_only: "EVAL ONLY · EXCLUDED FROM SFT", partial_context_eval_only: "PARTIAL CONTEXT · SEPARATE EVAL" };
    return labels[value] || value;
  }

  function renderPilotList() {
    const container = $("#pilot-list");
    container.innerHTML = state.data.pilot_study.examples.map((item, index) => {
      const isTrain = item.disposition === "train";
      return `<button type="button" class="${index === state.pilotIndex ? "active" : ""}" data-pilot-index="${index}"><span><b>${escapeHtml(item.query_id)}</b><em class="${isTrain ? "" : "eval"}">${isTrain ? "TRAIN" : "EVAL"}</em></span><small>${escapeHtml(item.gt_lineage.answer_type)}</small></button>`;
    }).join("");
    $$("#pilot-list button").forEach((button) => button.addEventListener("click", () => {
      state.pilotIndex = Number(button.dataset.pilotIndex);
      renderPilot();
    }));
  }

  function renderPilotEvidence(item) {
    const byId = new Map(state.data.views.map((view) => [view.view_id, view]));
    $("#pilot-evidence").innerHTML = item.gt_lineage.evidence_view_ids.map((viewId) => {
      const view = byId.get(viewId);
      if (!view) return "";
      return `<figure><img src="${escapeHtml(view.media.rgb)}" alt="${escapeHtml(viewId)}"><figcaption>${escapeHtml(viewId)}</figcaption></figure>`;
    }).join("");
  }

  function renderPilot() {
    const pilot = state.data.pilot_study;
    const item = currentPilot();
    renderPilotList();
    const disposition = $("#pilot-disposition");
    disposition.textContent = dispositionLabel(item.disposition);
    disposition.classList.toggle("eval", item.disposition !== "train");
    text("#pilot-fingerprint", `GT SHA256 · ${item.gt_fingerprint}`);
    text("#pilot-question", item.safe_rewrite_api.natural_question_zh);
    renderPilotEvidence(item);
    $("#pilot-lineage").innerHTML = `<p><b>SOURCE</b><br>${escapeHtml(item.gt_lineage.source)}</p><p><b>EVIDENCE</b><br>${item.gt_lineage.evidence_view_ids.map(escapeHtml).join(" · ")}</p><p><b>CERTIFICATE</b><br>${item.gt_lineage.certificate.map((check) => `${escapeHtml(check.check)}=${check.pass ? "PASS" : "FALSE"}`).join(" · ")}</p>`;
    text("#pilot-api", JSON.stringify({ oracle_answer_visible: false, ...item.safe_rewrite_api.semantic_slots }, null, 2));
    $("#pilot-program").innerHTML = item.operation_graph.map((node, index) => `${index ? "<i>→</i>" : ""}<div class="mini-node"><b>${escapeHtml(node.op)} · ${escapeHtml(node.node_id)}</b><small>${escapeHtml(node.output_type)}</small></div>`).join("");
    text("#pilot-nodes", JSON.stringify(item.node_targets, null, 2));
    text("#pilot-answer", item.answer_surface_zh);
    $("#pilot-numeric").innerHTML = Object.entries(item.numeric_policy).map(([key, value]) => `<p><b>${escapeHtml(key.toUpperCase())}</b><br>${escapeHtml(value)}</p>`).join("");
    $("#pilot-validation").innerHTML = Object.entries(item.validation).filter(([, value]) => value).map(([key]) => `<span>✓ ${escapeHtml(key)}</span>`).join("");

    const record = pilot.qwen_sft_record;
    text("#sft-record-id", record.record_id);
    const userContent = record.messages.find((message) => message.role === "user").content;
    const userText = userContent.filter((block) => block.type === "text").map((block) => block.text).join("\n");
    const assistant = record.messages.find((message) => message.role === "assistant").content;
    text("#sft-user-preview", `${record.messages.length} messages · ${state.data.views.length} image blocks\n${userText.slice(-1700)}`);
    text("#sft-target-preview", `${assistant.slice(0, 2100)}\n\n… FULL TARGET IN JSONL (${assistant.length.toLocaleString()} chars)`);
    const blocked = pilot.blocked_candidate;
    text("#blocked-query", blocked.query_id);
    text("#blocked-reason", blocked.reason);
    text("#blocked-fix", blocked.required_fix);
  }

  function bindPilotSummary() {
    const pilot = state.data.pilot_study;
    const trainCount = pilot.examples.filter((item) => item.disposition === "train").length;
    const evalCount = pilot.examples.length - trainCount;
    const metrics = [
      [pilot.canonical_state_target.entity_count, "observable entities"],
      [trainCount, "train queries in JSONL"],
      [evalCount, "strictly excluded eval"],
      [state.data.views.length, "ordered RGB views"],
    ];
    $("#pilot-metrics").innerHTML = metrics.map(([value, label]) => `<div><b>${escapeHtml(value)}</b><span>${escapeHtml(label)}</span></div>`).join("");
  }

  function variantBehavior(variant) {
    const behaviors = {
      canonical: "提交完整 observable state；accepted query 由 certificate 执行。",
      legal_set_shuffle: "最终 state 与 set-query 答案保持不变；顺序敏感路径题不参与 shuffle。",
      delayed_kitchen_reveal: "厨房实体出现前必须 unknown；揭示后恢复 canonical 答案。",
      decisive_views_deleted: "电视/冰箱 cross-room query 必须 abstain，不允许从 SceneIR 猜测。",
      loop_revisit: "末帧回访首位姿；depth 与 mask exact，RGB PSNR ≥ 30 dB。",
    };
    return behaviors[variant.variant] || variant.invariant;
  }

  function renderFamily() {
    const variants = state.data.family_variants;
    $("#variant-list").innerHTML = variants.map((variant, index) => `<button type="button" data-index="${index}" class="${index === state.variantIndex ? "active" : ""}"><b>${escapeHtml(variant.variant)}</b><small>${variant.view_order.length} views · FAMILY LOCK</small></button>`).join("");
    $$("#variant-list button").forEach((button) => button.addEventListener("click", () => { state.variantIndex = Number(button.dataset.index); renderFamily(); }));
    const variant = variants[state.variantIndex];
    text("#variant-name", variant.variant);
    text("#variant-invariant", variant.invariant);
    text("#variant-behavior", variantBehavior(variant));
    $("#order-ribbon").innerHTML = variant.view_order.map((view) => `<span>${escapeHtml(view.replace("view-", "v"))}</span>`).join("");
  }

  function renderTraining() {
    const policy = state.data.channel_policy;
    const cards = [
      ["MODEL VISIBLE", "模型输入", policy.model_visible, "部署时真实可获得"],
      ["OPTIONAL AUX", "独立结构导出", policy.supervision, "用于 auxiliary 实验；不计入主 surface-answer loss"],
      ["COMPILER ONLY", "生成器专用", policy.oracle_only, "编译、验证与筛选；严格屏蔽"],
    ];
    $("#channel-policy").innerHTML = cards.map(([tag, title, values, note]) => `<article><small>${tag}</small><h3>${title}</h3><p>${values.map(escapeHtml).join(" · ")}</p><p>${note}</p></article>`).join("");
    $("#train-atoms").innerHTML = state.data.research_contract.train_atoms.map((atom) => `<span>${escapeHtml(atom)}</span>`).join("");
    $("#heldout-signatures").innerHTML = state.data.research_contract.held_out_signatures.map((signature) => `<code>${escapeHtml(signature)}</code>`).join("");
    renderTokens();
  }

  function renderBoundary() {
    const boundary = state.data.evidence_boundary;
    const cards = [["MEASURED", "真实测量", boundary.measured], ["DESIGNED", "研究设计", boundary.designed], ["NOT CLAIMED", "尚未宣称", boundary.not_claimed]];
    $("#boundary-grid").innerHTML = cards.map(([tag, title, items]) => `<article><small>${tag}</small><h3>${title}</h3><ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></article>`).join("");
  }

  function bindEvents() {
    $$("#channel-tabs button").forEach((button) => button.addEventListener("click", () => { state.channel = button.dataset.channel; renderView(); }));
    $$("#trajectory-channels button").forEach((button) => button.addEventListener("click", () => {
      if (!button.dataset.trajectoryChannel) return;
      state.trajectoryChannel = button.dataset.trajectoryChannel;
      renderTrajectoryView();
    }));
    $("#trajectory-copy-link").addEventListener("click", async () => {
      const button = $("#trajectory-copy-link");
      try {
        if (navigator.clipboard) {
          await navigator.clipboard.writeText(location.href);
        } else {
          const temporary = document.createElement("textarea");
          temporary.value = location.href;
          document.body.append(temporary);
          temporary.select();
          document.execCommand("copy");
          temporary.remove();
        }
        button.textContent = "COPIED";
        setTimeout(() => { button.textContent = "COPY LINK"; }, 1400);
      } catch {
        button.textContent = "COPY FAILED";
      }
    });
    $("#expand-image").addEventListener("click", () => {
      $("#dialog-image").src = currentView().media[state.channel];
      text("#dialog-caption", `${currentView().view_id} · ${channelLabels[state.channel]}`);
      $("#image-dialog").showModal();
    });
    $("#close-dialog").addEventListener("click", () => $("#image-dialog").close());
    window.addEventListener("resize", positionBbox);
    window.addEventListener("scroll", () => {
      const maximum = document.documentElement.scrollHeight - innerHeight;
      $("#progress").style.width = `${maximum > 0 ? scrollY / maximum * 100 : 0}%`;
    }, { passive: true });
    window.addEventListener("hashchange", () => {
      void applyTrajectoryRoute();
      void applyReleaseRoute();
    });
  }

  async function loadReleaseCatalog() {
    const verified = await fetch("data/release_catalog.v1.json");
    if (verified.ok) {
      const catalog = await verified.json();
      if (catalog.schema_version !== "epispace.web_release_catalog.v1" || catalog.status !== "verified") throw new Error("release catalog is not verified");
      return catalog;
    }
    if (verified.status !== 404) throw new Error(`verified release fetch failed: ${verified.status}`);
    const candidate = await fetch("data/candidate_catalog.v1.json");
    if (!candidate.ok) throw new Error(`candidate catalog fetch failed after release 404: ${candidate.status}`);
    const catalog = await candidate.json();
    if (catalog.schema_version !== "epispace.web_candidate_catalog.v1" || catalog.status !== "machine_verified_candidate") throw new Error("candidate catalog has unsupported authority");
    const semanticState = catalog.authority?.semantic_visual_audit;
    if (!["pending", "failed", "passed"].includes(semanticState) || catalog.authority?.formal_release !== false) throw new Error("candidate catalog overstates its authority");
    if (semanticState === "failed" && catalog.authority?.disposition !== "rework_required") throw new Error("failed candidate is missing rework disposition");
    return catalog;
  }

  function renderReleaseUnavailable(error) {
    $("#release")?.classList.add("candidate-mode");
    text("#release-status", "CATALOG UNAVAILABLE");
    text("#release-dataset-id", "其他页面仍可正常浏览");
    text("#release-authority-note", "NO VERIFIED RELEASE OR MACHINE CANDIDATE LOADED");
    text("#release-description", `数据资产目录暂不可用：${error.message || error}`);
    text("#release-browser-title", "Episode 目录等待生成");
    text("#release-browser-count", "0 EPISODES");
    $("#release-episode-list").innerHTML = `<p class="release-empty">候选或正式 catalog 生成后会自动出现在这里；这不会影响轨迹库、Episode walkthrough 与训练说明。</p>`;
    text("#release-detail-badge", "NO CATALOG");
    text("#release-detail-id", error.message || error);
  }

  async function init() {
    try {
      const [episodeResponse, trajectoryResponse] = await Promise.all([
        fetch("data/generalization_episode.v1.json"),
        fetch("data/trajectory_catalog.v1.json"),
      ]);
      if (!episodeResponse.ok) throw new Error(`episode fetch failed: ${episodeResponse.status}`);
      if (!trajectoryResponse.ok) throw new Error(`trajectory fetch failed: ${trajectoryResponse.status}`);
      state.data = await episodeResponse.json();
      state.trajectoryData = await trajectoryResponse.json();
      bindSummary();
      bindTrajectory();
      await applyTrajectoryRoute();
      bindCompiledControls();
      try {
        await loadCompiledCorpus();
      } catch (compiledDialogueError) {
        renderCompiledUnavailable(compiledDialogueError);
      }
      bindSceneDialogueControls();
      try {
        await loadSceneDialogueCorpus();
      } catch (sceneDialogueError) {
        renderSceneDialogueUnavailable(sceneDialogueError);
      }
      renderTraining();
      renderBoundary();
      bindEvents();
    } catch (error) {
      document.body.innerHTML = `<main style="padding:40px;font-family:sans-serif"><h1>Episode 加载失败</h1><pre>${escapeHtml(error.stack || error)}</pre><p>请从 /data/shichao/data/dataV100/code 目录启动 HTTP server。</p></main>`;
    }
  }

  init();
})();
