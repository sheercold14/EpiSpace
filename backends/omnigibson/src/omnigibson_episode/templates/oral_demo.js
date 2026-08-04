(() => {
  "use strict";

  const data = JSON.parse(document.querySelector("#episode-data").textContent);
  const state = {
    viewIndex: 0,
    channel: "rgb",
    relation: "all",
    filteredQueries: data.queries,
    queryIndex: 0,
  };

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const text = (selector, value) => { $(selector).textContent = String(value); };
  const escapeXml = (value) => String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const pad = (value) => String(value).padStart(2, "0");
  const short = (value, length = 12) => String(value).slice(0, length);

  const relationLabels = {
    left_of: "左侧", right_of: "右侧", in_front_of: "前方",
    behind: "后方", above: "上方", below: "下方",
  };
  const channelLabels = {
    rgb: "RGB · model visible",
    depth: "metric depth · supervision",
    instance: "object instance · supervision",
    semantic: "semantic category · derived",
  };
  const roleLabels = {
    initial: "initial", outbound: "outbound", return: "return", loop_closure: "loop closure",
  };
  const opColors = {
    G: "#55d6be", F: "#7198ff", B: "#c9f36a", M: "#ffd166",
    R: "#ff9e86", P: "#b58cff", V: "#ff6b4a",
  };
  const opNames = {
    G: "Ground", F: "Frame", B: "Belief", M: "Metric",
    R: "Relation", P: "Perspective", V: "Verify",
  };
  const checkLabels = {
    subject_observed: "subject observed",
    reference_observed: "reference observed",
    axis_decision_margin_m: "axis margin",
    typed_operation_graph: "typed graph valid",
  };

  function bindStaticData() {
    $$('[data-bind="view-count"]').forEach((node) => { node.textContent = data.metrics.view_count; });
    $$('[data-bind="entity-count"]').forEach((node) => { node.textContent = data.metrics.entity_count; });
    $$('[data-bind="query-count"]').forEach((node) => { node.textContent = data.metrics.query_count; });
    $$('[data-bind="loop-psnr"]').forEach((node) => { node.textContent = Number(data.metrics.loop_rgb_psnr_db).toFixed(2); });
    $$('[data-bind="episode-id"]').forEach((node) => { node.textContent = data.episode.id; });
    $$('[data-bind="scene-source"]').forEach((node) => { node.textContent = `${data.episode.scene_name} / ${data.episode.source_version}`; });
    $$('[data-bind="recipe-id"]').forEach((node) => { node.textContent = data.episode.recipe_id; });
    $$('[data-bind="source-digest"]').forEach((node) => { node.textContent = data.provenance.source_digest; });
    $$('[data-bind="license-id"]').forEach((node) => { node.textContent = data.provenance.license_identifier; });
    $("#hero-image").src = data.views[0].media.rgb;
    text("#capabilities", data.capabilities.join(" · "));
    text("#capability-gaps", data.capability_gaps.map((gap) => `${gap.capability}: ${gap.reason}`).join(" · "));
  }

  function renderTimeline() {
    const root = $("#view-timeline");
    root.innerHTML = "";
    data.views.forEach((view, index) => {
      const button = document.createElement("button");
      const returning = view.role === "return" || view.role === "loop_closure";
      button.className = `view-step ${returning ? "return" : ""} ${index === state.viewIndex ? "active" : ""}`;
      button.type = "button";
      button.setAttribute("role", "option");
      button.setAttribute("aria-selected", index === state.viewIndex ? "true" : "false");
      button.title = `${view.id} · ${roleLabels[view.role] || view.role}`;
      button.innerHTML = `<img src="${escapeXml(view.media.thumbnail)}" alt="${escapeXml(view.id)}"><i></i><span>${pad(view.step)} · +${view.added_count}</span>`;
      button.addEventListener("click", () => selectView(index));
      root.append(button);
    });
  }

  function renderStateChart() {
    const width = 310;
    const height = 110;
    const padding = 10;
    const maxValue = Math.max(...data.views.map((view) => view.belief_entity_count), 1);
    const points = data.views.map((view, index) => {
      const x = padding + index * (width - padding * 2) / (data.views.length - 1);
      const y = height - padding - view.belief_entity_count / maxValue * (height - padding * 2);
      return [x, y];
    });
    const area = `M ${points[0][0]} ${height - padding} L ${points.map(([x, y]) => `${x} ${y}`).join(" L ")} L ${points.at(-1)[0]} ${height - padding} Z`;
    const line = `M ${points.map(([x, y]) => `${x} ${y}`).join(" L ")}`;
    const circles = points.map(([x, y], index) => `<circle cx="${x}" cy="${y}" r="${index === state.viewIndex ? 4.5 : 2.3}" fill="${index === state.viewIndex ? "#ff6b4a" : "#c9f36a"}"/>`).join("");
    $("#state-chart").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="belief entities accumulated by view"><path d="${area}" fill="rgba(201,243,106,.08)"/><path d="${line}" fill="none" stroke="#c9f36a" stroke-width="2"/>${circles}<text x="${padding}" y="${height - 1}" fill="rgba(255,255,255,.35)" font-size="7" font-family="monospace">STATE GROWTH / VIEW ORDER</text></svg>`;
  }

  function renderView() {
    const view = data.views[state.viewIndex];
    $("#sensor-image").src = view.media[state.channel];
    $("#sensor-image").alt = `${view.id} ${channelLabels[state.channel]}`;
    text("#sensor-view-id", `${view.id} · ${roleLabels[view.role] || view.role}`);
    text("#sensor-channel-label", channelLabels[state.channel]);
    text("#belief-count", view.belief_entity_count);
    text("#added-count", view.added_count);
    text("#reobserved-count", view.reobserved_count);
    text("#visible-count", view.visible_count);
    $("#view-facts").innerHTML = [
      ["camera xyz / m", view.position_m.map((value) => value.toFixed(2)).join(" · ")],
      ["path distance", `${view.cumulative_distance_m.toFixed(2)} m`],
      ["valid depth", `${(view.valid_depth_fraction * 100).toFixed(2)}%`],
      ["sharpness", `${Number(view.sharpness).toFixed(1)} LapVar`],
    ].map(([label, value]) => `<div>${label}<b>${value}</b></div>`).join("");
    $$(".channel-tabs button").forEach((button) => button.classList.toggle("active", button.dataset.channel === state.channel));
    renderTimeline();
    renderStateChart();
    renderSceneMap();
  }

  function selectView(index) {
    state.viewIndex = (index + data.views.length) % data.views.length;
    renderView();
  }

  function currentQuery() {
    return state.filteredQueries[state.queryIndex] || data.queries[0];
  }

  function mapProjector(width, height, padding) {
    const coordinates = [
      ...data.entities.map((entity) => entity.position_m.slice(0, 2)),
      ...data.views.map((view) => view.position_m.slice(0, 2)),
    ];
    const xs = coordinates.map((point) => point[0]);
    const ys = coordinates.map((point) => point[1]);
    const bounds = {
      minX: Math.min(...xs), maxX: Math.max(...xs),
      minY: Math.min(...ys), maxY: Math.max(...ys),
    };
    const xSpan = Math.max(bounds.maxX - bounds.minX, 1);
    const ySpan = Math.max(bounds.maxY - bounds.minY, 1);
    const scale = Math.min((width - padding * 2) / xSpan, (height - padding * 2) / ySpan);
    const usedWidth = xSpan * scale;
    const usedHeight = ySpan * scale;
    const xOffset = (width - usedWidth) / 2;
    const yOffset = (height - usedHeight) / 2;
    return ([x, y]) => [
      xOffset + (x - bounds.minX) * scale,
      height - yOffset - (y - bounds.minY) * scale,
    ];
  }

  function renderSceneMap() {
    const width = 900;
    const height = 510;
    const project = mapProjector(width, height, 48);
    const currentView = data.views[state.viewIndex];
    const visible = new Set(currentView.visible_entity_ids);
    const query = currentQuery();
    const subjectId = query?.subject?.id;
    const referenceId = query?.reference?.id;
    const grid = Array.from({ length: 9 }, (_, index) => {
      const x = 50 + index * 100;
      return `<path d="M ${x} 0 V ${height} M 0 ${x} H ${width}" stroke="rgba(255,255,255,.035)"/>`;
    }).join("");
    const pathPoints = data.views.map((view) => project(view.position_m)).map(([x, y]) => `${x},${y}`).join(" ");
    const objects = data.entities.map((entity) => {
      const [x, y] = project(entity.position_m);
      const isSubject = entity.id === subjectId;
      const isReference = entity.id === referenceId;
      const isVisible = visible.has(entity.id);
      const fill = isSubject ? "#ff6b4a" : isReference ? "#7198ff" : isVisible ? "#55d6be" : "#536477";
      const radius = isSubject || isReference ? 7 : isVisible ? 3.4 : 2.2;
      const opacity = isSubject || isReference ? 1 : isVisible ? .72 : .28;
      return `<circle cx="${x}" cy="${y}" r="${radius}" fill="${fill}" opacity="${opacity}"><title>${escapeXml(entity.name)} · ${escapeXml(entity.position_m.join(", "))}</title></circle>`;
    }).join("");
    const cameras = data.views.map((view, index) => {
      const [x, y] = project(view.position_m);
      const active = index === state.viewIndex;
      return `<circle cx="${x}" cy="${y}" r="${active ? 7 : 3}" fill="${active ? "#c9f36a" : "#efffc9"}" opacity="${active ? 1 : .5}"/><text x="${x + 7}" y="${y - 7}" fill="${active ? "#c9f36a" : "rgba(255,255,255,.28)"}" font-size="8" font-family="monospace">${active ? escapeXml(view.id) : pad(view.step)}</text>`;
    }).join("");
    $("#scene-map").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="canonical top-down scene map">${grid}<polyline points="${pathPoints}" fill="none" stroke="#c9f36a" stroke-width="2" stroke-dasharray="5 5" opacity=".52"/>${objects}${cameras}<text x="24" y="32" fill="rgba(255,255,255,.42)" font-size="9" font-family="monospace">TOP-DOWN · +X RIGHT / +Y FORWARD</text></svg>`;
  }

  function renderRelationFilters() {
    const relations = Object.keys(data.relation_counts);
    const options = ["all", ...relations];
    $("#relation-filters").innerHTML = options.map((relation) => {
      const count = relation === "all" ? data.queries.length : data.relation_counts[relation];
      const label = relation === "all" ? "全部" : relationLabels[relation] || relation;
      return `<button type="button" data-relation="${escapeXml(relation)}" class="${relation === state.relation ? "active" : ""}">${label} · ${count}</button>`;
    }).join("");
    $$("#relation-filters button").forEach((button) => button.addEventListener("click", () => {
      state.relation = button.dataset.relation;
      state.filteredQueries = state.relation === "all" ? data.queries : data.queries.filter((query) => query.relation === state.relation);
      state.queryIndex = 0;
      renderRelationFilters();
      renderQuery(true);
    }));
  }

  function renderCertificate(query) {
    $("#certificate-checks").innerHTML = query.certificate.checks.map((check) => {
      const measured = typeof check.measured_value === "number" ? `${Number(check.measured_value).toFixed(3)} m` : "true";
      return `<div class="certificate-check"><span>${escapeXml(checkLabels[check.name] || check.name)}</span><b>✓ ${measured}</b></div>`;
    }).join("");
    text("#margin-value", `${query.margin_m.toFixed(3)} m`);
    const maximum = Math.max(1.2, query.margin_m * 1.15);
    $("#margin-bar").style.width = `${clamp(query.margin_m / maximum * 100, 0, 100)}%`;
    $("#margin-threshold").style.left = `${clamp(.4 / maximum * 100, 0, 100)}%`;
  }

  function graphLevels(nodes) {
    const byId = new Map(nodes.map((node) => [node.id, node]));
    const levels = new Map();
    function level(node) {
      if (levels.has(node.id)) return levels.get(node.id);
      const dependencies = node.inputs.filter((input) => byId.has(input));
      const value = dependencies.length ? 1 + Math.max(...dependencies.map((id) => level(byId.get(id)))) : 0;
      levels.set(node.id, value);
      return value;
    }
    nodes.forEach(level);
    return levels;
  }

  function renderDag(query) {
    const nodes = query.nodes;
    const levels = graphLevels(nodes);
    const grouped = new Map();
    nodes.forEach((node) => {
      const value = levels.get(node.id);
      if (!grouped.has(value)) grouped.set(value, []);
      grouped.get(value).push(node);
    });
    const width = 1100;
    const height = 430;
    const nodeWidth = 154;
    const nodeHeight = 76;
    const maxLevel = Math.max(...levels.values(), 1);
    const positions = new Map();
    [...grouped.entries()].forEach(([level, group]) => {
      group.forEach((node, index) => {
        const x = 42 + level * (width - nodeWidth - 84) / maxLevel;
        const spacing = height / (group.length + 1);
        const y = spacing * (index + 1) - nodeHeight / 2;
        positions.set(node.id, { x, y });
      });
    });
    const edges = nodes.flatMap((node) => node.inputs.filter((input) => positions.has(input)).map((input) => {
      const source = positions.get(input);
      const target = positions.get(node.id);
      const sx = source.x + nodeWidth;
      const sy = source.y + nodeHeight / 2;
      const tx = target.x;
      const ty = target.y + nodeHeight / 2;
      const bend = (sx + tx) / 2;
      return `<path d="M ${sx} ${sy} C ${bend} ${sy}, ${bend} ${ty}, ${tx} ${ty}" fill="none" stroke="rgba(255,255,255,.22)" stroke-width="1.5" marker-end="url(#arrow)"/>`;
    })).join("");
    const cards = nodes.map((node) => {
      const { x, y } = positions.get(node.id);
      const color = opColors[node.operation] || "#fff";
      const external = node.inputs.filter((input) => !positions.has(input));
      return `<g><rect x="${x}" y="${y}" width="${nodeWidth}" height="${nodeHeight}" rx="10" fill="#112337" stroke="${color}" stroke-width="${node.id === query.answer_node ? 2.5 : 1}"/><rect x="${x}" y="${y}" width="7" height="${nodeHeight}" rx="4" fill="${color}"/><text x="${x + 20}" y="${y + 24}" fill="${color}" font-size="11" font-weight="800" font-family="monospace">${escapeXml(node.operation)} · ${escapeXml(node.operation_name)}</text><text x="${x + 20}" y="${y + 43}" fill="#fff" font-size="9" font-family="sans-serif">${escapeXml(short(node.id, 21))}</text><text x="${x + 20}" y="${y + 61}" fill="rgba(255,255,255,.4)" font-size="8" font-family="monospace">→ ${escapeXml(node.output_type)}${external.length ? ` · +${external.length} ext` : ""}</text></g>`;
    }).join("");
    $("#dag-canvas").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="typed operation graph"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(255,255,255,.34)"/></marker></defs>${edges}${cards}</svg>`;
    const signature = [...new Set(nodes.map((node) => node.operation))];
    text("#dag-signature", signature.join(" → "));
    $("#op-legend").innerHTML = signature.map((operation) => `<span><i style="background:${opColors[operation]}"></i>${operation} · ${opNames[operation]} · ${data.operation_counts[operation] || 0} nodes / episode</span>`).join("");
  }

  function renderQuery(syncView = false) {
    const query = currentQuery();
    if (!query) return;
    if (syncView) {
      const evidenceIndex = data.views.findIndex((view) => view.id === query.evidence_view_ids[0]);
      if (evidenceIndex >= 0) state.viewIndex = evidenceIndex;
    }
    text("#query-position", pad(state.queryIndex + 1));
    text("#query-id", `QUERY ${pad(query.index + 1)} · ${query.short_id}`);
    text("#formal-question", query.formal_question);
    text("#subject-name", query.subject.name);
    text("#reference-name", query.reference.name);
    text("#subject-position", `[${query.subject.position_m.join(", ")}] m`);
    text("#reference-position", `[${query.reference.position_m.join(", ")}] m`);
    text("#query-answer", query.answer_label);
    text("#query-answer-raw", query.answer);
    text("#evidence-view", query.evidence_view_ids.join(" + "));
    $("#evidence-view").onclick = () => {
      const index = data.views.findIndex((view) => view.id === query.evidence_view_ids[0]);
      if (index >= 0) selectView(index);
      $("#episode").scrollIntoView({ behavior: "smooth" });
    };
    renderCertificate(query);
    renderDag(query);
    renderView();
  }

  function moveQuery(delta) {
    state.queryIndex = (state.queryIndex + delta + state.filteredQueries.length) % state.filteredQueries.length;
    renderQuery(true);
  }

  function renderChannelPolicy() {
    const policies = [
      ["MODEL VISIBLE", "输入通道", data.episode.model_visible, "模型真正看到的证据"],
      ["DENSE SUPERVISION", "训练真值", data.episode.supervision, "提供状态与操作梯度"],
      ["ORACLE ONLY", "生成器专用", data.episode.oracle_only, "训练输入中严格隔离"],
    ];
    $("#channel-policy").innerHTML = policies.map(([tag, title, values, note]) => `<article class="policy-card"><small>${tag}</small><b>${title}</b><p>${values.join(" · ")}</p><p>${note}</p></article>`).join("");
  }

  function wireInteractions() {
    $$(".channel-tabs button").forEach((button) => button.addEventListener("click", () => {
      state.channel = button.dataset.channel;
      renderView();
    }));
    $("#query-prev").addEventListener("click", () => moveQuery(-1));
    $("#query-next").addEventListener("click", () => moveQuery(1));
    const dialog = $("#image-dialog");
    $("#expand-image").addEventListener("click", () => {
      const view = data.views[state.viewIndex];
      $("#dialog-image").src = view.media[state.channel];
      text("#dialog-caption", `${view.id} · ${channelLabels[state.channel]} · ${data.episode.resolution}`);
      dialog.showModal();
    });
    $("#close-dialog").addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && dialog.open) dialog.close();
      if (event.altKey && event.key === "ArrowLeft") selectView(state.viewIndex - 1);
      if (event.altKey && event.key === "ArrowRight") selectView(state.viewIndex + 1);
    });
    window.addEventListener("scroll", () => {
      const maximum = document.documentElement.scrollHeight - window.innerHeight;
      $("#reading-progress").style.width = `${maximum > 0 ? window.scrollY / maximum * 100 : 0}%`;
    }, { passive: true });
  }

  bindStaticData();
  renderRelationFilters();
  renderChannelPolicy();
  wireInteractions();
  renderQuery(false);
})();
