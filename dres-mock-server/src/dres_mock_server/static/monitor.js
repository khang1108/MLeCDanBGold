"use strict";

const state = {
  snapshot: null,
  selectedRecordId: null,
  filter: "ALL",
  paused: false,
  inFlight: false,
  refreshPending: false,
  controller: null,
  controlsDirty: false,
};

const elements = {
  connection: document.getElementById("connection-state"),
  mockState: document.getElementById("mock-state"),
  lastRefresh: document.getElementById("last-refresh"),
  evaluationName: document.getElementById("evaluation-name"),
  evaluationId: document.getElementById("evaluation-id"),
  taskName: document.getElementById("task-name"),
  taskMeta: document.getElementById("task-meta"),
  sessionCount: document.getElementById("session-count"),
  submissionCount: document.getElementById("submission-count"),
  resultLogCount: document.getElementById("result-log-count"),
  liveMessage: document.getElementById("live-message"),
  recordList: document.getElementById("record-list"),
  emptyJournal: document.getElementById("empty-journal"),
  journalCount: document.getElementById("journal-count-label"),
  selectedRecordLabel: document.getElementById("selected-record-label"),
  detailEmpty: document.getElementById("detail-empty"),
  detailContent: document.getElementById("detail-content"),
  detailMetadata: document.getElementById("detail-metadata"),
  recordJson: document.getElementById("record-json"),
  refreshButton: document.getElementById("refresh-button"),
  pauseButton: document.getElementById("pause-button"),
  resetButton: document.getElementById("reset-button"),
};

function allRecords(snapshot) {
  return [...snapshot.submissions, ...snapshot.result_logs]
    .sort((left, right) => right.record_id - left.record_id);
}

function filteredRecords(snapshot) {
  const records = allRecords(snapshot);
  return state.filter === "ALL"
    ? records
    : records.filter((record) => record.kind === state.filter);
}

function setConnection(online, message) {
  elements.connection.textContent = online ? "Connected" : "Offline";
  elements.connection.classList.toggle("is-online", online);
  elements.connection.classList.toggle("is-offline", !online);
  elements.mockState.textContent = online ? "Online" : "Offline";
  elements.liveMessage.textContent = message;
  elements.liveMessage.classList.toggle("is-error", !online);
}

function formatTimestamp(timestamp) {
  if (!Number.isFinite(timestamp)) return "unknown time";
  return new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function countFor(record) {
  if (record.kind === "RESULT_LOG") {
    return `${Array.isArray(record.payload.results) ? record.payload.results.length : 0} results`;
  }
  const answerSets = Array.isArray(record.payload.answerSets) ? record.payload.answerSets : [];
  const answerCount = answerSets.reduce((total, set) => total + (Array.isArray(set.answers) ? set.answers.length : 0), 0);
  return `${answerCount} ${answerCount === 1 ? "answer" : "answers"}`;
}

function taskNameFor(record) {
  if (record.kind !== "SUBMISSION") return "Result log";
  const answerSets = Array.isArray(record.payload.answerSets) ? record.payload.answerSets : [];
  return answerSets.find((answerSet) => typeof answerSet.taskName === "string")?.taskName || "Task inferred by DRES";
}

function makeRecordRow(record) {
  const button = document.createElement("button");
  const outcome = record.outcome;
  const accepted = outcome.status_code === 200 || outcome.status_code === 202;
  button.type = "button";
  button.className = `record-row${record.kind === "RESULT_LOG" ? " is-result" : ""}`;
  button.setAttribute("aria-pressed", String(record.record_id === state.selectedRecordId));
  button.setAttribute("aria-label", `${record.kind === "SUBMISSION" ? "Submission" : "Result log"} ${record.record_id}, HTTP ${outcome.status_code}, ${record.username}`);

  const time = document.createElement("span");
  time.className = "record-time";
  time.textContent = formatTimestamp(record.received_at_ms);

  const main = document.createElement("span");
  main.className = "record-main";
  const title = document.createElement("span");
  title.className = "record-title";
  title.textContent = taskNameFor(record);
  const subtitle = document.createElement("span");
  subtitle.className = "record-subtitle";
  subtitle.textContent = `${record.username} · ${countFor(record)} · #${record.record_id}`;
  main.append(title, subtitle);

  const result = document.createElement("span");
  result.className = "record-result";
  const status = document.createElement("span");
  status.className = `status-chip${accepted ? " is-accepted" : " is-error"}`;
  status.textContent = `HTTP ${outcome.status_code}`;
  result.append(status);
  if (outcome.verdict) {
    const verdict = document.createElement("span");
    verdict.className = "verdict";
    verdict.textContent = outcome.verdict;
    result.append(verdict);
  }
  button.append(time, main, result);
  button.addEventListener("click", () => {
    state.selectedRecordId = record.record_id;
    renderRecords();
    renderDetail(record);
  });
  return button;
}

function renderRecords() {
  if (!state.snapshot) return;
  const records = filteredRecords(state.snapshot);
  elements.recordList.replaceChildren(...records.map(makeRecordRow));
  elements.emptyJournal.hidden = records.length > 0;
  if (allRecords(state.snapshot).length === 0) {
    const link = document.createElement("a");
    link.href = "/docs";
    link.textContent = "Swagger";
    elements.emptyJournal.replaceChildren(
      document.createTextNode("No requests received yet. Send one through "),
      link,
      document.createTextNode(" or point HCMAI at this local server."),
    );
  } else {
    elements.emptyJournal.textContent = "No requests match this filter.";
  }
  elements.journalCount.textContent = `${records.length} ${records.length === 1 ? "record" : "records"} · newest first`;

  const selected = allRecords(state.snapshot).find((record) => record.record_id === state.selectedRecordId);
  if (selected) renderDetail(selected);
  else renderDetail(null);
}

function addMetadata(label, value) {
  const item = document.createElement("div");
  const term = document.createElement("dt");
  term.textContent = label;
  const detail = document.createElement("dd");
  detail.textContent = value == null || value === "" ? "—" : String(value);
  item.append(term, detail);
  elements.detailMetadata.append(item);
}

function renderDetail(record) {
  if (!record) {
    elements.selectedRecordLabel.textContent = "—";
    elements.detailEmpty.hidden = false;
    elements.detailContent.hidden = true;
    elements.detailMetadata.replaceChildren();
    elements.recordJson.textContent = "";
    return;
  }

  elements.detailEmpty.hidden = true;
  elements.detailContent.hidden = false;
  elements.selectedRecordLabel.textContent = `#${record.record_id}`;
  elements.detailMetadata.replaceChildren();
  addMetadata("Received", `${formatTimestamp(record.received_at_ms)} · ${record.received_at_ms} ms`);
  addMetadata("Request", record.kind === "SUBMISSION" ? "Submission" : "Result log");
  addMetadata("Actor", record.username);
  addMetadata("Evaluation", record.evaluation_id);
  addMetadata("HTTP outcome", String(record.outcome.status_code));
  addMetadata("Verdict", record.outcome.verdict || "Not evaluated");
  addMetadata("Response", record.outcome.description);
  addMetadata("Delay", `${record.outcome.delay_ms} ms`);
  elements.recordJson.textContent = JSON.stringify(record, null, 2);
}

function renderSnapshot(snapshot) {
  state.snapshot = snapshot;
  const task = snapshot.task;
  const evaluation = snapshot.evaluation;
  elements.evaluationName.textContent = evaluation.name;
  elements.evaluationId.textContent = evaluation.id;
  elements.taskName.textContent = task.active ? task.name : "No active task";
  elements.taskMeta.textContent = task.active ? `${task.taskGroup} / ${task.taskType} · ${task.duration ?? "—"} s` : "Task disabled";
  elements.sessionCount.textContent = String(snapshot.session_count);
  elements.submissionCount.textContent = String(snapshot.submission_count);
  elements.resultLogCount.textContent = String(snapshot.result_log_count);
  const refreshedAt = new Date();
  elements.lastRefresh.textContent = `Refreshed ${refreshedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;

  const records = allRecords(snapshot);
  if (!records.some((record) => record.record_id === state.selectedRecordId)) {
    state.selectedRecordId = records[0]?.record_id ?? null;
  }
  renderRecords();
  syncControls(snapshot);
}

function syncControls(snapshot) {
  if (state.controlsDirty) return;
  const task = snapshot.task;
  document.getElementById("task-active").checked = Boolean(task.active);
  document.getElementById("task-name-input").value = task.name ?? "";
  document.getElementById("task-group-input").value = task.taskGroup ?? "";
  document.getElementById("task-type-input").value = task.taskType ?? "";
  document.getElementById("task-duration-input").value = task.duration ?? "";
  fillScenarioForm("submission", snapshot.scenarios.submission);
  fillScenarioForm("result-log", snapshot.scenarios.result_log);
}

function fillScenarioForm(prefix, scenario) {
  document.getElementById(`${prefix}-status`).value = String(scenario.statusCode);
  document.getElementById(`${prefix}-delay`).value = String(scenario.delayMs);
  document.getElementById(`${prefix}-malformed`).checked = Boolean(scenario.malformedBody);
  document.getElementById(`${prefix}-description`).value = scenario.description;
  if (prefix === "submission") {
    document.getElementById("submission-verdict").value = scenario.verdict;
  }
}

async function refreshState() {
  if (state.inFlight) {
    state.refreshPending = true;
    return;
  }
  if (state.paused || document.hidden) return;
  state.inFlight = true;
  state.controller = new AbortController();
  try {
    const response = await fetch("/__test/state", {
      headers: { Accept: "application/json" },
      signal: state.controller.signal,
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.description || `State request failed (${response.status})`);
    renderSnapshot(body);
    setConnection(true, "Connected · state is held in memory and clears on reset or restart.");
  } catch (error) {
    if (error.name !== "AbortError") {
      setConnection(false, `Refresh failed · ${error.message}. Showing the last successful snapshot.`);
    }
  } finally {
    state.inFlight = false;
    state.controller = null;
    if (state.refreshPending) {
      state.refreshPending = false;
      if (!state.paused && !document.hidden) void refreshState();
    }
  }
}

async function sendControl(url, method, payload) {
  const response = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: payload === undefined ? undefined : JSON.stringify(payload),
  });
  let body = {};
  try { body = await response.json(); } catch (_) { /* Show the HTTP status below. */ }
  if (!response.ok) throw new Error(body.description || `Control request failed (${response.status})`);
  state.controlsDirty = false;
  elements.liveMessage.textContent = "Control saved · refreshing server state…";
  elements.liveMessage.classList.remove("is-error");
  await refreshState();
}

function formValues(form) {
  return Object.fromEntries(new FormData(form).entries());
}

function scenarioPayload(form, submission) {
  const formData = new FormData(form);
  const payload = {
    statusCode: Number(formData.get("statusCode")),
    delayMs: Number(formData.get("delayMs")),
    malformedBody: formData.has("malformedBody"),
    description: String(formData.get("description") ?? ""),
  };
  if (submission) payload.verdict = String(formData.get("verdict"));
  return payload;
}

document.querySelectorAll("[data-filter]").forEach((button) => {
  button.addEventListener("click", () => {
    state.filter = button.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach((item) => {
      item.setAttribute("aria-pressed", String(item === button));
    });
    renderRecords();
  });
});

document.querySelectorAll(".control-form").forEach((form) => {
  form.addEventListener("input", () => { state.controlsDirty = true; });
  form.addEventListener("change", () => { state.controlsDirty = true; });
});

elements.refreshButton.addEventListener("click", () => { void refreshState(); });
elements.pauseButton.addEventListener("click", () => {
  state.paused = !state.paused;
  elements.pauseButton.textContent = state.paused ? "Resume updates" : "Pause updates";
  if (state.paused) {
    state.controller?.abort();
    elements.liveMessage.textContent = "Updates paused · use Refresh for a manual snapshot.";
  } else {
    void refreshState();
  }
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) state.controller?.abort();
  else void refreshState();
});

document.getElementById("task-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = formValues(event.currentTarget);
  const duration = document.getElementById("task-duration-input").value;
  const payload = {
    active: document.getElementById("task-active").checked,
    name: String(data.name ?? ""),
    taskGroup: String(data.taskGroup ?? ""),
    taskType: String(data.taskType ?? ""),
    duration: duration === "" ? null : Number(duration),
  };
  try { await sendControl("/__test/task", "PUT", payload); }
  catch (error) { elements.liveMessage.textContent = error.message; elements.liveMessage.classList.add("is-error"); }
});

document.getElementById("submission-scenario-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try { await sendControl("/__test/scenario/submission", "PUT", scenarioPayload(event.currentTarget, true)); }
  catch (error) { elements.liveMessage.textContent = error.message; elements.liveMessage.classList.add("is-error"); }
});

document.getElementById("result-log-scenario-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try { await sendControl("/__test/scenario/result-log", "PUT", scenarioPayload(event.currentTarget, false)); }
  catch (error) { elements.liveMessage.textContent = error.message; elements.liveMessage.classList.add("is-error"); }
});

elements.resetButton.addEventListener("click", async () => {
  if (!window.confirm("Reset all DRES mock sessions, traffic, task changes, and scenarios?")) return;
  try { await sendControl("/__test/reset", "POST"); }
  catch (error) { elements.liveMessage.textContent = error.message; elements.liveMessage.classList.add("is-error"); }
});

window.setInterval(() => { void refreshState(); }, 1_000);
void refreshState();
