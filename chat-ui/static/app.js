(() => {
  const MAX_FILE_BYTES = 512 * 1024;
  const MAX_TOTAL_ATTACH = 1.5 * 1024 * 1024;
  const MAX_OCR_UPLOAD = 12 * 1024 * 1024;
  const CONTEXT_WARN_RATIO = 0.85;
  const CONTEXT_SUMMARIZE_RATIO = 0.72;
  const CONTEXT_HARD_RATIO = 0.95;
  const KEEP_RECENT_MESSAGES = 6;
  /** Manual / fork / RAM force: keep a short tail so compress can actually shrink. */
  const FORCE_KEEP_RECENT_MESSAGES = 2;
  const MAX_COMPRESS_PASSES = 4;
  const SUMMARY_PREFIX = "[conversation summary]";
  // Same 6 GiB reserve as scripts/safe_ctx.py DEFAULT_RESERVE_GIB / RAM_PRESSURE_*.
  const RAM_PRESSURE_TRIGGER_GIB = 6;
  const RAM_PRESSURE_CLEAR_GIB = 7;
  const TEXT_EXT = new Set([
    "txt", "md", "py", "c", "h", "cpp", "hpp", "cc", "js", "ts", "tsx", "jsx",
    "json", "csv", "toml", "yaml", "yml", "rs", "go", "java", "sh", "bash",
    "zsh", "css", "html", "xml", "sql", "ini", "cfg", "log", "makefile", "cmake",
    "swift", "kt", "rb", "php", "r", "lua", "vim", "diff", "patch",
  ]);
  const OCR_EXT = new Set(["png", "jpg", "jpeg", "webp", "gif", "tif", "tiff", "bmp", "pdf"]);

  const els = {
    rail: document.getElementById("rail"),
    railToggle: document.getElementById("railToggle"),
    chatList: document.getElementById("chatList"),
    statusLine: document.getElementById("statusLine"),
    chatTitle: document.getElementById("chatTitle"),
    transcript: document.getElementById("transcript"),
    prompt: document.getElementById("prompt"),
    sendBtn: document.getElementById("sendBtn"),
    newChatBtn: document.getElementById("newChatBtn"),
    renameBtn: document.getElementById("renameBtn"),
    deleteBtn: document.getElementById("deleteBtn"),
    summarizeBtn: document.getElementById("summarizeBtn"),
    summarizeForkBtn: document.getElementById("summarizeForkBtn"),
    fileInput: document.getElementById("fileInput"),
    attachBar: document.getElementById("attachBar"),
    webToggle: document.getElementById("webToggle"),
    ttsToggle: document.getElementById("ttsToggle"),
    ttsStopBtn: document.getElementById("ttsStopBtn"),
    composerNote: document.querySelector(".composer-note"),
    ramStatus: document.getElementById("ram-status"),
    ramMeterFill: document.getElementById("ram-meter-fill"),
    summarizeProgress: document.getElementById("summarizeProgress"),
    summarizeProgressLabel: document.getElementById("summarizeProgressLabel"),
    ramNotice: document.getElementById("ramNotice"),
  };

  /** @type {{id:string,title:string,messages:any[],created_at?:string,updated_at?:string}|null} */
  let current = null;
  /** @type {{name:string,text:string,bytes:number,source?:string,method?:string}[]} */
  let pendingFiles = [];
  /** @type {WeakMap<object, boolean>} session expand state for reasoning mirrors */
  const thinkExpandedByMsg = new WeakMap();
  /** @type {WeakMap<object, boolean>} session expand state for web-context panels */
  const webExpandedByMsg = new WeakMap();
  /** @type {HTMLElement|null} */
  let activityLoaderEl = null;
  let activityToken = 0;
  /** @type {ReturnType<typeof setTimeout>|null} */
  let activityPhaseTimer = null;
  let busy = false;
  let modelId = "deepseek-v4-flash";
  let contextLength = 100000;
  let safeCtx = null;
  /** @type {HTMLAudioElement|null} */
  let ttsAudio = null;
  let ttsToken = 0;
  let ttsAvailable = false;
  let audioUnlocked = false;
  /** @type {AbortController|null} */
  let generationAbort = null;
  let ramAbortRequested = false;
  let ramPressureLatched = false;
  let ramFallbackRunning = false;
  let ramTriggerGib = RAM_PRESSURE_TRIGGER_GIB;
  let ramClearGib = RAM_PRESSURE_CLEAR_GIB;
  let lastRamActionLogged = null;
  let ramNoticeTimer = null;
  let summarizeLoaderDepth = 0;

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  const COPY_ICON =
    '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" focusable="false"><rect x="9" y="9" width="11" height="11" rx="2" fill="none" stroke="currentColor" stroke-width="1.75"/><path d="M6 15H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v1" fill="none" stroke="currentColor" stroke-width="1.75"/></svg>';
  const CHECK_ICON =
    '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" focusable="false"><path d="M5 12.5l4.2 4.2L19 7.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const SPEAK_ICON =
    '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" focusable="false"><path d="M4 9v6h3l5 4V5L7 9H4z" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linejoin="round"/><path d="M16 9.5a3.5 3.5 0 0 1 0 5" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/><path d="M18.2 7a6 6 0 0 1 0 10" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/></svg>';

  function renderMarkdownHtml(src) {
    const text = String(src || "");
    if (!text) return "";
    try {
      if (typeof marked !== "undefined" && marked.parse) {
        marked.setOptions({ gfm: true, breaks: true });
        const html = marked.parse(text);
        if (typeof DOMPurify !== "undefined" && DOMPurify.sanitize) {
          return DOMPurify.sanitize(html, {
            USE_PROFILES: { html: true },
          });
        }
        return html;
      }
    } catch {
      /* fall through */
    }
    return escapeHtml(text).replace(/\n/g, "<br>");
  }

  function ensurePreviewEl(body) {
    let preview = body.querySelector(".md-preview");
    if (!preview) {
      body.textContent = "";
      preview = document.createElement("div");
      preview.className = "md-preview";
      body.appendChild(preview);
    }
    return preview;
  }

  function setBubbleText(bubble, text) {
    const body = bubble.querySelector(".bubble-body");
    const raw = text || "";
    if (body) {
      body.dataset.raw = raw;
      const preview = ensurePreviewEl(body);
      preview.innerHTML = renderMarkdownHtml(raw);
      return;
    }
    bubble.textContent = raw;
  }

  function thinkBodyText(wrap) {
    const body = wrap.querySelector(".think-body");
    if (body) return (body.textContent || "").trim();
    const think = wrap.querySelector(".bubble.think");
    return think ? (think.textContent || "").trim() : "";
  }

  function messageCopyText(wrap, body) {
    const raw = (body.dataset.raw || "").trim();
    if (raw) return raw;
    return thinkBodyText(wrap);
  }

  function messagePreviewText(wrap, body) {
    const preview = body.querySelector(".md-preview");
    const visible = preview
      ? (preview.innerText || preview.textContent || "")
      : body.innerText || body.textContent || "";
    const main = String(visible || "").replace(/\u00a0/g, " ").trim();
    if (main) return main;
    return thinkBodyText(wrap);
  }

  function isThinkExpanded(msg) {
    return !!(msg && thinkExpandedByMsg.get(msg));
  }

  function setThinkExpanded(msg, expanded) {
    if (msg) thinkExpandedByMsg.set(msg, !!expanded);
  }

  function createThinkMirror(msg, text) {
    const expanded = isThinkExpanded(msg);
    const think = document.createElement("div");
    think.className = "bubble think" + (expanded ? " expanded" : "");
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "think-toggle";
    toggle.setAttribute("aria-expanded", expanded ? "true" : "false");
    const label = document.createElement("span");
    label.className = "think-label";
    label.textContent = "Reasoning";
    const chevron = document.createElement("span");
    chevron.className = "think-chevron";
    chevron.setAttribute("aria-hidden", "true");
    toggle.appendChild(label);
    toggle.appendChild(chevron);
    const body = document.createElement("div");
    body.className = "think-body";
    body.textContent = text != null ? text : (msg && msg.reasoning) || "";
    toggle.addEventListener("click", () => {
      const next = !think.classList.contains("expanded");
      think.classList.toggle("expanded", next);
      toggle.setAttribute("aria-expanded", next ? "true" : "false");
      setThinkExpanded(msg, next);
    });
    think.appendChild(toggle);
    think.appendChild(body);
    return think;
  }

  function setThinkMirrorText(thinkEl, text) {
    if (!thinkEl) return;
    const body = thinkEl.querySelector(".think-body");
    if (body) {
      body.textContent = text || "";
      return;
    }
    thinkEl.textContent = text || "";
  }

  const WEB_CONTEXT_RE =
    /----- web context[\s\S]*?----- end web context -----\s*/i;

  function extractWebContextBlock(content) {
    const raw = String(content || "");
    const match = raw.match(WEB_CONTEXT_RE);
    if (!match) return { web: "", rest: raw };
    const web = match[0].trim();
    const rest = (raw.slice(0, match.index) + raw.slice(match.index + match[0].length))
      .replace(/^\s+/, "")
      .replace(/\s+$/, "");
    return { web, rest };
  }

  function isWebExpanded(msg) {
    return !!(msg && webExpandedByMsg.get(msg));
  }

  function setWebExpanded(msg, expanded) {
    if (msg) webExpandedByMsg.set(msg, !!expanded);
  }

  function createWebMirror(msg, text) {
    const expanded = isWebExpanded(msg);
    const web = document.createElement("div");
    web.className = "bubble webctx" + (expanded ? " expanded" : "");
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "webctx-toggle";
    toggle.setAttribute("aria-expanded", expanded ? "true" : "false");
    const label = document.createElement("span");
    label.className = "webctx-label";
    label.textContent = "Web context";
    const chevron = document.createElement("span");
    chevron.className = "webctx-chevron";
    chevron.setAttribute("aria-hidden", "true");
    toggle.appendChild(label);
    toggle.appendChild(chevron);
    const body = document.createElement("div");
    body.className = "webctx-body";
    body.textContent = text != null ? text : "";
    toggle.addEventListener("click", () => {
      const next = !web.classList.contains("expanded");
      web.classList.toggle("expanded", next);
      toggle.setAttribute("aria-expanded", next ? "true" : "false");
      setWebExpanded(msg, next);
    });
    web.appendChild(toggle);
    web.appendChild(body);
    return web;
  }

  function contentBubbleEl(node) {
    return node.querySelector(".bubble-main") || node.querySelector(".bubble:not(.think):not(.webctx)");
  }

  function clearActivityPhaseTimer() {
    if (activityPhaseTimer != null) {
      clearTimeout(activityPhaseTimer);
      activityPhaseTimer = null;
    }
  }

  /** Drop the in-flight activity row / spinner class (status text left alone). */
  function clearActivity() {
    activityToken += 1;
    clearActivityPhaseTimer();
    if (activityLoaderEl) {
      activityLoaderEl.remove();
      activityLoaderEl = null;
    }
    if (els.statusLine) {
      els.statusLine.classList.remove("status-searching", "status-activity");
    }
  }

  /**
   * Human-readable pipeline phase. Updates status line; optionally a transcript row
   * (same chrome as the old web-search loader).
   * @param {string} message
   * @param {{ transcript?: boolean, searching?: boolean }} [opts]
   */
  function setActivity(message, opts = {}) {
    const text = String(message || "").trim();
    if (!text) {
      clearActivity();
      return;
    }
    const showTranscript = opts.transcript !== false;
    const searching = !!opts.searching;
    const summarizing = summarizeLoaderDepth > 0;

    if (els.statusLine) {
      els.statusLine.classList.remove("status-searching", "status-activity");
      if (!summarizing) {
        els.statusLine.classList.add(searching ? "status-searching" : "status-activity");
      }
    }
    setStatus(text, null);

    if (!showTranscript || summarizing) {
      if (activityLoaderEl) {
        const span = activityLoaderEl.querySelector(".activity-label");
        if (span) span.textContent = text;
      }
      return;
    }

    const empty = els.transcript.querySelector(".empty");
    if (empty) empty.remove();

    if (!activityLoaderEl) {
      const el = document.createElement("div");
      el.className = "activity-loader" + (searching ? " web-search-loader" : "");
      el.setAttribute("role", "status");
      el.setAttribute("aria-live", "polite");
      el.innerHTML =
        '<span class="activity-spinner web-search-spinner" aria-hidden="true"></span>' +
        `<span class="activity-label">${escapeHtml(text)}</span>`;
      els.transcript.appendChild(el);
      activityLoaderEl = el;
    } else {
      activityLoaderEl.className =
        "activity-loader" + (searching ? " web-search-loader" : "");
      const span = activityLoaderEl.querySelector(".activity-label");
      if (span) span.textContent = text;
    }
    els.transcript.scrollTop = els.transcript.scrollHeight;
  }

  function showWebSearchLoader(label) {
    setActivity(label ? `Searching: ${label}` : "Searching the web…", {
      searching: true,
    });
  }

  function setWebSearchLoaderQuery(query) {
    if (!query) return;
    setActivity(`Searching: ${query}`, { searching: true });
  }

  function hideWebSearchLoader() {
    clearActivity();
  }

  /** Timed web-pipeline labels while /api/web-context is in flight (no SSE). */
  function startWebActivityPhases() {
    const token = ++activityToken;
    clearActivityPhaseTimer();
    setActivity("Resolving search intent…", { searching: true });
    activityPhaseTimer = setTimeout(() => {
      if (token !== activityToken) return;
      setActivity("Searching the web…", { searching: true });
      activityPhaseTimer = setTimeout(() => {
        if (token !== activityToken) return;
        setActivity("Reading pages…", { searching: true });
      }, 2200);
    }, 850);
  }

  async function unlockAudio() {
    if (audioUnlocked) return;
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (AC) {
        const ctx = new AC();
        if (ctx.state === "suspended") await ctx.resume();
        const buf = ctx.createBuffer(1, 1, 22050);
        const src = ctx.createBufferSource();
        src.buffer = buf;
        src.connect(ctx.destination);
        src.start(0);
        audioUnlocked = true;
        return;
      }
    } catch {
      /* fall through */
    }
    try {
      const silent =
        "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA=";
      const a = new Audio(silent);
      await a.play();
      a.pause();
      audioUnlocked = true;
    } catch {
      /* browser may still allow play() after a later click */
    }
  }

  async function copyTextToClipboard(text) {
    const value = String(text || "");
    if (!value) return false;
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      try {
        await navigator.clipboard.writeText(value);
        return true;
      } catch {
        /* fall through */
      }
    }
    const ta = document.createElement("textarea");
    ta.value = value;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    ta.style.top = "0";
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch {
      ok = false;
    }
    ta.remove();
    return ok;
  }

  function makeCopyButton(getText) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "msg-action-btn copy-btn";
    btn.setAttribute("aria-label", "Copy");
    btn.title = "Copy";
    btn.innerHTML = COPY_ICON;
    let resetTimer = null;
    btn.addEventListener("click", async (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      const text = String(getText() || "").trim();
      if (!text) return;
      const ok = await copyTextToClipboard(text);
      if (!ok) return;
      btn.classList.add("is-copied");
      btn.setAttribute("aria-label", "Copied");
      btn.title = "Copied";
      btn.innerHTML = `${CHECK_ICON}<span class="copy-feedback">Copied</span>`;
      if (resetTimer) clearTimeout(resetTimer);
      resetTimer = setTimeout(() => {
        btn.classList.remove("is-copied");
        btn.setAttribute("aria-label", "Copy");
        btn.title = "Copy";
        btn.innerHTML = COPY_ICON;
        resetTimer = null;
      }, 1000);
    });
    return btn;
  }

  function makeSpeakButton(getText) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "msg-action-btn speak-btn";
    btn.setAttribute("aria-label", "Read aloud");
    btn.title = "Read this message aloud";
    btn.innerHTML = SPEAK_ICON;
    btn.addEventListener("click", async (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      const text = String(getText() || "").trim();
      if (!text) {
        setStatus("Nothing to read in this message.", false);
        return;
      }
      await unlockAudio();
      await speakFinal(text, { force: true });
    });
    return btn;
  }

  function fmtTime(iso) {
    if (!iso) return "";
    try {
      return new Date(iso).toLocaleString();
    } catch {
      return iso;
    }
  }

  function isUiFailureAssistant(msg) {
    if (!msg || msg.role !== "assistant") return false;
    const c = (msg.content || "").trim();
    return c.startsWith("Error:");
  }

  function estimateTokens(text) {
    // Rough local estimate; server tokenizer is authoritative.
    return Math.ceil(String(text || "").length / 4);
  }

  function historyTokenEstimate(messages) {
    return messages.reduce((n, m) => n + estimateTokens(m.content), 0);
  }

  function formatTokenEstimate(n) {
    const v = Math.max(0, Number(n) || 0);
    if (v >= 1000) {
      const k = v / 1000;
      // Keep one decimal under 100k so "12.4k" stays readable.
      return (k >= 100 ? k.toFixed(0) : k.toFixed(1).replace(/\.0$/, "")) + "k";
    }
    return String(Math.round(v));
  }

  function formatSizeLabel(bytes) {
    const n = Math.max(0, Number(bytes) || 0);
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) {
      const kb = n / 1024;
      return (kb >= 10 ? kb.toFixed(0) : kb.toFixed(1)) + " KB";
    }
    const mb = n / (1024 * 1024);
    return (mb >= 10 ? mb.toFixed(0) : mb.toFixed(1)) + " MB";
  }

  function chatListMeta(chat) {
    // List rows from /api/chats omit messages; prefer server token_estimate.
    const toks =
      chat.token_estimate != null
        ? chat.token_estimate
        : historyTokenEstimate(chat.messages || []);
    const size =
      chat.size_label ||
      (chat.size_bytes != null ? formatSizeLabel(chat.size_bytes) : "");
    const when = fmtTime(chat.updated_at);
    const parts = [];
    if (when) parts.push(`Updated ${when}`);
    parts.push(`${formatTokenEstimate(toks)} tok`);
    if (size) parts.push(size);
    return parts.join(" · ");
  }

  function parseErrorPayload(text) {
    const raw = (text || "").trim();
    if (!raw) return "request failed";
    try {
      const data = JSON.parse(raw);
      const err = data && data.error;
      if (typeof err === "string") return err;
      if (err && typeof err === "object") {
        const msg = err.message || JSON.stringify(err);
        if (err.code && !String(msg).includes(err.code)) return `${msg} (${err.code})`;
        return String(msg);
      }
      if (data && data.message) return String(data.message);
    } catch {
      /* plain text */
    }
    return raw.slice(0, 2000);
  }

  async function api(path, opts = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
      ...opts,
    });
    const text = await res.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = { raw: text };
    }
    if (!res.ok) {
      const msg = (data && data.error) || parseErrorPayload(text) || res.statusText || "request failed";
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return data;
  }

  function setStatus(text, ok) {
    if (ok === true || ok === false) {
      activityToken += 1;
      clearActivityPhaseTimer();
      if (activityLoaderEl) {
        activityLoaderEl.remove();
        activityLoaderEl = null;
      }
      if (els.statusLine) {
        els.statusLine.classList.remove("status-searching", "status-activity");
      }
    }
    els.statusLine.textContent = text;
    els.statusLine.style.color = ok === false ? "var(--danger)" : ok ? "var(--teal)" : "var(--muted)";
  }

  function isAbortError(err) {
    if (!err) return false;
    if (err.name === "AbortError") return true;
    return /aborted|AbortError/i.test(String(err.message || ""));
  }

  function beginGeneration() {
    if (generationAbort) {
      try {
        generationAbort.abort();
      } catch {
        /* ignore */
      }
    }
    generationAbort = new AbortController();
    return generationAbort;
  }

  function endGeneration(controller) {
    if (generationAbort === controller) generationAbort = null;
  }

  function abortGenerationForRam() {
    ramAbortRequested = true;
    if (!generationAbort) return false;
    try {
      generationAbort.abort();
    } catch {
      /* ignore */
    }
    return true;
  }

  /** Mirror of safe_ctx.evaluate_ram_pressure (trigger <=6 GiB, clear >7 GiB). */
  function evaluateRamPressure(freeGib, latched) {
    if (!Number.isFinite(freeGib)) return "idle";
    if (freeGib <= ramTriggerGib) return latched ? "hold" : "trigger";
    if (latched && freeGib > ramClearGib) return "clear";
    return "idle";
  }

  function wantForceRamPressure() {
    try {
      if (new URLSearchParams(location.search).get("forceRamPressure") === "1") {
        return true;
      }
      return localStorage.getItem("ds4-force-ram-pressure") === "1";
    } catch {
      return false;
    }
  }

  function logRamPressureEval(freeGib, action) {
    if (action === lastRamActionLogged) return;
    lastRamActionLogged = action;
    console.debug("[ram-pressure]", {
      free_gib: Number(Number(freeGib).toFixed(2)),
      action,
      trigger: ramTriggerGib,
      clear: ramClearGib,
      latched: ramPressureLatched,
    });
  }

  function showRamNotice(text) {
    if (!els.ramNotice) return;
    els.ramNotice.textContent = text;
    els.ramNotice.hidden = false;
    if (ramNoticeTimer) clearTimeout(ramNoticeTimer);
    ramNoticeTimer = setTimeout(() => {
      if (els.ramNotice) els.ramNotice.hidden = true;
    }, 14000);
  }

  function showSummarizeLoader(label) {
    summarizeLoaderDepth += 1;
    if (els.summarizeProgress) {
      els.summarizeProgress.hidden = false;
      els.summarizeProgress.setAttribute("aria-hidden", "false");
    }
    if (els.summarizeProgressLabel) {
      els.summarizeProgressLabel.textContent = label || "Summarizing…";
    }
    if (els.statusLine) els.statusLine.classList.add("status-summarizing");
  }

  function setSummarizeLoaderLabel(label) {
    if (els.summarizeProgressLabel && label) {
      els.summarizeProgressLabel.textContent = label;
    }
  }

  function hideSummarizeLoader() {
    summarizeLoaderDepth = Math.max(0, summarizeLoaderDepth - 1);
    if (summarizeLoaderDepth > 0) return;
    if (els.summarizeProgress) {
      els.summarizeProgress.hidden = true;
      els.summarizeProgress.setAttribute("aria-hidden", "true");
    }
    if (els.statusLine) els.statusLine.classList.remove("status-summarizing");
  }

  function warnIfHuge(messages) {
    const est = historyTokenEstimate(messages);
    const limit = Math.floor(contextLength * CONTEXT_WARN_RATIO);
    if (est >= limit) {
      setStatus(
        `History ~${est.toLocaleString()} tokens (ctx ${contextLength.toLocaleString()}). Auto-summarize will run on send if needed.`,
        false
      );
      return true;
    }
    return false;
  }

  function isSummaryMessage(msg) {
    if (!msg) return false;
    if (msg.compressed) return true;
    return String(msg.content || "").startsWith(SUMMARY_PREFIX);
  }

  function workingMessages() {
    return filterWorkingMessages(current && current.messages ? current.messages : []);
  }

  function formatForSummary(msgs) {
    return msgs
      .map((m) => {
        const role = (m.role || "unknown").toUpperCase();
        let content = String(m.content || "").trim();
        if (content.length > 14000) {
          content = content.slice(0, 14000) + "\n…[clipped for summary]";
        }
        return `${role}:\n${content}`;
      })
      .join("\n\n---\n\n");
  }

  async function requestSummary(excerpt, signal) {
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({
        model: modelId,
        stream: false,
        temperature: 0.2,
        max_tokens: 2048,
        thinking: { type: "disabled" },
        messages: [
          {
            role: "system",
            content:
              "You compress conversation history for a continuing chat. " +
              "Write a dense factual summary: goals, constraints, decisions, key facts, numbers, " +
              "names, file/topic references, and open questions. No preamble or meta commentary.",
          },
          {
            role: "user",
            content: "Summarize this conversation excerpt:\n\n" + excerpt,
          },
        ],
      }),
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(parseErrorPayload(errText) || res.statusText);
    }
    const data = await res.json();
    const text = (data?.choices?.[0]?.message?.content || "").trim();
    if (!text) throw new Error("empty summary from model");
    return text;
  }

  function filterWorkingMessages(messages) {
    return (messages || []).filter((m) => !isUiFailureAssistant(m));
  }

  function keepRecentForPass(force, pass) {
    const baseKeep = force ? FORCE_KEEP_RECENT_MESSAGES : KEEP_RECENT_MESSAGES;
    // Each pass drops two more recent msgs so a stuck fat tail can shrink.
    return Math.max(0, baseKeep - (Math.max(1, pass) - 1) * 2);
  }

  /**
   * Pure slice planner (mirrored in chat-ui/compress_history.py).
   * Returns null when there is nothing older than the keep tail to summarize.
   */
  function planCompressSlices(messages, keepRecent, opts = {}) {
    const base = Array.isArray(messages) ? messages.slice() : [];
    if (base.length < 2) return null;
    const keep = Math.max(0, Math.min(Number(keepRecent) || 0, base.length - 1));
    const older = base.slice(0, base.length - keep);
    const recent = base.slice(base.length - keep);
    if (!older.length) return null;

    const allowLeftover = opts.allowLeftover !== false;
    const summarizeBudget = opts.summarizeBudget;
    let chunk = older;
    let leftover = [];
    if (
      allowLeftover &&
      Number.isFinite(summarizeBudget) &&
      summarizeBudget > 0
    ) {
      while (chunk.length > 2 && historyTokenEstimate(chunk) > summarizeBudget) {
        const half = Math.max(1, Math.ceil(chunk.length / 2));
        leftover = chunk.slice(half).concat(leftover);
        chunk = chunk.slice(0, half);
      }
    }
    return { older, recent, chunk, leftover, keep };
  }

  /** One compress pass over a message list; does not touch `current` or disk. */
  async function buildCompressedMessages(messages, signal, opts = {}) {
    const keepRecent =
      opts.keepRecent != null ? opts.keepRecent : KEEP_RECENT_MESSAGES;
    const allowLeftover = opts.allowLeftover !== false;
    const base = filterWorkingMessages(messages);
    const summarizeBudget = Math.max(2048, Math.floor(contextLength * 0.5));
    const plan = planCompressSlices(base, keepRecent, {
      allowLeftover,
      summarizeBudget,
    });
    if (!plan) return null;

    const summaryText = await requestSummary(
      formatForSummary(plan.chunk),
      signal
    );
    return [
      {
        role: "system",
        content: `${SUMMARY_PREFIX}\n${summaryText}`,
        compressed: true,
      },
      ...plan.leftover,
      ...plan.recent,
    ];
  }

  async function compressHistoryOnce(signal) {
    const next = await buildCompressedMessages(workingMessages(), signal);
    if (!next) return false;
    current.messages = next;
    renderTranscript();
    await saveCurrent();
    return true;
  }

  function summarizePassStatus(reason, pass, est) {
    if (reason === "manual") {
      return `Summarizing older turns (pass ${pass})…`;
    }
    if (reason === "fork") {
      return `Summarizing into a new chat (pass ${pass})…`;
    }
    if (reason === "ram") {
      return `Low free RAM — summarizing into a new chat (pass ${pass})…`;
    }
    return (
      `Context ~${est.toLocaleString()} / ${contextLength.toLocaleString()} — ` +
      `summarizing older turns (pass ${pass})…`
    );
  }

  /**
   * Compress a message array in memory (force ignores auto token threshold).
   * When applyToCurrent is true, also updates the open chat and saves it.
   */
  async function compressMessagesList(messages, opts = {}) {
    const force = !!opts.force;
    const signal = opts.signal;
    const reason = opts.reason || (force ? "ram" : "auto");
    const pendingUserContent = opts.pendingUserContent || "";
    const applyToCurrent = !!opts.applyToCurrent;
    let msgs = filterWorkingMessages(messages);
    let compressed = false;
    let loaderOwned = false;
    try {
      for (let pass = 1; pass <= MAX_COMPRESS_PASSES; pass++) {
        const preview = pendingUserContent
          ? msgs.concat([{ role: "user", content: pendingUserContent }])
          : msgs.slice();
        const est = historyTokenEstimate(preview);
        const trigger = Math.floor(contextLength * CONTEXT_SUMMARIZE_RATIO);
        const hard = Math.floor(contextLength * CONTEXT_HARD_RATIO);
        if (!force && est < trigger) {
          return { messages: msgs, compressed, stillHuge: est >= hard };
        }
        const keepRecent = keepRecentForPass(force, pass);
        // Need at least one message beyond the keep tail (old keep+1 gate
        // froze force passes at summary+6 and could not shrink further).
        if (msgs.length <= keepRecent || msgs.length < 2) {
          return { messages: msgs, compressed, stillHuge: est >= hard };
        }
        const passStatus = summarizePassStatus(reason, pass, est);
        setActivity("Summarizing context…", { transcript: false });
        setStatus(passStatus, null);
        if (!loaderOwned) {
          showSummarizeLoader(passStatus);
          loaderOwned = true;
        } else {
          setSummarizeLoaderLabel(passStatus);
        }
        try {
          const beforeTok = historyTokenEstimate(msgs);
          const next = await buildCompressedMessages(msgs, signal, {
            keepRecent,
            // Force/fork: summarize the whole older block; leftover would
            // retain fat mid-history and defeat the shrink goal.
            allowLeftover: !force,
          });
          if (!next) return { messages: msgs, compressed, stillHuge: true };
          const afterTok = historyTokenEstimate(next);
          if (afterTok >= beforeTok) {
            // Do not keep a larger history; later passes use a shorter keep.
            if (pass < MAX_COMPRESS_PASSES && keepRecent > 0) continue;
            return { messages: msgs, compressed, stillHuge: true };
          }
          msgs = next;
          compressed = true;
          if (applyToCurrent && current) {
            current.messages = msgs;
            renderTranscript();
            await saveCurrent();
          }
          // Force target is summary + short tail. Only peel further (keep→0)
          // when still at/over the hard context ceiling.
          if (force && afterTok < hard) {
            return { messages: msgs, compressed, stillHuge: false };
          }
        } catch (err) {
          if (isAbortError(err)) throw err;
          const label =
            reason === "manual" || reason === "fork" || reason === "ram"
              ? "Summarize"
              : "Auto-summarize";
          setStatus(`${label} failed: ${err.message}`, false);
          return { messages: msgs, compressed, stillHuge: true };
        }
      }
      const est = historyTokenEstimate(
        pendingUserContent
          ? msgs.concat([{ role: "user", content: pendingUserContent }])
          : msgs
      );
      return {
        messages: msgs,
        compressed,
        stillHuge: est >= Math.floor(contextLength * CONTEXT_HARD_RATIO),
      };
    } finally {
      if (loaderOwned) hideSummarizeLoader();
    }
  }

  async function compressHistoryIfNeeded(pendingUserContent, opts = {}) {
    if (!current) return { compressed: false, stillHuge: false };
    const force = !!opts.force;
    const result = await compressMessagesList(workingMessages(), {
      force,
      signal: opts.signal,
      reason: opts.reason || (force ? "ram" : "auto"),
      pendingUserContent: pendingUserContent || "",
      applyToCurrent: true,
    });
    return {
      compressed: result.compressed,
      stillHuge: result.stillHuge,
    };
  }

  function setSummarizeBusy(on) {
    if (els.summarizeBtn) els.summarizeBtn.disabled = on;
    if (els.summarizeForkBtn) els.summarizeForkBtn.disabled = on;
  }

  function setChatBusy(on) {
    busy = on;
    els.sendBtn.disabled = on;
    setSummarizeBusy(on);
  }

  async function summarizeInPlace() {
    if (!current || busy || ramFallbackRunning) return;
    if (workingMessages().length < 2) {
      setStatus("Not enough history to summarize.", null);
      return;
    }
    setChatBusy(true);
    showSummarizeLoader("Summarizing this chat…");
    const gen = beginGeneration();
    try {
      const result = await compressHistoryIfNeeded("", {
        force: true,
        reason: "manual",
        signal: gen.signal,
      });
      if (result.compressed) {
        setStatus("History summarized in this chat.", true);
        warnIfHuge(apiMessages());
      } else {
        setStatus("Nothing left to compress in this chat.", null);
      }
    } catch (err) {
      if (!isAbortError(err)) {
        setStatus(`Summarize failed: ${err.message}`, false);
      }
    } finally {
      endGeneration(gen);
      hideSummarizeLoader();
      setChatBusy(false);
    }
  }

  async function forkSummarizedChat(sourceMessages, sourceTitle, opts = {}) {
    const reason = opts.reason || "fork";
    const signal = opts.signal;
    const result = await compressMessagesList(sourceMessages, {
      force: true,
      reason,
      applyToCurrent: false,
      signal,
    });
    if (!result.compressed) {
      return { ok: false, reason: "nothing" };
    }
    const created = await api("/api/chats", {
      method: "POST",
      body: JSON.stringify({
        title: `Summary · ${String(sourceTitle || "chat").slice(0, 40)}`,
      }),
    });
    created.messages = result.messages;
    current = created;
    pendingFiles = [];
    renderAttachBar();
    els.chatTitle.textContent = current.title;
    renderTranscript();
    await saveCurrent();
    await refreshList();
    return { ok: true };
  }

  async function summarizeToNewChat() {
    if (!current || busy || ramFallbackRunning) return;
    const sourceTitle = current.title || "chat";
    const sourceMessages = JSON.parse(JSON.stringify(current.messages || []));
    if (filterWorkingMessages(sourceMessages).length < 2) {
      setStatus("Not enough history to summarize into a new chat.", null);
      return;
    }
    setChatBusy(true);
    showSummarizeLoader("Summarizing into a new chat…");
    const gen = beginGeneration();
    try {
      const forked = await forkSummarizedChat(sourceMessages, sourceTitle, {
        reason: "fork",
        signal: gen.signal,
      });
      if (!forked.ok) {
        setStatus("Nothing left to compress for a new chat.", null);
        return;
      }
      setStatus("Opened new chat from summary. Original left unchanged.", true);
      warnIfHuge(apiMessages());
    } catch (err) {
      if (!isAbortError(err)) {
        setStatus(`Summarize to new chat failed: ${err.message}`, false);
      }
    } finally {
      endGeneration(gen);
      hideSummarizeLoader();
      setChatBusy(false);
    }
  }

  function annotateStoppedForRam() {
    if (!current) return;
    const msgs = current.messages || [];
    const last = msgs[msgs.length - 1];
    if (!(last && last.role === "assistant")) return;
    const empty =
      !String(last.content || "").trim() && !String(last.reasoning || "").trim();
    if (empty) {
      msgs.pop();
      return;
    }
    if (!/\[stopped: low free RAM\]/.test(String(last.content || ""))) {
      last.content =
        (String(last.content || "").trim() ? String(last.content).trim() + "\n\n" : "") +
        "[stopped: low free RAM]";
    }
  }

  async function handleRamPressure(freeGib) {
    // Client poll of /api/ram aborts UI-owned work; bare ds4-server is not killed.
    if (ramFallbackRunning) return;
    ramFallbackRunning = true;
    stopSpeaking();
    const notice =
      `Low RAM (${freeGib.toFixed(1)} GiB free) — summarizing into a new chat…`;
    showRamNotice(notice);
    setStatus(notice, false);
    abortGenerationForRam();
    setChatBusy(true);
    showSummarizeLoader(notice);
    const gen = beginGeneration();
    try {
      await new Promise((r) => setTimeout(r, 50));
      if (!current) {
        setStatus(
          `Low free RAM (${freeGib.toFixed(1)} GiB): nothing open to summarize.`,
          false
        );
        return;
      }
      annotateStoppedForRam();
      renderTranscript();
      await saveCurrent();
      const sourceTitle = current.title || "chat";
      const sourceMessages = JSON.parse(JSON.stringify(current.messages || []));
      if (filterWorkingMessages(sourceMessages).length < 2) {
        setStatus(
          `Low free RAM (${freeGib.toFixed(1)} GiB): generation stopped. ` +
            "Little history left to summarize into a new chat.",
          false
        );
        return;
      }
      const forked = await forkSummarizedChat(sourceMessages, sourceTitle, {
        reason: "ram",
        signal: gen.signal,
      });
      if (forked.ok) {
        const done =
          `Low RAM (${freeGib.toFixed(1)} GiB free) — opened summarized new chat. ` +
          "Original left unchanged.";
        showRamNotice(done);
        setStatus(done, false);
        warnIfHuge(apiMessages());
      } else {
        setStatus(
          `Low free RAM (${freeGib.toFixed(1)} GiB): generation stopped. ` +
            "Nothing left to compress for a new chat.",
          false
        );
      }
    } catch (err) {
      if (!isAbortError(err)) {
        setStatus(`Low free RAM fallback failed: ${err.message}`, false);
      }
    } finally {
      endGeneration(gen);
      hideSummarizeLoader();
      ramFallbackRunning = false;
      ramAbortRequested = false;
      setChatBusy(false);
    }
  }

  function isContextLengthError(message) {
    const m = String(message || "");
    return /context_length_exceeded|configured context size|Prompt has \d+ tokens/i.test(m);
  }

  async function refreshModels() {
    try {
      const data = await api("/v1/models");
      const row = data?.data?.[0];
      if (row?.id) modelId = row.id;
      if (row?.context_length) contextLength = row.context_length;
      let status = `API up · model ${modelId} · ctx ${contextLength.toLocaleString()}`;
      let ok = true;
      if (safeCtx && contextLength > safeCtx) {
        status += ` · above safe ${safeCtx.toLocaleString()} (RAM-6GiB)`;
        ok = false;
      } else if (safeCtx) {
        status += ` · safe≤${safeCtx.toLocaleString()}`;
      }
      setStatus(status, ok);
      if (current) warnIfHuge(apiMessages());
    } catch (err) {
      setStatus(`ds4-server unreachable (${err.message}). Start it on :8000.`, false);
    }
  }

  async function refreshList() {
    const data = await api("/api/chats");
    const chats = data.chats || [];
    els.chatList.innerHTML = "";
    if (!chats.length) {
      const empty = document.createElement("p");
      empty.className = "m";
      empty.style.padding = "0.5rem";
      empty.textContent = "No saved chats yet.";
      els.chatList.appendChild(empty);
    }
    for (const chat of chats) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.dataset.id = chat.id;
      btn.className = "chat-item" + (current && current.id === chat.id ? " active" : "");
      btn.innerHTML = `<span class="t">${escapeHtml(chat.title)}</span><span class="m">${escapeHtml(chatListMeta(chat))}</span>`;
      btn.addEventListener("click", () => openChat(chat.id));
      els.chatList.appendChild(btn);
    }
  }

  function renderTranscript() {
    els.transcript.innerHTML = "";
    if (!current || !current.messages.length) {
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.innerHTML = `<strong>DwarfStar</strong>Start a conversation. Attach text, code, images, or PDFs. Images/PDFs are OCR’d to text locally. Chats autosave under ~/.ds4/chats.`;
      els.transcript.appendChild(empty);
      return;
    }
    for (const msg of current.messages) {
      els.transcript.appendChild(renderMessage(msg));
    }
    els.transcript.scrollTop = els.transcript.scrollHeight;
  }

  function renderMessage(msg, streaming) {
    const wrap = document.createElement("article");
    wrap.className = `msg ${msg.role}` + (isSummaryMessage(msg) ? " summary" : "");
    const role = document.createElement("p");
    role.className = "role";
    role.textContent = isSummaryMessage(msg) ? "summary" : msg.role;
    wrap.appendChild(role);
    if (msg.reasoning) {
      wrap.appendChild(createThinkMirror(msg, msg.reasoning));
    }
    const split = extractWebContextBlock(msg.content || "");
    if (split.web) {
      wrap.appendChild(createWebMirror(msg, split.web));
    }
    const displayContent = split.web ? split.rest : msg.content || "";
    const bubble = document.createElement("div");
    bubble.className = "bubble bubble-main" + (streaming ? " streaming" : "");
    const body = document.createElement("div");
    body.className = "bubble-body";
    body.dataset.raw = displayContent;
    const preview = document.createElement("div");
    preview.className = "md-preview";
    preview.innerHTML = renderMarkdownHtml(displayContent || (streaming ? "" : ""));
    body.appendChild(preview);
    const actions = document.createElement("div");
    actions.className = "bubble-actions";
    actions.appendChild(makeSpeakButton(() => messagePreviewText(wrap, body)));
    actions.appendChild(makeCopyButton(() => messageCopyText(wrap, body)));
    bubble.appendChild(actions);
    bubble.appendChild(body);
    wrap.appendChild(bubble);
    if (msg.files && msg.files.length) {
      const files = document.createElement("div");
      files.className = "files";
      for (const f of msg.files) {
        const chip = document.createElement("span");
        chip.className = "file-chip";
        chip.textContent = f;
        files.appendChild(chip);
      }
      wrap.appendChild(files);
    }
    return wrap;
  }

  function renderAttachBar() {
    if (!pendingFiles.length) {
      els.attachBar.hidden = true;
      els.attachBar.innerHTML = "";
      return;
    }
    els.attachBar.hidden = false;
    els.attachBar.innerHTML = "";
    pendingFiles.forEach((file, idx) => {
      const chip = document.createElement("span");
      chip.className = "attach-chip";
      const tag = file.source === "ocr"
        ? `OCR${file.method === "text" ? "/text" : ""}`
        : "text";
      chip.innerHTML = `<em>${escapeHtml(tag)}</em> ${escapeHtml(file.name)} <button type="button" aria-label="Remove">×</button>`;
      chip.querySelector("button").addEventListener("click", () => {
        pendingFiles.splice(idx, 1);
        renderAttachBar();
      });
      els.attachBar.appendChild(chip);
    });
  }

  function buildUserContent(text, files, webContext) {
    const parts = [];
    if (webContext) {
      parts.push(String(webContext).trim());
      parts.push("");
    }
    parts.push((text || "").trim());
    for (const f of files) {
      if (f.source === "ocr") {
        const via = f.method === "text" ? "extracted text" : "OCR";
        parts.push(
          `\n\n----- attached ${f.kind || "file"} (${via}): ${f.name} -----\n${f.text}\n----- end ${f.name} -----`
        );
      } else {
        parts.push(
          `\n\n----- attached file: ${f.name} -----\n${f.text}\n----- end file: ${f.name} -----`
        );
      }
    }
    return parts.join("\n").trim();
  }

  function recentSearchMessages() {
    const msgs = (current && current.messages) || [];
    const out = [];
    for (const m of msgs) {
      if (m.role !== "user" && m.role !== "assistant") continue;
      if (m.role === "assistant" && isUiFailureAssistant(m)) continue;
      let content = String(m.content || "");
      content = content.replace(
        /----- web context[\s\S]*?----- end web context -----\s*/gi,
        ""
      );
      content = content.replace(
        /----- attached[\s\S]*?----- end [^\n-]+ -----\s*/gi,
        ""
      );
      content = content.trim();
      if (!content) continue;
      if (content.length > 800) content = content.slice(0, 800);
      out.push({ role: m.role, content });
    }
    return out.slice(-8);
  }

  async function fetchWebContext(query, signal) {
    const data = await api("/api/web-context", {
      method: "POST",
      body: JSON.stringify({
        query,
        messages: recentSearchMessages(),
        model: modelId,
        max_results: 8,
        max_fetch: 5,
        fetch_pages: true,
      }),
      signal,
    });
    if (data && data.query) setWebSearchLoaderQuery(data.query);
    return data;
  }

  function setTtsSpeaking(active) {
    if (els.ttsStopBtn) els.ttsStopBtn.disabled = !active;
  }

  function stopSpeaking() {
    ttsToken += 1;
    if (ttsAudio) {
      try {
        ttsAudio.pause();
        ttsAudio.removeAttribute("src");
        ttsAudio.load();
      } catch {
        /* ignore */
      }
      ttsAudio = null;
    }
    setTtsSpeaking(false);
  }

  async function speakFinal(text, opts = {}) {
    const force = !!opts.force;
    if (!force && (!els.ttsToggle || !els.ttsToggle.checked)) return;
    if (!ttsAvailable) {
      if (force || (els.ttsToggle && els.ttsToggle.checked)) {
        setStatus(
          "TTS unavailable — restart chat-ui so /api/tts is loaded, then hard-refresh.",
          false
        );
      }
      return;
    }
    const spoken = String(text || "").trim();
    if (!spoken || spoken.startsWith("Error:")) return;

    stopSpeaking();
    const token = ttsToken;
    setTtsSpeaking(true);
    setStatus("Speaking…", true);
    try {
      await unlockAudio();
      if (token !== ttsToken) return;
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: spoken }),
      });
      if (token !== ttsToken) return;
      if (!res.ok) {
        const errText = await res.text();
        if (res.status === 404) {
          throw new Error(" /api/tts missing — restart chat-ui, then hard-refresh");
        }
        throw new Error(parseErrorPayload(errText) || res.statusText);
      }
      const blob = await res.blob();
      if (token !== ttsToken) return;
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      ttsAudio = audio;
      audio.onended = () => {
        URL.revokeObjectURL(url);
        if (ttsAudio === audio) {
          ttsAudio = null;
          setTtsSpeaking(false);
          setStatus(`API up · model ${modelId} · ctx ${contextLength}`, true);
        }
      };
      audio.onerror = () => {
        URL.revokeObjectURL(url);
        if (ttsAudio === audio) {
          ttsAudio = null;
          setTtsSpeaking(false);
        }
      };
      await audio.play();
    } catch (err) {
      if (token !== ttsToken) return;
      setTtsSpeaking(false);
      const name = err && err.name;
      if (name === "NotAllowedError") {
        setStatus(
          "Browser blocked autoplay — click the speaker icon on a message to read it.",
          false
        );
      } else {
        setStatus(`TTS failed: ${err.message}`, false);
      }
    }
  }

  async function saveCurrent() {
    if (!current) return;
    if (!current.title || current.title === "New chat") {
      const firstUser = current.messages.find((m) => m.role === "user");
      if (firstUser) {
        const line = firstUser.content.split("\n")[0].trim();
        current.title = (line.slice(0, 48) || "Untitled").replace(/\s+/g, " ");
        els.chatTitle.textContent = current.title;
      }
    }
    current = await api(`/api/chats/${current.id}`, {
      method: "PUT",
      body: JSON.stringify(current),
    });
    await refreshList();
  }

  async function createChat() {
    current = await api("/api/chats", {
      method: "POST",
      body: JSON.stringify({ title: "New chat" }),
    });
    pendingFiles = [];
    renderAttachBar();
    els.chatTitle.textContent = current.title;
    renderTranscript();
    await refreshList();
    els.prompt.focus();
  }

  async function openChat(id) {
    current = await api(`/api/chats/${id}`);
    pendingFiles = [];
    renderAttachBar();
    els.chatTitle.textContent = current.title;
    renderTranscript();
    await refreshList();
    warnIfHuge(apiMessages());
    els.rail.classList.remove("open");
  }

  async function renameChat() {
    if (!current) return;
    const next = window.prompt("Conversation title", current.title);
    if (next == null) return;
    current.title = next.trim() || current.title;
    els.chatTitle.textContent = current.title;
    await saveCurrent();
  }

  async function deleteChat() {
    if (!current) return;
    if (!window.confirm(`Delete “${current.title}”?`)) return;
    const id = current.id;
    await api(`/api/chats/${id}`, { method: "DELETE" });
    current = null;
    await refreshList();
    const data = await api("/api/chats");
    if (data.chats && data.chats[0]) await openChat(data.chats[0].id);
    else await createChat();
  }

  function apiMessages() {
    return (current.messages || [])
      .filter((m) => m.role === "system" || m.role === "user" || m.role === "assistant")
      .filter((m) => !isUiFailureAssistant(m))
      .map((m) => ({ role: m.role, content: m.content || "" }));
  }

  function bytesToBase64(bytes) {
    let binary = "";
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
    }
    return btoa(binary);
  }

  async function ocrUpload(file) {
    const buf = new Uint8Array(await file.arrayBuffer());
    const data = await api("/api/ocr", {
      method: "POST",
      body: JSON.stringify({
        filename: file.name,
        data_base64: bytesToBase64(buf),
      }),
    });
    return {
      name: data.filename || file.name,
      text: data.text || "",
      bytes: (data.text || "").length,
      source: "ocr",
      method: data.method || "ocr",
      kind: data.kind || "file",
    };
  }

  async function sendMessage() {
    if (busy || !current) return;
    const text = els.prompt.value.trim();
    if (!text && !pendingFiles.length) return;

    if (els.ttsToggle && els.ttsToggle.checked) {
      await unlockAudio();
    }

    const files = pendingFiles.slice();
    const useWeb = !!(els.webToggle && els.webToggle.checked);
    if (useWeb && !text) {
      window.alert("Web search needs a text query in the message box.");
      return;
    }

    setChatBusy(true);
    stopSpeaking();
    const gen = beginGeneration();
    const signal = gen.signal;
    setActivity("Preparing…");

    let webContext = "";
    let webMeta = null;
    try {
      if (useWeb) {
        startWebActivityPhases();
        webMeta = await fetchWebContext(text, signal);
        webContext = (webMeta && webMeta.context) || "";
        if (!webContext) {
          throw new Error("web search returned empty context");
        }
        const n = (webMeta.results && webMeta.results.length) || 0;
        const pages = webMeta.pages_fetched || 0;
        const tried = webMeta.pages_attempted || pages;
        const fails = (webMeta.errors && webMeta.errors.length) || 0;
        hideWebSearchLoader();
        const qLabel = (webMeta.query || "").trim();
        setStatus(
          `Web ok` +
            (qLabel ? ` · ${qLabel}` : "") +
            ` · ${n} results · ${pages}/${tried} pages` +
            (fails ? ` · ${fails} fetch issues` : ""),
          true
        );
      }
    } catch (err) {
      hideWebSearchLoader();
      endGeneration(gen);
      if (ramAbortRequested || isAbortError(err)) {
        if (!ramFallbackRunning) setChatBusy(false);
        return;
      }
      setChatBusy(false);
      setStatus(`Web failed: ${err.message}`, false);
      window.alert(`Web search failed: ${err.message}`);
      return;
    }

    const content = buildUserContent(text, files, webContext);
    const userMsg = {
      role: "user",
      content,
      files: files.map((f) => {
        if (f.source === "ocr") {
          const via = f.method === "text" ? "text" : "OCR";
          return `${f.name} (${via})`;
        }
        return f.name;
      }),
    };
    if (useWeb) {
      userMsg.web = {
        provider: (webMeta && webMeta.provider) || "duckduckgo-html",
        results: (webMeta && webMeta.results) || [],
      };
    }

    try {
      setActivity("Preparing…");
      const compressed = await compressHistoryIfNeeded(content, { signal });
      if (compressed.compressed) {
        setActivity("History compressed — continuing…");
      } else {
        setActivity("Sending…");
      }
      const preview = workingMessages().concat([{ role: "user", content }]);
      const est = historyTokenEstimate(preview);
      if (compressed.stillHuge || est >= Math.floor(contextLength * CONTEXT_HARD_RATIO)) {
        const proceed = window.confirm(
          `Still ~${est.toLocaleString()} tokens after auto-summarize ` +
            `(ctx ${contextLength.toLocaleString()}). Send anyway?`
        );
        if (!proceed) {
          clearActivity();
          endGeneration(gen);
          setChatBusy(false);
          return;
        }
      }
    } catch (err) {
      clearActivity();
      endGeneration(gen);
      if (ramAbortRequested || isAbortError(err)) {
        if (!ramFallbackRunning) setChatBusy(false);
        return;
      }
      setChatBusy(false);
      setStatus(`Auto-summarize failed: ${err.message}`, false);
      return;
    }

    current.messages.push(userMsg);
    els.prompt.value = "";
    pendingFiles = [];
    renderAttachBar();
    renderTranscript();
    await saveCurrent();

    const assistantMsg = { role: "assistant", content: "", reasoning: "" };
    current.messages.push(assistantMsg);
    let node = renderMessage(assistantMsg, true);
    els.transcript.appendChild(node);
    // Bubble is visible — keep phases on the status line only.
    setActivity("Waiting for the model…", { transcript: false });

    try {
      await streamAssistantInto(assistantMsg, node, signal);
    } catch (err) {
      if (ramAbortRequested || isAbortError(err)) {
        clearActivity();
        const bubble = contentBubbleEl(node);
        if (bubble) bubble.classList.remove("streaming");
        // handleRamPressure owns status, cleanup, and summarize
        if (!ramFallbackRunning) setChatBusy(false);
        endGeneration(gen);
        return;
      }
      if (isContextLengthError(err.message)) {
        try {
          setActivity("Hit context limit — summarizing…", { transcript: false });
          showSummarizeLoader("Context limit — summarizing…");
          if (current.messages[current.messages.length - 1] === assistantMsg) {
            current.messages.pop();
          }
          await compressHistoryIfNeeded("", { signal });
          hideSummarizeLoader();
          assistantMsg.content = "";
          delete assistantMsg.reasoning;
          current.messages.push(assistantMsg);
          renderTranscript();
          const fresh = renderMessage(assistantMsg, true);
          const last = els.transcript.lastElementChild;
          if (last) last.replaceWith(fresh);
          else els.transcript.appendChild(fresh);
          node = fresh;
          await saveCurrent();
          setActivity("Waiting for the model…", { transcript: false });
          await streamAssistantInto(assistantMsg, node, signal);
        } catch (retryErr) {
          hideSummarizeLoader();
          clearActivity();
          if (ramAbortRequested || isAbortError(retryErr)) {
            const bubble = contentBubbleEl(node);
            if (bubble) bubble.classList.remove("streaming");
            if (!ramFallbackRunning) setChatBusy(false);
            endGeneration(gen);
            return;
          }
          stopSpeaking();
          assistantMsg.content = `Error: ${retryErr.message}`;
          const bubble = contentBubbleEl(node);
          if (bubble) {
            bubble.classList.remove("streaming");
            setBubbleText(bubble, assistantMsg.content);
          }
          await saveCurrent();
          setStatus(`Request failed: ${retryErr.message}`, false);
          setChatBusy(false);
          endGeneration(gen);
          els.prompt.focus();
          return;
        }
      } else {
        clearActivity();
        stopSpeaking();
        assistantMsg.content = `Error: ${err.message}`;
        const bubble = contentBubbleEl(node);
        if (bubble) {
          bubble.classList.remove("streaming");
          setBubbleText(bubble, assistantMsg.content);
        }
        await saveCurrent();
        setStatus(`Request failed: ${err.message}`, false);
        setChatBusy(false);
        endGeneration(gen);
        els.prompt.focus();
        return;
      }
    }

    endGeneration(gen);
    setChatBusy(false);
    els.prompt.focus();
  }

  async function streamAssistantInto(assistantMsg, node, signal) {
    let thinkEl = node.querySelector(".bubble.think");
    const bubble = contentBubbleEl(node);
    if (!bubble) throw new Error("missing assistant bubble");

    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({
        model: modelId,
        messages: apiMessages().slice(0, -1),
        stream: true,
        temperature: 0.7,
      }),
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(parseErrorPayload(errText) || res.statusText);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n");
      buffer = parts.pop() || "";
      for (const line of parts) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === "[DONE]") continue;
        try {
          const json = JSON.parse(payload);
          const delta = json.choices?.[0]?.delta || {};
          if (delta.reasoning_content) {
            assistantMsg.reasoning = (assistantMsg.reasoning || "") + delta.reasoning_content;
            if (!thinkEl) {
              thinkEl = createThinkMirror(assistantMsg, assistantMsg.reasoning);
              node.insertBefore(thinkEl, bubble);
            } else {
              setThinkMirrorText(thinkEl, assistantMsg.reasoning);
            }
            els.transcript.scrollTop = els.transcript.scrollHeight;
          }
          if (delta.content) {
            assistantMsg.content += delta.content;
            setBubbleText(bubble, assistantMsg.content);
            els.transcript.scrollTop = els.transcript.scrollHeight;
          }
        } catch {
          /* ignore partial SSE JSON */
        }
      }
    }
    if (!assistantMsg.content && !assistantMsg.reasoning) {
      assistantMsg.content = "(empty reply)";
    } else if (!assistantMsg.content && assistantMsg.reasoning) {
      assistantMsg.content = "";
    }
    if (!assistantMsg.reasoning) delete assistantMsg.reasoning;
    bubble.classList.remove("streaming");
    setBubbleText(bubble, assistantMsg.content);
    await saveCurrent();
    setStatus(`API up · model ${modelId} · ctx ${contextLength}`, true);
    const spoken = messagePreviewText(node, bubble.querySelector(".bubble-body") || bubble);
    if (els.ttsToggle && els.ttsToggle.checked) {
      if (spoken) {
        await speakFinal(spoken);
      } else {
        setStatus("Read aloud skipped: empty answer text (thinking-only).", null);
      }
    }
  }

  function looksBinary(bytes) {
    const n = Math.min(bytes.length, 4096);
    let weird = 0;
    for (let i = 0; i < n; i++) {
      const b = bytes[i];
      if (b === 0) return true;
      if (b < 7 || (b > 13 && b < 32)) weird++;
    }
    return weird / n > 0.3;
  }

  async function onFilesSelected(fileList) {
    let total = pendingFiles.reduce((s, f) => s + f.bytes, 0);
    for (const file of Array.from(fileList || [])) {
      const ext = (file.name.split(".").pop() || "").toLowerCase();

      if (OCR_EXT.has(ext) || (file.type || "").startsWith("image/") || file.type === "application/pdf") {
        if (file.size > MAX_OCR_UPLOAD) {
          window.alert(`${file.name} is larger than 12 MiB.`);
          continue;
        }
        setStatus(`OCR running on ${file.name}…`, null);
        try {
          const row = await ocrUpload(file);
          if (!row.text) {
            window.alert(`${file.name}: OCR returned empty text.`);
            continue;
          }
          if (total + row.bytes > MAX_TOTAL_ATTACH) {
            window.alert("Attached text would exceed ~1.5 MiB for this turn.");
            break;
          }
          pendingFiles.push(row);
          total += row.bytes;
          setStatus(`OCR ok · ${file.name} · ${row.chars || row.bytes} chars`, true);
        } catch (err) {
          window.alert(`${file.name}: ${err.message}`);
          setStatus(`OCR failed: ${err.message}`, false);
        }
        continue;
      }

      if (file.size > MAX_FILE_BYTES) {
        window.alert(`${file.name} is larger than 512 KiB.`);
        continue;
      }
      if (total + file.size > MAX_TOTAL_ATTACH) {
        window.alert("Attached text would exceed ~1.5 MiB for this turn.");
        break;
      }
      const buf = new Uint8Array(await file.arrayBuffer());
      const isText =
        TEXT_EXT.has(ext) ||
        file.type === "application/json" ||
        (file.type || "").startsWith("text");
      if (!isText || looksBinary(buf)) {
        window.alert(
          `${file.name}: not a supported attachment. Use text/code, or image/PDF for OCR.`
        );
        continue;
      }
      const text = new TextDecoder("utf-8", { fatal: false }).decode(buf);
      pendingFiles.push({ name: file.name, text, bytes: file.size, source: "text" });
      total += file.size;
    }
    els.fileInput.value = "";
    renderAttachBar();
  }

  els.newChatBtn.addEventListener("click", () => createChat());
  els.renameBtn.addEventListener("click", () => renameChat());
  els.deleteBtn.addEventListener("click", () => deleteChat());
  if (els.summarizeBtn) {
    els.summarizeBtn.addEventListener("click", () => summarizeInPlace());
  }
  if (els.summarizeForkBtn) {
    els.summarizeForkBtn.addEventListener("click", () => summarizeToNewChat());
  }
  els.sendBtn.addEventListener("click", () => sendMessage());
  els.fileInput.addEventListener("change", (e) => onFilesSelected(e.target.files));
  els.railToggle.addEventListener("click", () => els.rail.classList.toggle("open"));
  if (els.ttsStopBtn) els.ttsStopBtn.addEventListener("click", () => stopSpeaking());
  if (els.ttsToggle) {
    try {
      els.ttsToggle.checked = localStorage.getItem("ds4-tts-read-aloud") === "1";
    } catch {
      /* private mode */
    }
    els.ttsToggle.addEventListener("change", async () => {
      try {
        localStorage.setItem("ds4-tts-read-aloud", els.ttsToggle.checked ? "1" : "0");
      } catch {
        /* ignore */
      }
      if (!els.ttsToggle.checked) stopSpeaking();
      else await unlockAudio();
    });
  }
  els.prompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  async function pollRam() {
    try {
      const ram = await api("/api/ram");
      if (!ram?.available) throw new Error(ram?.error || "unavailable");
      const used = Number(ram.used_gib);
      const total = Number(ram.total_gib);
      const pct = Number(ram.percent);
      let freeGib =
        ram.free_gib != null
          ? Number(ram.free_gib)
          : ram.free_bytes != null
            ? Number(ram.free_bytes) / (1024 ** 3)
            : Number.NaN;
      if (!Number.isFinite(freeGib) && Number.isFinite(used) && Number.isFinite(total)) {
        freeGib = total - used;
      }
      if (ram.pressure_trigger_gib != null) {
        ramTriggerGib = Number(ram.pressure_trigger_gib);
      }
      if (ram.pressure_clear_gib != null) {
        ramClearGib = Number(ram.pressure_clear_gib);
      }
      const evalFree =
        wantForceRamPressure() && !ramPressureLatched
          ? Math.min(freeGib, ramTriggerGib)
          : freeGib;
      if (els.ramStatus && els.ramMeterFill) {
        const shown = Number.isFinite(freeGib) ? freeGib : 0;
        els.ramStatus.textContent = `${used.toFixed(1)} / ${total.toFixed(1)} GiB (${pct.toFixed(0)}%) · ${shown.toFixed(1)} free`;
        els.ramMeterFill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
      }
      const action = evaluateRamPressure(evalFree, ramPressureLatched);
      logRamPressureEval(evalFree, action);
      if (action === "trigger") {
        ramPressureLatched = true;
        void handleRamPressure(Number.isFinite(evalFree) ? evalFree : ramTriggerGib);
      } else if (action === "clear") {
        ramPressureLatched = false;
      }
    } catch {
      if (els.ramStatus) els.ramStatus.textContent = "RAM —";
      if (els.ramMeterFill) els.ramMeterFill.style.width = "0%";
    }
  }

  async function boot() {
    pollRam();
    setInterval(pollRam, 2000);
    try {
      const health = await api("/api/health");
      if (health?.ram_policy?.available && health.ram_policy.safe_ctx) {
        safeCtx = health.ram_policy.safe_ctx;
      }
      const tts = health?.tts;
      ttsAvailable = !!(tts && tts.available);
      if (els.ttsToggle) {
        if (!ttsAvailable) {
          els.ttsToggle.disabled = true;
          els.ttsToggle.checked = false;
          els.ttsToggle.title = tts
            ? "Local TTS tools missing on this host"
            : "Restart chat-ui to enable /api/tts, then hard-refresh";
        } else {
          els.ttsToggle.disabled = false;
          els.ttsToggle.title =
            "Read assistant answers aloud (Piper preferred; macOS say fallback)";
        }
      }
      if (els.composerNote) {
        const ocr = health?.ocr;
        const webBit = health?.web?.enabled
          ? " Toggle Web for free DuckDuckGo context (no API key)."
          : "";
        const ttsBit = ttsAvailable
          ? " Toggle Read aloud for new answers, or use the speaker icon on any message."
          : tts
            ? " Read aloud needs Piper (make install-piper) or macOS say+afconvert."
            : " Restart chat-ui to enable Read aloud (/api/tts).";
        const summarizeBit =
          " Use Summarize to compress this chat, or Summarize to new chat to fork a summarized copy.";
        if (ocr && !ocr.images) {
          els.composerNote.textContent =
            "Text/code attach works. Image/PDF OCR needs: brew install tesseract poppler." +
            webBit +
            ttsBit +
            summarizeBit;
        } else {
          els.composerNote.textContent =
            "Text/code are inlined. Images and PDFs are OCR’d to text locally (model is text-only)." +
            webBit +
            ttsBit +
            summarizeBit;
        }
      }
    } catch {
      /* health optional at boot */
    }
    await refreshModels();
    const data = await api("/api/chats");
    if (data.chats && data.chats[0]) await openChat(data.chats[0].id);
    else await createChat();
    setInterval(refreshModels, 15000);
  }

  boot().catch((err) => {
    setStatus(`UI boot failed: ${err.message}`, false);
  });
})();
