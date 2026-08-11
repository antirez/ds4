(() => {
  const MAX_FILE_BYTES = 512 * 1024;
  const MAX_TOTAL_ATTACH = 1.5 * 1024 * 1024;
  const MAX_OCR_UPLOAD = 12 * 1024 * 1024;
  const CONTEXT_WARN_RATIO = 0.85;
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
    fileInput: document.getElementById("fileInput"),
    attachBar: document.getElementById("attachBar"),
    webToggle: document.getElementById("webToggle"),
    ttsToggle: document.getElementById("ttsToggle"),
    ttsStopBtn: document.getElementById("ttsStopBtn"),
    composerNote: document.querySelector(".composer-note"),
  };

  /** @type {{id:string,title:string,messages:any[],created_at?:string,updated_at?:string}|null} */
  let current = null;
  /** @type {{name:string,text:string,bytes:number,source?:string,method?:string}[]} */
  let pendingFiles = [];
  let busy = false;
  let modelId = "deepseek-v4-flash";
  let contextLength = 100000;
  /** @type {HTMLAudioElement|null} */
  let ttsAudio = null;
  let ttsToken = 0;
  let ttsAvailable = true;

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
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
    els.statusLine.textContent = text;
    els.statusLine.style.color = ok === false ? "var(--danger)" : ok ? "var(--teal)" : "var(--muted)";
  }

  function warnIfHuge(messages) {
    const est = historyTokenEstimate(messages);
    const limit = Math.floor(contextLength * CONTEXT_WARN_RATIO);
    if (est >= limit) {
      setStatus(
        `History ~${est.toLocaleString()} tokens (ctx ${contextLength.toLocaleString()}). Trim or start a new chat before send.`,
        false
      );
      return true;
    }
    return false;
  }

  async function refreshModels() {
    try {
      const data = await api("/v1/models");
      const row = data?.data?.[0];
      if (row?.id) modelId = row.id;
      if (row?.context_length) contextLength = row.context_length;
      setStatus(`API up · model ${modelId} · ctx ${contextLength}`, true);
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
      btn.innerHTML = `<span class="t">${escapeHtml(chat.title)}</span><span class="m">${chat.message_count || 0} msgs · ${escapeHtml(fmtTime(chat.updated_at))}</span>`;
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
    wrap.className = `msg ${msg.role}`;
    const role = document.createElement("p");
    role.className = "role";
    role.textContent = msg.role;
    wrap.appendChild(role);
    if (msg.reasoning) {
      const think = document.createElement("div");
      think.className = "bubble think";
      think.textContent = msg.reasoning;
      wrap.appendChild(think);
    }
    const bubble = document.createElement("div");
    bubble.className = "bubble" + (streaming ? " streaming" : "");
    bubble.textContent = msg.content || (streaming ? "" : "");
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

  async function fetchWebContext(query) {
    const data = await api("/api/web-context", {
      method: "POST",
      body: JSON.stringify({
        query,
        max_results: 5,
        max_fetch: 3,
        fetch_pages: true,
      }),
    });
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

  async function speakFinal(text) {
    if (!els.ttsToggle || !els.ttsToggle.checked) return;
    if (!ttsAvailable) return;
    const spoken = String(text || "").trim();
    if (!spoken || spoken.startsWith("Error:")) return;

    stopSpeaking();
    const token = ttsToken;
    setTtsSpeaking(true);
    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: spoken }),
      });
      if (token !== ttsToken) return;
      if (!res.ok) {
        const errText = await res.text();
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
      setStatus(`TTS failed: ${err.message}`, false);
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

    const files = pendingFiles.slice();
    const useWeb = !!(els.webToggle && els.webToggle.checked);
    if (useWeb && !text) {
      window.alert("Web search needs a text query in the message box.");
      return;
    }

    busy = true;
    els.sendBtn.disabled = true;
    stopSpeaking();

    let webContext = "";
    let webMeta = null;
    try {
      if (useWeb) {
        setStatus("Web search running…", null);
        webMeta = await fetchWebContext(text);
        webContext = (webMeta && webMeta.context) || "";
        if (!webContext) {
          throw new Error("web search returned empty context");
        }
        const n = (webMeta.results && webMeta.results.length) || 0;
        setStatus(`Web ok · ${n} results · ${webMeta.pages_fetched || 0} pages`, true);
      }
    } catch (err) {
      busy = false;
      els.sendBtn.disabled = false;
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

    // Preview size after this turn (exclude prior UI Error assistants).
    const preview = apiMessages().concat([{ role: "user", content }]);
    if (warnIfHuge(preview)) {
      const proceed = window.confirm(
        "This turn looks close to or over the model context window. Send anyway?"
      );
      if (!proceed) {
        busy = false;
        els.sendBtn.disabled = false;
        return;
      }
    }

    current.messages.push(userMsg);
    els.prompt.value = "";
    pendingFiles = [];
    renderAttachBar();
    renderTranscript();
    await saveCurrent();

    const assistantMsg = { role: "assistant", content: "", reasoning: "" };
    current.messages.push(assistantMsg);
    const node = renderMessage(assistantMsg, true);
    els.transcript.appendChild(node);
    let thinkEl = node.querySelector(".bubble.think");
    const bubble = node.querySelector(".bubble:not(.think)");

    try {
      const res = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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
                thinkEl = document.createElement("div");
                thinkEl.className = "bubble think";
                node.insertBefore(thinkEl, bubble);
              }
              thinkEl.textContent = assistantMsg.reasoning;
              els.transcript.scrollTop = els.transcript.scrollHeight;
            }
            if (delta.content) {
              assistantMsg.content += delta.content;
              bubble.textContent = assistantMsg.content;
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
      bubble.textContent = assistantMsg.content;
      await saveCurrent();
      setStatus(`API up · model ${modelId} · ctx ${contextLength}`, true);
      // Speak final answer text only (not reasoning dumps).
      speakFinal(assistantMsg.content);
    } catch (err) {
      stopSpeaking();
      assistantMsg.content = `Error: ${err.message}`;
      bubble.classList.remove("streaming");
      bubble.textContent = assistantMsg.content;
      await saveCurrent();
      setStatus(`Request failed: ${err.message}`, false);
    } finally {
      busy = false;
      els.sendBtn.disabled = false;
      els.prompt.focus();
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
    els.ttsToggle.addEventListener("change", () => {
      try {
        localStorage.setItem("ds4-tts-read-aloud", els.ttsToggle.checked ? "1" : "0");
      } catch {
        /* ignore */
      }
      if (!els.ttsToggle.checked) stopSpeaking();
    });
  }
  els.prompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  async function boot() {
    try {
      const health = await api("/api/health");
      const tts = health?.tts;
      ttsAvailable = !tts || tts.available !== false;
      if (els.ttsToggle && !ttsAvailable) {
        els.ttsToggle.disabled = true;
        els.ttsToggle.checked = false;
        els.ttsToggle.title = "Local TTS unavailable on this host";
      }
      if (els.composerNote) {
        const ocr = health?.ocr;
        const webBit = health?.web?.enabled
          ? " Toggle Web for free DuckDuckGo context (no API key)."
          : "";
        const ttsBit = ttsAvailable
          ? " Toggle Read aloud for local TTS (macOS say; optional Piper via DS4_PIPER_MODEL)."
          : " Read aloud needs macOS say+afconvert (or Piper).";
        if (ocr && !ocr.images) {
          els.composerNote.textContent =
            "Text/code attach works. Image/PDF OCR needs: brew install tesseract poppler." +
            webBit +
            ttsBit;
        } else {
          els.composerNote.textContent =
            "Text/code are inlined. Images and PDFs are OCR’d to text locally (model is text-only)." +
            webBit +
            ttsBit;
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
