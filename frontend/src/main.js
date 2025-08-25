// Vite entry for code_descriptor (vanilla)
// If you have a stylesheet, keep this import (safe if missing will be ignored by Vite during dev, but best to add src/style.css).
try { await import('./style.css'); } catch (_) {}

let lastProjectDir = '';
// UI elements
const dirSection = document.getElementById('dirSection');
const uploadSection = document.getElementById('uploadSection');
const projectDirInput = document.getElementById('projectDirInput');
const uploadBtn = document.getElementById('uploadBtn');
const runBtn = document.getElementById('runBtn');
const progressContainer = document.getElementById('progressContainer');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const debugOutput = document.getElementById('debugOutput');

// Log appending with max lines and auto-scroll
const LOG_MAX_LINES = import.meta.env.VITE_LOG_MAX_LINES
  ? parseInt(import.meta.env.VITE_LOG_MAX_LINES, 10)
  : 5000;

function appendLog(line) {
  if (!debugOutput) return;
  const div = document.createElement('div');
  div.textContent = line;
  debugOutput.appendChild(div);
  while (debugOutput.childNodes.length > LOG_MAX_LINES) {
    debugOutput.removeChild(debugOutput.firstChild);
  }
  debugOutput.scrollTop = debugOutput.scrollHeight;
}

function appendLogs(lines) {
  lines.forEach(appendLog);
}

function clearLogs() {
  if (!debugOutput) return;
  debugOutput.innerHTML = '';
}

// Mode switch: dir vs upload
document.querySelectorAll('input[name="mode"]').forEach(radio => {
  radio.addEventListener('change', () => {
    if (radio.value === 'dir') {
      dirSection.style.display = '';
      uploadSection.style.display = 'none';
    } else {
      dirSection.style.display = 'none';
      uploadSection.style.display = '';
    }
  });
});

// Always keep Run enabled by default
if (runBtn) runBtn.disabled = false;

// Upload local files to server
if (uploadBtn) {
  uploadBtn.addEventListener('click', async () => {
    const input = document.getElementById('fileInput');
    const files = input?.files || [];
    const formData = new FormData();
    for (let file of files) formData.append('files', file);
    const res = await fetch('/upload', { method: 'POST', body: formData });
    const result = await res.json();
    if (result.status === 'success') {
      lastProjectDir = result.project_dir || '';
      const run = document.getElementById('runBtn');
      if (run) run.disabled = !lastProjectDir;
    }
    alert('Upload result: ' + JSON.stringify(result));
  });
}

// Build AST handler (streams logs)
if (runBtn) {
  runBtn.addEventListener('click', async () => {
    const project = document.getElementById('projectNameInput').value.trim();
    if (!project) { alert('Please enter a project name.'); return; }
    const projectDir = projectDirInput.value.trim();
    if (!projectDir) { alert('Please enter a project directory.'); return; }

    runBtn.disabled = true;
    clearLogs();
    if (progressContainer) progressContainer.style.display = '';
    if (progressBar) progressBar.value = 0;
    if (progressText) progressText.textContent = '';

    const res = await fetch('/build_ast', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project, project_dir: projectDir }),
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop();
      for (let line of lines) {
        appendLog(line);
        const m = /Processing file\s*(\d+)\/(\d+)/.exec(line);
        if (m && progressBar && progressText) {
          const current = +m[1], total = +m[2];
          const pct = Math.floor((current/total)*100);
          progressBar.value = pct;
          progressText.textContent = `Processing ${current}/${total}`;
        }
      }
    }
    if (progressBar) progressBar.value = 100;
    if (progressText) progressText.textContent = 'AST build completed';
    runBtn.disabled = false;
  });
}

// Build RAG handler (stream logs)
const buildRagBtn = document.getElementById('buildRagBtn');
if (buildRagBtn) {
  buildRagBtn.addEventListener('click', async () => {
    const project = document.getElementById('projectNameInput').value.trim();
    if (!project) { alert('Please enter a project name.'); return; }
    const projectDir = projectDirInput.value.trim();
    if (!projectDir) { alert('Please enter a project directory.'); return; }

    buildRagBtn.disabled = true;
    clearLogs();
    if (progressContainer) progressContainer.style.display = '';
    if (progressBar) progressBar.value = 0;
    if (progressText) progressText.textContent = 'Starting RAG build...';

    console.log('Build RAG clicked');
    appendLog('>>> Build RAG clicked');
    appendLog('>>> Build RAG: sending POST request to /build_rag');
    const force = document.getElementById('forceCheckbox').checked;
    let res;
    try {
    res = await fetch('/build_rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project, project_dir: projectDir, force }),
      });
      appendLog(`>>> Build RAG: response status ${res.status}`);
    } catch (err) {
      appendLog(`>>> Build RAG: fetch error ${err}`);
      buildRagBtn.disabled = false;
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalStatus = null;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          appendLog('>>> Build RAG: stream closed');
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop();
        for (let line of lines) {
          appendLog(line);
          if (line.startsWith('{') && line.endsWith('}')) {
            try {
              finalStatus = JSON.parse(line);
              appendLog('>>> Build RAG: parsed finalStatus ' + JSON.stringify(finalStatus));
            } catch (e) {
              appendLog('>>> Build RAG: finalStatus JSON parse error ' + e);
            }
          }
        }
      }
    } catch (err) {
      appendLog(`>>> Build RAG: stream read error ${err}`);
    }
    if (progressBar) progressBar.value = 100;
    if (progressText) progressText.textContent = 'RAG build completed';
    buildRagBtn.disabled = false;

    if (finalStatus) {
      appendLog(`>>> Build RAG: finalStatus.status=${finalStatus.status}`);
      if (finalStatus.status === 'success') {
        alert(`RAG build succeeded: project=${finalStatus.project}, points=${finalStatus.points}`);
      } else if (finalStatus.status === 'exists') {
        alert(`RAG exists: project=${finalStatus.project}, points=${finalStatus.points}`);
      } else {
        alert('RAG build error: ' + JSON.stringify(finalStatus));
      }
    } else {
      appendLog('>>> Build RAG: no finalStatus received');
      alert('RAG build completed with unknown status');
    }
  });
}

// Chat handler
const askBtn = document.getElementById('askBtn');
if (askBtn) {
  askBtn.addEventListener('click', async () => {
    const project = document.getElementById('chatProjectInput').value.trim();
    if (!project) { alert('Please enter a project name.'); return; }
    const question = document.getElementById('questionInput').value.trim();
    if (!question) { alert('Please enter a question.'); return; }
    const res = await fetch('/ask_rag', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project, question }),
    });
    const data = await res.json();
    if (data.status === 'not_found') {
      alert('Project not found, please build RAG first.');
      return;
    } else if (data.status === 'success') {
      const out = document.getElementById('chatOutput');
      if (out) out.textContent = data.answer;
    } else {
      alert('Error: ' + JSON.stringify(data));
    }
  });
}
