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
runBtn.disabled = false;
document.getElementById('uploadBtn').addEventListener('click', async () => {
    // Upload local files to server
    const input = document.getElementById('fileInput');
    const files = input.files;
    const formData = new FormData();
    for (let file of files) {
        formData.append('files', file);
    }
    const res = await fetch('/upload', { method: 'POST', body: formData });
    const result = await res.json();
    if (result.status === 'success') {
        lastProjectDir = result.project_dir || '';
        document.getElementById('runBtn').disabled = !lastProjectDir;
    }
    alert('Upload result: ' + JSON.stringify(result));
});

// Build AST handler
runBtn.addEventListener('click', async () => {
    const project = document.getElementById('projectNameInput').value.trim();
    if (!project) { alert('Please enter a project name.'); return; }
    const projectDir = projectDirInput.value.trim();
    if (!projectDir) { alert('Please enter a project directory.'); return; }

    runBtn.disabled = true;
    debugOutput.textContent = '';
    progressContainer.style.display = '';
    progressBar.value = 0;
    progressText.textContent = '';

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
            debugOutput.textContent += line + '\n';
            const m = /Processing file\s*(\d+)\/(\d+)/.exec(line);
            if (m) {
                const current = +m[1], total = +m[2];
                const pct = Math.floor((current/total)*100);
                progressBar.value = pct;
                progressText.textContent = `Processing ${current}/${total}`;
            }
        }
    }
    progressBar.value = 100;
    progressText.textContent = 'AST build completed';
    runBtn.disabled = false;
});

// Build RAG handler (stream logs)
const buildRagBtn = document.getElementById('buildRagBtn');
buildRagBtn.addEventListener('click', async () => {
    const project = document.getElementById('projectNameInput').value.trim();
    if (!project) { alert('Please enter a project name.'); return; }
    const projectDir = projectDirInput.value.trim();
    if (!projectDir) { alert('Please enter a project directory.'); return; }

    // Prepare UI for streaming logs
    buildRagBtn.disabled = true;
    debugOutput.textContent = '';
    progressContainer.style.display = '';
    progressBar.value = 0;
    progressText.textContent = 'Starting RAG build...';

    console.log('Build RAG clicked');
    debugOutput.textContent += '>>> Build RAG clicked\n';
    debugOutput.textContent += '>>> Build RAG: sending request to /build_rag\n';
    const force = document.getElementById('forceCheckbox').checked;
    let res;
    try {
        res = await fetch('/build_rag', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ project, project_dir: projectDir, force }),
        });
        debugOutput.textContent += `>>> Build RAG: response status ${res.status}\n`;
    } catch (err) {
        debugOutput.textContent += `>>> Build RAG: fetch error ${err}\n`;
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
                debugOutput.textContent += '>>> Build RAG: stream closed\n';
                break;
            }
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split(/\r?\n/);
            buffer = lines.pop();
            for (let line of lines) {
                debugOutput.textContent += line + '\n';
                // detect final JSON status
                if (line.startsWith('{') && line.endsWith('}')) {
                    try {
                        finalStatus = JSON.parse(line);
                        debugOutput.textContent += '>>> Build RAG: parsed finalStatus ' + JSON.stringify(finalStatus) + '\n';
                    } catch (e) {
                        debugOutput.textContent += '>>> Build RAG: finalStatus JSON parse error ' + e + '\n';
                    }
                }
            }
        }
    } catch (err) {
        debugOutput.textContent += `>>> Build RAG: stream read error ${err}\n`;
    }
    // finalize progress
    progressBar.value = 100;
    progressText.textContent = 'RAG build completed';
    buildRagBtn.disabled = false;

    // show alert based on final status
    if (finalStatus) {
        debugOutput.textContent += `>>> Build RAG: finalStatus.status=${finalStatus.status}\n`;
        if (finalStatus.status === 'success') {
            alert(`RAG build succeeded: project=${finalStatus.project}, points=${finalStatus.points}`);
        } else if (finalStatus.status === 'exists') {
            alert(`RAG exists: project=${finalStatus.project}, points=${finalStatus.points}`);
        } else {
            alert('RAG build error: ' + JSON.stringify(finalStatus));
        }
    } else {
        debugOutput.textContent += '>>> Build RAG: no finalStatus received\n';
        alert('RAG build completed with unknown status');
    }
});

// Chat handler remains unchanged
document.getElementById('askBtn').addEventListener('click', async () => {
    const project = document.getElementById('chatProjectInput').value.trim();
    if (!project) {
        alert('Please enter a project name.');
        return;
    }
    const question = document.getElementById('questionInput').value.trim();
    if (!question) {
        alert('Please enter a question.');
        return;
    }
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
        document.getElementById('chatOutput').textContent = data.answer;
    } else {
        alert('Error: ' + JSON.stringify(data));
    }
});
