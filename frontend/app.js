let lastProjectDir = '';
// UI elements
const dirSection = document.getElementById('dirSection');
const uploadSection = document.getElementById('uploadSection');
const projectDirInput = document.getElementById('projectDirInput');
const uploadBtn = document.getElementById('uploadBtn');
const runBtn = document.getElementById('runBtn');

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

document.getElementById('runBtn').addEventListener('click', async () => {
    // Determine project_dir from selected mode
    const mode = document.querySelector('input[name="mode"]:checked').value;
    let projectDir = '';
    if (mode === 'dir') {
        projectDir = projectDirInput.value.trim();
        if (!projectDir) {
            alert('Please enter a valid project directory.');
            return;
        }
    } else {
        projectDir = lastProjectDir;
        if (!projectDir) {
            alert('Please upload files first to obtain a project directory.');
            return;
        }
    }
    const res = await fetch('/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_dir: projectDir })
    });
    const result = await res.json();
    alert('Run result: ' + JSON.stringify(result));
});

// Chat handler remains unchanged
document.getElementById('askBtn').addEventListener('click', async () => {
    const question = document.getElementById('questionInput').value;
    const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
    });
    const data = await res.json();
    const out = document.getElementById('chatOutput');
    out.textContent = data.answer || JSON.stringify(data);
});
