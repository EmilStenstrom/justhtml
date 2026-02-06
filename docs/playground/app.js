/* global loadPyodide */

const STATUS = {
	INIT: "Loading Pyodide…",
	INSTALL: "Installing justhtml…",
	READY: "Ready.",
	RUNNING: "Running…",
};

let pyodide;
let renderFn;
let scheduledRun = null;
let uiEnabled = true;

function isGitHubPages() {
	const host = window.location.hostname;
	return host === "github.io" || host.endsWith(".github.io");
}

function shouldUsePyPI() {
	// Dev override: append ?pypi=1 (or ?pypi=true) to force PyPI install even when
	// running locally. This is useful for testing the GitHub Pages code path.
	const params = new URLSearchParams(window.location.search);
	if (params.has("pypi")) {
		const v = params.get("pypi");
		return (
			v === null ||
			v === "" ||
			v.toLowerCase() === "1" ||
			v.toLowerCase() === "true"
		);
	}
	return isGitHubPages();
}

async function runPythonFile(pyodideInstance, relativePath) {
	const url = new URL(relativePath, window.location.href).toString();
	const res = await fetch(url, { cache: "no-store" });
	if (!res.ok) {
		throw new Error(
			`Failed to fetch Python file: ${relativePath} (${res.status})`,
		);
	}
	const code = await res.text();
	await pyodideInstance.runPythonAsync(code);
}

async function installJusthtmlFromPyPI(pyodideInstance) {
	await pyodideInstance.loadPackage("micropip");
	await runPythonFile(pyodideInstance, "./py/install_latest_justhtml.py");
}

async function installJusthtmlFromLocalRepo(pyodideInstance) {
	// Load the local working tree version of justhtml by fetching the sources
	// from the repo and writing them into Pyodide's virtual filesystem.
	// This requires serving the repository root over HTTP.
	const baseUrl = new URL("/src/justhtml/", window.location.href).toString();
	const files = [
		"__init__.py",
		"__main__.py",
		"constants.py",
		"context.py",
		"encoding.py",
		"entities.py",
		"errors.py",
		"linkify.py",
		"node.py",
		"parser.py",
		"sanitize.py",
		"selector.py",
		"serialize.py",
		"stream.py",
		"transforms.py",
		"transforms_spec.py",
		"tokenizer.py",
		"tokens.py",
		"treebuilder.py",
		"treebuilder_modes.py",
		"treebuilder_utils.py",
	];

	const rootDir = "/justhtml_local/justhtml";
	pyodideInstance.FS.mkdirTree(rootDir);

	for (const file of files) {
		const res = await fetch(`${baseUrl}${file}`, { cache: "no-store" });
		if (!res.ok) {
			throw new Error(
				`Failed to fetch local justhtml source: ${file} (${res.status})`,
			);
		}
		const content = await res.text();
		pyodideInstance.FS.writeFile(`${rootDir}/${file}`, content);
	}

	await runPythonFile(pyodideInstance, "./py/use_local_repo.py");
}

async function installJusthtml(pyodideInstance) {
	// Use released builds on GitHub Pages, otherwise prefer the local working tree.
	if (shouldUsePyPI()) {
		await installJusthtmlFromPyPI(pyodideInstance);
		return;
	}

	await installJusthtmlFromLocalRepo(pyodideInstance);
}

function getRadioValue(name) {
	const el = document.querySelector(`input[name="${name}"]:checked`);
	return el ? el.value : "";
}

function escapeHtml(text) {
	return text
		.replaceAll("&", "&amp;")
		.replaceAll("<", "&lt;")
		.replaceAll(">", "&gt;")
		.replaceAll('"', "&quot;")
		.replaceAll("'", "&#39;");
}

function highlightHtmlTag(tag) {
	// tag includes <...>
	if (tag.startsWith("<!--")) {
		return `<span class="tok-comment">${escapeHtml(tag)}</span>`;
	}

	if (tag.startsWith("<!")) {
		return `<span class="tok-punct">${escapeHtml(tag)}</span>`;
	}

	const inner = tag.slice(1, -1);
	let i = 0;
	let closing = false;
	if (inner.startsWith("/")) {
		closing = true;
		i = 1;
	}

	while (i < inner.length && inner[i] === " ") i += 1;
	const nameStart = i;
	while (i < inner.length && !" \t\n\r\f/>".includes(inner[i])) i += 1;
	const name = inner.slice(nameStart, i);
	const rest = inner.slice(i);

	let out = "";
	out += `<span class="tok-punct">&lt;${closing ? "/" : ""}</span>`;
	out += `<span class="tok-tag">${escapeHtml(name)}</span>`;

	// Highlight attributes in a conservative way (serializer output is well-formed)
	let r = rest;
	// Preserve trailing "/" before '>' if present
	const selfClose = r.trimEnd().endsWith("/");
	if (selfClose) {
		r = r.replace(/\s*\/\s*$/, "");
	}

	// Tokenize attribute region by scanning
	let j = 0;
	while (j < r.length) {
		const ch = r[j];
		if (
			ch === " " ||
			ch === "\t" ||
			ch === "\n" ||
			ch === "\r" ||
			ch === "\f"
		) {
			out += escapeHtml(ch);
			j += 1;
			continue;
		}

		// Attribute name
		const attrStart = j;
		while (j < r.length && !"= \t\n\r\f".includes(r[j])) j += 1;
		const attrName = r.slice(attrStart, j);
		out += `<span class="tok-attr">${escapeHtml(attrName)}</span>`;

		// Whitespace
		while (
			j < r.length &&
			(r[j] === " " ||
				r[j] === "\t" ||
				r[j] === "\n" ||
				r[j] === "\r" ||
				r[j] === "\f")
		) {
			out += escapeHtml(r[j]);
			j += 1;
		}

		if (j < r.length && r[j] === "=") {
			out += `<span class="tok-punct">=</span>`;
			j += 1;
			// Whitespace after '='
			while (
				j < r.length &&
				(r[j] === " " ||
					r[j] === "\t" ||
					r[j] === "\n" ||
					r[j] === "\r" ||
					r[j] === "\f")
			) {
				out += escapeHtml(r[j]);
				j += 1;
			}

			if (j < r.length && (r[j] === '"' || r[j] === "'")) {
				const quote = r[j];
				let k = j + 1;
				while (k < r.length && r[k] !== quote) k += 1;
				const value = r.slice(j, Math.min(k + 1, r.length));
				out += `<span class="tok-string">${escapeHtml(value)}</span>`;
				j = Math.min(k + 1, r.length);
			}
		}
	}

	if (selfClose) {
		out += `<span class="tok-punct"> /</span>`;
	}
	out += `<span class="tok-punct">&gt;</span>`;
	return out;
}

function highlightHtml(source) {
	const parts = source.split(/(<[^>]+>)/g);
	let out = "";
	for (const part of parts) {
		if (part.startsWith("<") && part.endsWith(">")) {
			out += highlightHtmlTag(part);
		} else {
			out += escapeHtml(part);
		}
	}
	return out;
}

function highlightMarkdown(source) {
	// Keep it intentionally minimal: headings, inline code, and link URLs.
	const lines = source.split("\n");
	const outLines = [];

	for (const line of lines) {
		let html = escapeHtml(line);

		// Headings
		const headingMatch = /^(#{1,6})\s+(.*)$/.exec(line);
		if (headingMatch) {
			const hashes = escapeHtml(headingMatch[1]);
			const text = escapeHtml(headingMatch[2]);
			html = `<span class="tok-md-heading">${hashes} ${text}</span>`;
			outLines.push(html);
			continue;
		}

		// Inline code `...`
		html = html.replace(
			/`([^`]+)`/g,
			(_m, code) => `<span class="tok-md-code">${code}</span>`,
		);
		html = html.replaceAll("", "`");

		// Links: highlight the URL part of [text](url)
		html = html.replace(
			/\]\(([^)]+)\)/g,
			(_m, url) => `](<span class="tok-md-link">${escapeHtml(url)}</span>)`,
		);

		outLines.push(html);
	}

	return outLines.join("\n");
}

function setOutput(text, format, ok) {
	const outputEl = document.getElementById("outputCode");
	if (!ok) {
		outputEl.textContent = text;
		return;
	}

	if (format === "html") {
		outputEl.innerHTML = highlightHtml(text);
		return;
	}

	if (format === "markdown") {
		outputEl.innerHTML = highlightMarkdown(text);
		return;
	}

	outputEl.textContent = text;
}

function setErrors(errors) {
	const el = document.getElementById("errors");
	if (!el) return;

	el.innerHTML = "";

	if (!errors || errors.length === 0) {
		const empty = document.createElement("div");
		empty.className = "error-empty";
		empty.textContent = "No errors detected.";
		el.appendChild(empty);
		return;
	}

	for (const err of errors) {
		const row = document.createElement("div");
		row.className = "error-row";

		const loc = document.createElement("div");
		loc.className = "error-loc";
		const l = err.line !== null ? err.line : "?";
		const c = err.column !== null ? err.column : "?";
		loc.textContent = `${l}:${c}`;

		const cat = document.createElement("div");
		cat.className = `error-cat cat-${err.category}`;
		cat.textContent = err.category;

		const msg = document.createElement("div");
		msg.className = "error-msg";
		msg.textContent = err.message;

		row.appendChild(loc);
		row.appendChild(cat);
		row.appendChild(msg);
		el.appendChild(row);
	}
}

function setStatus(text) {
	document.getElementById("status").textContent = text;
}

function formatInitError(err) {
	if (err && typeof err === "object") {
		const name = err.name || (err.constructor ? err.constructor.name : "Error");
		const message = err.message ? String(err.message) : "";
		const stack = err.stack ? String(err.stack) : "";

		let out = `${name}${message ? `: ${message}` : ""}`;
		if (stack && !stack.includes(out)) out += `\n\n${stack}`;

		if (name === "SecurityError") {
			out +=
				"\n\nHint: this usually happens when running from `file://` or when the browser blocks local file access. Serve the repo over HTTP (for example `python -m http.server` from the repo root) and open the playground via `http://localhost/...`.";
		}

		if (
			name === "TypeError" &&
			message.toLowerCase().includes("failed to fetch")
		) {
			out +=
				"\n\nHint: local mode fetches `/src/justhtml/*.py` from the same origin. Make sure you are serving the repository root over HTTP and not only the `docs/` folder.";
		}

		return out;
	}

	return String(err);
}

function setEnabled(enabled) {
	uiEnabled = enabled;
	const ids = [
		"input",
		"selector",
		"safe",
		"cleanup",
		"pretty",
		"indentSize",
		"textSeparator",
		"textStrip",
	];

	for (const id of ids) {
		const el = document.getElementById(id);
		if (el) el.disabled = !enabled;
	}

	for (const el of document.querySelectorAll('input[name="parseMode"]')) {
		el.disabled = !enabled;
	}
	for (const el of document.querySelectorAll('input[name="outputFormat"]')) {
		el.disabled = !enabled;
	}
}

function scheduleRerender() {
	if (!renderFn) return;
	if (!uiEnabled) return;

	if (scheduledRun) clearTimeout(scheduledRun);
	scheduledRun = setTimeout(() => {
		scheduledRun = null;
		void run();
	}, 80);
}

function updateVisibleSettings() {
	const outputFormat = getRadioValue("outputFormat");
	const htmlSettings = document.getElementById("htmlSettings");
	const textSettings = document.getElementById("textSettings");

	htmlSettings.hidden = outputFormat !== "html";
	textSettings.hidden = outputFormat !== "text";
}

function updateFragmentControls() {
	// Fragment mode uses default div context
}

async function initPyodide() {
	setEnabled(false);
	setStatus(STATUS.INIT);

	pyodide = await loadPyodide({
		indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/",
	});

	setStatus(STATUS.INSTALL);
	await installJusthtml(pyodide);

	await runPythonFile(pyodide, "./py/render.py");
	renderFn = pyodide.globals.get("render");
	if (!renderFn) {
		throw new Error(
			"Python function `render` was not defined by ./py/render.py",
		);
	}

	setStatus(STATUS.READY);
	setEnabled(true);

	updateVisibleSettings();
	updateFragmentControls();
	void run();
}

async function run() {
	if (!renderFn) return;

	setStatus(STATUS.RUNNING);
	setEnabled(false);

	const html = document.getElementById("input").value;
	const parseMode = getRadioValue("parseMode");
	const selector = document.getElementById("selector").value.trim();
	const outputFormat = getRadioValue("outputFormat");

	const safe = document.getElementById("safe").checked;
	const cleanup = document.getElementById("cleanup").checked;
	const pretty = document.getElementById("pretty").checked;
	const indentSize = document.getElementById("indentSize").value;

	const textSeparator = document.getElementById("textSeparator").value;
	const textStrip = document.getElementById("textStrip").checked;

	const result = renderFn(
		html,
		parseMode,
		selector,
		outputFormat,
		safe,
		cleanup,
		pretty,
		indentSize,
		textSeparator,
		textStrip,
	).toJs({ dict_converter: Object.fromEntries });

	if (result.ok) {
		setOutput(result.output || "", outputFormat, true);
		setErrors(result.errors || []);
		setStatus(STATUS.READY);
	} else {
		const message =
			result.errors && result.errors.length > 0
				? result.errors.join("\n")
				: "Error";
		setOutput(message, "text", false);
		setErrors(result.errors || []);
		setStatus("Error");
	}

	setEnabled(true);
}

document.getElementById("input").addEventListener("input", scheduleRerender);
document.getElementById("selector").addEventListener("input", scheduleRerender);

for (const el of document.querySelectorAll('input[name="outputFormat"]')) {
	el.addEventListener("change", () => {
		updateVisibleSettings();
		scheduleRerender();
	});
}

for (const el of document.querySelectorAll('input[name="parseMode"]')) {
	el.addEventListener("change", () => {
		scheduleRerender();
	});
}

document.getElementById("safe").addEventListener("change", scheduleRerender);
document.getElementById("cleanup").addEventListener("change", scheduleRerender);
document.getElementById("pretty").addEventListener("change", scheduleRerender);
document
	.getElementById("indentSize")
	.addEventListener("change", scheduleRerender);
document
	.getElementById("textStrip")
	.addEventListener("change", scheduleRerender);
document
	.getElementById("textSeparator")
	.addEventListener("change", scheduleRerender);

initPyodide().catch((e) => {
	setEnabled(true);
	setStatus("Init failed");
	setOutput(`Init failed:\n\n${formatInitError(e)}`, "text", false);
});
