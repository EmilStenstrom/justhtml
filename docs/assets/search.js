(() => {
	const firstSegment = location.pathname.split("/").filter(Boolean)[0] || "";
	const BASE_PATH =
		firstSegment && !firstSegment.includes(".") ? `/${firstSegment}` : "";
	const SESSION_KEY = "justhtml_docs_search_v1";
	const MAX_RESULTS = 30;

	const rootEl = document.getElementById("jh-search");
	const inputEl = document.getElementById("jh-search-input");
	const statusEl = document.getElementById("jh-search-status");
	const resultsEl = document.getElementById("jh-search-results");

	if (!rootEl || !inputEl || !statusEl || !resultsEl) return;

	const setStatus = (text) => {
		statusEl.textContent = text;
	};

	const escapeHtml = (s) =>
		String(s)
			.replaceAll("&", "&amp;")
			.replaceAll("<", "&lt;")
			.replaceAll(">", "&gt;")
			.replaceAll('"', "&quot;")
			.replaceAll("'", "&#39;");

	const normalize = (s) =>
		String(s)
			.toLowerCase()
			.normalize("NFKD")
			.replace(/[\u0300-\u036f]/g, "")
			.replace(/[^a-z0-9]+/g, " ")
			.trim();

	const tokenize = (q) => {
		const n = normalize(q);
		if (!n) return [];
		return n.split(/\s+/g).filter(Boolean);
	};

	const debounce = (fn, ms) => {
		let t;
		return (...args) => {
			clearTimeout(t);
			t = setTimeout(() => fn(...args), ms);
		};
	};

	const renderResults = (items) => {
		if (!items.length) {
			resultsEl.innerHTML = "";
			return;
		}

		resultsEl.innerHTML = items
			.map((r) => {
				const snippet = r.snippet
					? `<div class="jh-search__snippet">${escapeHtml(r.snippet)}</div>`
					: "";
				return `
          <li class="jh-search__result">
            <a class="jh-search__link" href="${escapeHtml(r.url)}">${escapeHtml(r.title)}</a>
            ${snippet}
          </li>
        `.trim();
			})
			.join("");
	};

	const addStyles = () => {
		if (document.getElementById("jh-search-styles")) return;

		const css = `
      .jh-search { margin-top: 1rem; }
      .jh-search__label { display: block; font-weight: 600; margin-bottom: 0.25rem; }
      .jh-search__input { width: 100%; max-width: 44rem; padding: 0.55rem 0.7rem; border: 1px solid #d0d7de; border-radius: 6px; font-size: 1rem; }
      .jh-search__status { color: #57606a; margin-top: 0.35rem; min-height: 1.25rem; }
      .jh-search__results { list-style: none; padding-left: 0; margin-top: 0.75rem; display: grid; gap: 0.6rem; }
      .jh-search__result { padding: 0.6rem 0.75rem; border: 1px solid #d0d7de; border-radius: 8px; background: #fff; }
      .jh-search__link { font-weight: 600; text-decoration: none; }
      .jh-search__link:hover { text-decoration: underline; }
      .jh-search__snippet { margin-top: 0.25rem; color: #24292f; }
    `.trim();

		const style = document.createElement("style");
		style.id = "jh-search-styles";
		style.textContent = css;
		document.head.appendChild(style);
	};

	const getDocLinksFromIndex = async () => {
		const res = await fetch(`${BASE_PATH}/`, { cache: "no-store" });
		if (!res.ok) throw new Error(`Failed to load docs index: ${res.status}`);

		const html = await res.text();
		const doc = new DOMParser().parseFromString(html, "text/html");
		const links = Array.from(doc.querySelectorAll('a[href$=".html"]'))
			.map((a) => a.getAttribute("href"))
			.filter(Boolean)
			.filter((href) => href.endsWith(".html"))
			.filter((href) => !href.endsWith("/index.html"))
			.filter((href) => !href.endsWith("/search.html"))
			.filter((href) => (BASE_PATH ? href.startsWith(`${BASE_PATH}/`) : true));

		const uniq = Array.from(new Set(links));
		return uniq;
	};

	const extractTitleAndText = (html) => {
		const doc = new DOMParser().parseFromString(html, "text/html");
		const h1 = doc.querySelector("h1");
		const title = (h1 ? h1.textContent : doc.title || "").trim();

		const body = doc.querySelector(".markdown-body") || doc.body;
		const text = body ? body.textContent || "" : "";

		return { title, text };
	};

	const buildIndex = async () => {
		const cached = sessionStorage.getItem(SESSION_KEY);
		if (cached) {
			try {
				const parsed = JSON.parse(cached);
				if (parsed && Array.isArray(parsed.items) && parsed.items.length)
					return parsed.items;
			} catch {
				// ignore
			}
		}

		setStatus("Building search index…");
		const urls = await getDocLinksFromIndex();
		const items = [];

		for (let i = 0; i < urls.length; i++) {
			setStatus(`Indexing ${i + 1}/${urls.length}…`);
			const url = urls[i];
			const res = await fetch(url, { cache: "no-store" });
			if (!res.ok) continue;
			const html = await res.text();
			const { title, text } = extractTitleAndText(html);
			const normalized = normalize(`${title}\n${text}`);

			items.push({
				url,
				title,
				normalized,
				raw: text,
			});
		}

		sessionStorage.setItem(
			SESSION_KEY,
			JSON.stringify({ items, builtAt: Date.now() }),
		);
		return items;
	};

	const scoreMatch = (tokens, normalizedText) => {
		let score = 0;
		for (const t of tokens) {
			const idx = normalizedText.indexOf(t);
			if (idx === -1) return -1;
			score += idx === 0 ? 50 : Math.max(1, 20 - Math.min(20, idx));
			score += Math.min(20, t.length * 2);
		}
		return score;
	};

	const makeSnippet = (rawText, tokens) => {
		const raw = String(rawText || "")
			.replace(/\s+/g, " ")
			.trim();
		if (!raw) return "";

		const lower = raw.toLowerCase();
		let best = -1;
		for (const t of tokens) {
			const i = lower.indexOf(t);
			if (i !== -1 && (best === -1 || i < best)) best = i;
		}
		if (best === -1) return raw.slice(0, 200);

		const start = Math.max(0, best - 80);
		const end = Math.min(raw.length, best + 120);
		const prefix = start > 0 ? "…" : "";
		const suffix = end < raw.length ? "…" : "";
		return `${prefix}${raw.slice(start, end)}${suffix}`;
	};

	const search = (items, q) => {
		const tokens = tokenize(q);
		if (!tokens.length) return [];

		const scored = [];
		for (const it of items) {
			const score = scoreMatch(tokens, it.normalized);
			if (score >= 0) {
				scored.push({
					url: it.url,
					title: it.title,
					score,
					snippet: makeSnippet(it.raw, tokens),
				});
			}
		}

		scored.sort((a, b) => b.score - a.score);
		return scored.slice(0, MAX_RESULTS);
	};

	const init = async () => {
		addStyles();
		rootEl.hidden = false;

		let items;
		try {
			items = await buildIndex();
		} catch (e) {
			setStatus("Failed to build search index.");
			return;
		}

		setStatus(`Ready. Indexed ${items.length} pages.`);

		const run = debounce(() => {
			const q = inputEl.value;
			if (!q.trim()) {
				setStatus(`Ready. Indexed ${items.length} pages.`);
				renderResults([]);
				return;
			}

			const results = search(items, q);
			setStatus(`${results.length} result${results.length === 1 ? "" : "s"}.`);
			renderResults(results);
		}, 50);

		inputEl.addEventListener("input", run);
		inputEl.focus();
	};

	init();
})();
