const list = document.querySelector("[data-collection-list]");
const emptyPanel = document.querySelector("[data-collection-empty]");

if (list && emptyPanel && window.location.protocol !== "file:") {
  const collection = list.dataset.collectionList;

  function safeUrl(value) {
    const raw = String(value ?? "").trim();
    if (!raw) return null;

    if (/^(\/|\.\/|\.\.\/)/.test(raw) && !raw.startsWith("//")) {
      return raw;
    }

    try {
      const url = new URL(raw);
      return url.protocol === "https:" || url.protocol === "http:" ? url.href : null;
    } catch {
      return null;
    }
  }

  function textElement(tag, className, value) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    element.textContent = String(value ?? "");
    return element;
  }

  function makeAnimeCard(item) {
    const card = document.createElement("article");
    card.className = "entry-card";

    const coverUrl = safeUrl(item.cover_url);
    if (coverUrl) {
      const image = document.createElement("img");
      image.className = "anime-cover";
      image.src = coverUrl;
      image.alt = `${item.title || "番剧"} 海报`;
      image.loading = "lazy";
      image.addEventListener("error", () => {
        image.replaceWith(textElement("div", "anime-cover-placeholder", "番"));
      }, { once: true });
      card.append(image);
    } else {
      card.append(textElement("div", "anime-cover-placeholder", "番"));
    }

    const content = document.createElement("div");
    content.className = "entry-content";
    content.append(textElement("h2", "entry-title", item.title || "未命名作品"));

    const meta = [];
    if (item.release_year) meta.push(item.release_year);
    if (item.score !== null && item.score !== undefined && item.score !== "") {
      meta.push(`${item.score} / 10`);
    }
    if (meta.length) content.append(textElement("p", "entry-meta", meta.join(" · ")));
    if (item.review) content.append(textElement("p", "entry-description", item.review));

    card.append(content);
    return card;
  }

  function makeLinkCard(item, kind) {
    const href = safeUrl(item.url);
    const card = document.createElement(href ? "a" : "article");
    card.className = "entry-card resource-entry";

    if (href) {
      card.href = href;
      card.target = "_blank";
      card.rel = "noopener noreferrer";
    }

    const name = item.name || item.title || "未命名条目";
    if (kind === "friends" && safeUrl(item.avatar_url)) {
      const avatar = document.createElement("img");
      avatar.className = "entry-avatar";
      avatar.src = safeUrl(item.avatar_url);
      avatar.alt = `${name} 头像`;
      avatar.loading = "lazy";
      avatar.addEventListener("error", () => {
        avatar.replaceWith(textElement("span", "entry-avatar-placeholder", name.slice(0, 1)));
      }, { once: true });
      card.append(avatar);
    } else {
      card.append(textElement("span", "entry-symbol", kind === "japanese" ? "日" : name.slice(0, 1)));
    }

    const content = document.createElement("div");
    content.className = "entry-content";
    const category = kind === "japanese" ? item.category : "友情链接";
    if (category) content.append(textElement("span", "entry-kind", category));
    content.append(textElement("h2", "entry-title", name));
    if (item.description) content.append(textElement("p", "entry-description", item.description));
    if (href) content.append(textElement("span", "entry-url-hint", "打开链接 ↗"));

    card.append(content);
    return card;
  }

  async function loadCollection() {
    try {
      const response = await fetch(`/api/${collection}`, { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error("Collection request failed");

      const items = await response.json();
      if (!Array.isArray(items) || items.length === 0) return;

      const fragment = document.createDocumentFragment();
      for (const item of items) {
        fragment.append(collection === "anime" ? makeAnimeCard(item) : makeLinkCard(item, collection));
      }
      list.replaceChildren(fragment);
      emptyPanel.hidden = true;
    } catch {
      const heading = emptyPanel.querySelector("h2");
      const description = emptyPanel.querySelector("p");
      if (heading) heading.textContent = "内容暂时无法加载";
      if (description) description.textContent = "请稍后再试。";
    }
  }

  loadCollection();
}
