/**
 * Obsidian 数字花园博客核心渲染引擎
 * 支持：Obsidian 文件夹树、Marked Markdown 渲染、Callouts 提示框、Wikilinks 内部双链、
 * 代码高亮与复制、TOC 大纲与滚动监听、实时全文搜索、沉浸阅读模式
 */

(function () {
  'use strict';

  // 全局状态
  const state = {
    manifest: null,
    notes: [],
    currentNote: null,
    currentCategory: 'ALL',
    searchQuery: '',
    sidebarOpen: window.innerWidth > 900,
    outlineOpen: window.innerWidth > 1200,
    focusMode: false,
    headings: [],
    rawMarkdownMap: new Map()
  };

  // DOM 节点引用
  const el = {
    app: document.getElementById('blog-app'),
    sidebar: document.getElementById('blog-sidebar'),
    reader: document.getElementById('blog-reader'),
    outline: document.getElementById('blog-outline'),
    backdrop: document.getElementById('sidebar-backdrop'),
    breadcrumbs: document.getElementById('breadcrumbs'),
    searchInput: document.getElementById('search-input'),
    searchClear: document.getElementById('search-clear'),
    searchResultsPanel: document.getElementById('search-results-panel'),
    searchResultsList: document.getElementById('search-results-list'),
    searchResultsCount: document.getElementById('search-results-count'),
    fileTree: document.getElementById('file-tree'),
    vaultStats: document.getElementById('vault-stats'),
    categoryChips: document.getElementById('category-chips'),
    sheetTitle: document.getElementById('sheet-title'),
    sheetCategory: document.getElementById('sheet-category'),
    sheetTags: document.getElementById('sheet-tags'),
    sheetDate: document.getElementById('sheet-date'),
    sheetWords: document.getElementById('sheet-words'),
    sheetTime: document.getElementById('sheet-time'),
    articleBody: document.getElementById('article-body'),
    tocList: document.getElementById('toc-list'),
    navPrev: document.getElementById('nav-prev'),
    navPrevTitle: document.getElementById('nav-prev-title'),
    navNext: document.getElementById('nav-next'),
    navNextTitle: document.getElementById('nav-next-title'),
    cardPath: document.getElementById('card-path'),
    cardWords: document.getElementById('card-words'),
    cardMtime: document.getElementById('card-mtime'),
    relatedList: document.getElementById('related-list'),
    toast: document.getElementById('toast'),
    btnToggleSidebar: document.getElementById('btn-toggle-sidebar'),
    btnToggleOutline: document.getElementById('btn-toggle-outline'),
    btnFocusMode: document.getElementById('btn-focus-mode'),
    sidebarCloseBtn: document.getElementById('sidebar-close-btn'),
    outlineCloseBtn: document.getElementById('outline-close-btn'),
    mobileTreeBtn: document.getElementById('mobile-tree-btn'),
    mobileOutlineBtn: document.getElementById('mobile-outline-btn'),
    mobileBarTitle: document.getElementById('mobile-bar-title'),
    btnCopyLink: document.getElementById('btn-copy-link'),
    btnCopyMarkdown: document.getElementById('btn-copy-markdown')
  };

  // 安全 HTML 挂载器 (使用 ContextualFragment 规避跨文档异常与 XSS)
  function renderSafeHTML(target, htmlString) {
    const range = document.createRange();
    range.selectNodeContents(target);
    const fragment = range.createContextualFragment(htmlString);
    target.replaceChildren(fragment);
  }

  // 1. 初始化
  async function init() {
    setupEventListeners();
    configureMarked();

    try {
      // 优先从打包脚本读取离线数据（0延迟、免 CORS、即开即用）
      if (window.__OBSIDIAN_NOTES_DATA__ && window.__OBSIDIAN_NOTES_DATA__.manifest) {
        state.manifest = window.__OBSIDIAN_NOTES_DATA__.manifest;
        state.notes = state.manifest.notes || [];
        if (window.__OBSIDIAN_NOTES_DATA__.notesContent) {
          for (const [k, v] of Object.entries(window.__OBSIDIAN_NOTES_DATA__.notesContent)) {
            state.rawMarkdownMap.set(k, v);
          }
        }
      } else {
        const response = await fetch('content/notes-manifest.json', { cache: 'no-cache' });
        if (!response.ok) throw new Error('无法读取 notes-manifest.json');
        state.manifest = await response.json();
        state.notes = state.manifest.notes || [];
      }

      renderVaultStats();
      renderCategoryChips();
      renderFileTree();

      // 路由：根据 URL hash 或 query 加载指定笔记
      routeInitialNote();
    } catch (err) {
      console.error('初始化失败:', err);
      const errBox = document.createElement('div');
      errBox.className = 'sheet-error';
      const h2 = document.createElement('h2');
      h2.textContent = '⚠️ 知识库索引加载失败';
      const p = document.createElement('p');
      p.textContent = '请确认 content/notes-manifest.json 存在或运行 node scripts/sync-obsidian.js 重新构建。';
      errBox.append(h2, p);
      el.articleBody.replaceChildren(errBox);
    }
  }

  // 2. 配置 Marked Markdown 解析器 (适配 Marked v15 与旧版对象/双参数签名)
  function configureMarked() {
    if (typeof marked === 'undefined') return;

    const renderer = new marked.Renderer();

    // 自定义标题渲染，加入自动锚点
    renderer.heading = function (param1, param2) {
      let text = '';
      let level = 1;
      if (typeof param1 === 'object' && param1 !== null) {
        text = String(param1.text || '');
        level = param1.depth || 1;
      } else {
        text = String(param1 || '');
        level = param2 || 1;
      }
      const plainText = text.replace(/<[^>]+>/g, '').trim();
      const slug = plainText.toLowerCase().replace(/[^\w\u4e00-\u9fa5\-]+/g, '-') || `heading-${level}`;
      return `<h${level} id="${slug}" class="article-heading">
        <a href="#${slug}" class="heading-anchor" aria-hidden="true">#</a>${text}
      </h${level}>\n`;
    };

    // 自定义代码块，添加语言标签与复制按钮
    renderer.code = function (param1, param2) {
      let code = '';
      let lang = 'text';
      if (typeof param1 === 'object' && param1 !== null) {
        code = String(param1.text || '');
        lang = param1.lang || 'text';
      } else {
        code = String(param1 || '');
        lang = param2 || 'text';
      }
      const safeLang = (lang || 'text').toLowerCase();
      const encodedCode = encodeURIComponent(code);
      const escapedCode = code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      return `
        <div class="code-block-wrapper">
          <div class="code-header">
            <span class="code-lang">${safeLang}</span>
            <button type="button" class="btn-copy-code" data-code="${encodedCode}">复制代码</button>
          </div>
          <pre class="language-${safeLang}"><code class="language-${safeLang}">${escapedCode}</code></pre>
        </div>
      `;
    };

    marked.setOptions({
      renderer,
      gfm: true,
      breaks: true
    });
  }

  // 3. 针对 Obsidian 特色的 Markdown 预处理与后处理
  function processObsidianMarkdown(markdown) {
    let md = markdown;

    // 移除 YAML Frontmatter
    if (md.startsWith('---')) {
      const endIdx = md.indexOf('---', 3);
      if (endIdx !== -1) {
        md = md.slice(endIdx + 3).trim();
      }
    }

    // 处理 Obsidian 高亮：==高亮文本== -> <mark>高亮文本</mark>
    md = md.replace(/==([^=]+)==/g, '<mark class="obsidian-mark">$1</mark>');

    // 处理 Obsidian 双链：[[Note Title]] 或 [[Note Path|Display Title]]
    md = md.replace(/\[\[(.*?)(?:\|(.*?))?\]\]/g, (_match, noteTarget, displayText) => {
      const text = displayText || noteTarget;
      return `<a href="#note=${encodeURIComponent(noteTarget.trim())}" class="obsidian-wikilink" data-wikilink="${noteTarget.trim()}">🔗 ${text}</a>`;
    });

    return md;
  }

  // 处理渲染完成后的 DOM：Obsidian Callouts
  function postProcessCallouts(container) {
    const blockquotes = container.querySelectorAll('blockquote');
    blockquotes.forEach(bq => {
      const firstP = bq.querySelector('p');
      if (!firstP) return;

      const rawText = firstP.textContent.trim();
      const match = rawText.match(/^\[!([A-Z0-9_-]+)\]([+-]?)(.*)$/i);
      if (match) {
        const type = match[1].toLowerCase();
        const titleText = match[3].trim() || type.toUpperCase();

        const callout = document.createElement('div');
        callout.className = `obsidian-callout callout-${type}`;

        const iconMap = {
          note: '📝',
          info: 'ℹ️',
          tip: '💡',
          warning: '⚠️',
          danger: '🚨',
          caution: '⚠️',
          quote: '💬',
          example: '📋',
          question: '❓',
          todo: '☑️',
          success: '✅',
          check: '✅'
        };

        const icon = iconMap[type] || '📌';

        const titleDiv = document.createElement('div');
        titleDiv.className = 'callout-title';
        const iconSpan = document.createElement('span');
        iconSpan.className = 'callout-icon';
        iconSpan.textContent = icon;
        const textSpan = document.createElement('span');
        textSpan.className = 'callout-title-text';
        textSpan.textContent = titleText;
        titleDiv.append(iconSpan, textSpan);

        const bodyDiv = document.createElement('div');
        bodyDiv.className = 'callout-body';

        // 移除首行中的 callout 标识符文本
        const cleanedText = firstP.textContent.replace(/^\[!([A-Z0-9_-]+)\]([+-]?)(.*)/i, '').trim();
        if (cleanedText) {
          firstP.textContent = cleanedText;
        } else {
          firstP.remove();
        }

        while (bq.firstChild) {
          bodyDiv.appendChild(bq.firstChild);
        }

        callout.append(titleDiv, bodyDiv);
        bq.replaceWith(callout);
      }
    });
  }

  // 4. 统计与筛选药丸渲染
  function renderVaultStats() {
    if (!state.manifest) return;
    el.vaultStats.textContent = `${state.manifest.totalNotes} 篇笔记 · ${state.manifest.categories.length} 个分类`;
  }

  function renderCategoryChips() {
    if (!state.manifest) return;
    const fragment = document.createDocumentFragment();

    state.manifest.categories.forEach(cat => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'chip-filter';
      btn.dataset.category = cat.name;

      const label = document.createTextNode(`${cat.name} `);
      const small = document.createElement('small');
      small.textContent = String(cat.count);
      btn.append(label, small);

      btn.addEventListener('click', () => {
        setCategoryFilter(cat.name);
      });
      fragment.appendChild(btn);
    });

    el.categoryChips.appendChild(fragment);

    // 全部按钮事件
    const allBtn = el.categoryChips.querySelector('[data-category="ALL"]');
    if (allBtn) {
      allBtn.addEventListener('click', () => setCategoryFilter('ALL'));
    }
  }

  function setCategoryFilter(category) {
    state.currentCategory = category;
    el.categoryChips.querySelectorAll('.chip-filter').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.category === category);
    });
    renderFileTree();
  }

  // 5. 渲染 Obsidian 文件夹树
  function renderFileTree() {
    el.fileTree.replaceChildren();
    const filteredNotes = state.notes.filter(n => {
      if (state.currentCategory === 'ALL') return true;
      return n.category === state.currentCategory || n.category.startsWith(state.currentCategory + '/');
    });

    if (filteredNotes.length === 0) {
      const emptyDiv = document.createElement('div');
      emptyDiv.className = 'tree-empty';
      emptyDiv.textContent = '暂无笔记';
      el.fileTree.appendChild(emptyDiv);
      return;
    }

    // 构建嵌套树结构
    const tree = { folders: {}, files: [] };

    filteredNotes.forEach(note => {
      const parts = note.relPath.split('/');
      if (parts.length === 1) {
        tree.files.push(note);
      } else {
        let current = tree;
        for (let i = 0; i < parts.length - 1; i++) {
          const folderName = parts[i];
          if (!current.folders[folderName]) {
            current.folders[folderName] = { folders: {}, files: [] };
          }
          current = current.folders[folderName];
        }
        current.files.push(note);
      }
    });

    const rootEl = document.createElement('ul');
    rootEl.className = 'tree-list';
    buildTreeNodes(tree, rootEl);
    el.fileTree.appendChild(rootEl);
  }

  function buildTreeNodes(node, containerEl) {
    // 先列文件夹
    for (const [folderName, childNode] of Object.entries(node.folders)) {
      const li = document.createElement('li');
      li.className = 'tree-item folder-item open';

      const folderBtn = document.createElement('div');
      folderBtn.className = 'tree-folder-title';

      const arrow = document.createElement('span');
      arrow.className = 'folder-arrow';
      arrow.textContent = '▾';

      const icon = document.createElement('span');
      icon.className = 'folder-icon';
      icon.textContent = '📁';

      const name = document.createElement('span');
      name.className = 'folder-name';
      name.textContent = folderName;

      folderBtn.append(arrow, icon, name);

      folderBtn.addEventListener('click', () => {
        li.classList.toggle('open');
        const isOpen = li.classList.contains('open');
        arrow.textContent = isOpen ? '▾' : '▸';
        icon.textContent = isOpen ? '📂' : '📁';
      });

      li.appendChild(folderBtn);

      const subList = document.createElement('ul');
      subList.className = 'tree-sublist';
      buildTreeNodes(childNode, subList);
      li.appendChild(subList);

      containerEl.appendChild(li);
    }

    // 再列文件
    node.files.forEach(note => {
      const li = document.createElement('li');
      li.className = 'tree-item file-item';
      if (state.currentNote && state.currentNote.id === note.id) {
        li.classList.add('active');
      }

      const fileBtn = document.createElement('a');
      fileBtn.className = 'tree-file-link';
      fileBtn.href = `#note=${encodeURIComponent(note.relPath)}`;

      const fileIcon = document.createElement('span');
      fileIcon.className = 'file-icon';
      fileIcon.textContent = '📄';

      const fileTitle = document.createElement('span');
      fileTitle.className = 'file-title';
      fileTitle.textContent = note.title;
      fileTitle.title = note.title;

      fileBtn.append(fileIcon, fileTitle);

      fileBtn.addEventListener('click', (e) => {
        e.preventDefault();
        loadNote(note);
        if (window.innerWidth <= 900) {
          toggleSidebar(false);
        }
      });

      li.appendChild(fileBtn);
      containerEl.appendChild(li);
    });
  }

  // 6. 加载与渲染单篇笔记
  async function loadNote(note) {
    if (!note) return;
    state.currentNote = note;

    // 更新地址栏
    window.location.hash = `note=${encodeURIComponent(note.relPath)}`;

    // 更新树选中高亮
    el.fileTree.querySelectorAll('.tree-item.file-item').forEach(item => {
      const link = item.querySelector('.tree-file-link');
      const isMatch = link && link.getAttribute('href') === `#note=${encodeURIComponent(note.relPath)}`;
      item.classList.toggle('active', isMatch);
    });

    // 更新面包屑
    updateBreadcrumbs(note);

    // 更新头部元信息
    el.sheetTitle.textContent = note.title;
    el.sheetCategory.textContent = note.category;
    el.sheetDate.textContent = note.date || '2026';
    el.sheetWords.textContent = `${Number(note.wordCount || 0).toLocaleString()} 字`;
    el.sheetTime.textContent = `${note.readingTime || 1} 分钟阅读`;

    // 移动端操作栏标题
    if (el.mobileBarTitle) el.mobileBarTitle.textContent = note.title;

    // 标签
    el.sheetTags.replaceChildren();
    if (note.tags && note.tags.length) {
      note.tags.forEach(tag => {
        const span = document.createElement('span');
        span.className = 'tag-pill';
        span.textContent = `#${tag}`;
        el.sheetTags.appendChild(span);
      });
    }

    // 右侧大纲卡片详情
    el.cardPath.textContent = `content/notes/${note.relPath}`;
    el.cardWords.textContent = `${note.wordCount} 字`;
    el.cardMtime.textContent = note.mtime ? note.mtime.slice(0, 10) : note.date;

    // 同分类推荐笔记
    renderRelatedNotes(note);

    // 上一篇 / 下一篇
    renderPrevNext(note);

    // 骨架占位
    const placeholder = document.createElement('div');
    placeholder.className = 'sheet-placeholder';
    const spinner = document.createElement('div');
    spinner.className = 'spinner';
    const p = document.createElement('p');
    p.textContent = `正在读取《${note.title}》...`;
    placeholder.append(spinner, p);
    el.articleBody.replaceChildren(placeholder);

    // 拉取 Markdown 原文
    try {
      let rawMarkdown = state.rawMarkdownMap.get(note.relPath);
      if (!rawMarkdown && window.__OBSIDIAN_NOTES_DATA__ && window.__OBSIDIAN_NOTES_DATA__.notesContent) {
        rawMarkdown = window.__OBSIDIAN_NOTES_DATA__.notesContent[note.relPath];
        if (rawMarkdown) state.rawMarkdownMap.set(note.relPath, rawMarkdown);
      }

      if (!rawMarkdown) {
        const response = await fetch(`content/notes/${encodeURI(note.relPath)}`);
        if (!response.ok) throw new Error('无法拉取笔记原文');
        rawMarkdown = await response.text();
        state.rawMarkdownMap.set(note.relPath, rawMarkdown);
      }

      // 解析渲染
      const processed = processObsidianMarkdown(rawMarkdown);
      const html = marked.parse(processed);
      renderSafeHTML(el.articleBody, html);

      // 后处理表格自适应滑动包裹
      el.articleBody.querySelectorAll('table').forEach(table => {
        if (!table.parentElement.classList.contains('table-wrap')) {
          const wrap = document.createElement('div');
          wrap.className = 'table-wrap';
          table.replaceWith(wrap);
          wrap.append(table);
        }
      });

      // 后处理 Callouts
      postProcessCallouts(el.articleBody);

      // 代码高亮
      if (typeof Prism !== 'undefined') {
        Prism.highlightAllUnder(el.articleBody);
      }

      // 提取正文标题生成大纲目录
      buildTOCFromDOM();

      // 滚动到顶部
      el.reader.scrollTo({ top: 0, behavior: 'instant' });

      // 处理 Wikilink 点击
      el.articleBody.querySelectorAll('.obsidian-wikilink').forEach(link => {
        link.addEventListener('click', (e) => {
          e.preventDefault();
          const target = link.dataset.wikilink;
          resolveWikilink(target);
        });
      });

    } catch (err) {
      console.error('加载文章内容出错:', err);
      const errBox = document.createElement('div');
      errBox.className = 'sheet-error';
      const h3 = document.createElement('h3');
      h3.textContent = '无法载入文章正文';
      const errMsg = document.createElement('p');
      errMsg.textContent = `请确认该文件在 content/notes/${note.relPath} 中存在。`;
      errBox.append(h3, errMsg);
      el.articleBody.replaceChildren(errBox);
    }
  }

  // 7. 解析 Wikilink 智能跳转
  function resolveWikilink(target) {
    if (!target) return;
    const cleanTarget = target.replace(/\.md$/, '').trim();

    // 寻找匹配笔记
    const matched = state.notes.find(n => {
      const base = n.title.replace(/\.md$/, '');
      const slug = n.relPath.replace(/\.md$/, '');
      return base === cleanTarget || slug === cleanTarget || n.relPath.endsWith(cleanTarget + '.md');
    });

    if (matched) {
      loadNote(matched);
    } else {
      showToast(`未找到链接对应的笔记: ${cleanTarget}`);
    }
  }

  // 8. 构建右侧大纲 (TOC) 并绑定滚动监听
  function buildTOCFromDOM() {
    el.tocList.replaceChildren();
    const headings = el.articleBody.querySelectorAll('h1, h2, h3, h4');
    state.headings = Array.from(headings);

    if (state.headings.length === 0) {
      const emptyLi = document.createElement('li');
      emptyLi.className = 'toc-empty';
      emptyLi.textContent = '本文无小节标题';
      el.tocList.appendChild(emptyLi);
      return;
    }

    const fragment = document.createDocumentFragment();

    state.headings.forEach((heading, idx) => {
      // 忽略第一个主标题
      if (heading.tagName === 'H1' && idx === 0) return;

      const li = document.createElement('li');
      li.className = `toc-item toc-level-${heading.tagName.toLowerCase()}`;

      const a = document.createElement('a');
      a.className = 'toc-link';
      a.href = `#${heading.id}`;
      a.textContent = heading.textContent.replace(/^#\s*/, '').trim();

      a.addEventListener('click', (e) => {
        e.preventDefault();
        heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
        history.replaceState(null, '', `#${heading.id}`);
        highlightActiveTOC(heading.id);
        if (window.innerWidth <= 1200) {
          toggleOutline(false);
        }
      });

      li.appendChild(a);
      fragment.appendChild(li);
    });

    el.tocList.appendChild(fragment);
    setupScrollSpy();
  }

  // 滚动监听高亮大纲
  let isScrolling = false;
  function setupScrollSpy() {
    el.reader.removeEventListener('scroll', onReaderScroll);
    el.reader.addEventListener('scroll', onReaderScroll, { passive: true });
  }

  function onReaderScroll() {
    if (isScrolling || !state.headings.length) return;
    const readerRect = el.reader.getBoundingClientRect();

    let activeId = null;
    for (let i = 0; i < state.headings.length; i++) {
      const h = state.headings[i];
      const rect = h.getBoundingClientRect();
      if (rect.top - readerRect.top <= 120) {
        activeId = h.id;
      } else {
        break;
      }
    }

    if (activeId) {
      highlightActiveTOC(activeId);
    }
  }

  function highlightActiveTOC(id) {
    el.tocList.querySelectorAll('.toc-link').forEach(link => {
      const isMatch = link.getAttribute('href') === `#${id}`;
      link.classList.toggle('active', isMatch);
    });
  }

  // 9. 面包屑与上一篇/下一篇
  function updateBreadcrumbs(note) {
    el.breadcrumbs.replaceChildren();

    const homeA = document.createElement('a');
    homeA.href = 'index.html';
    homeA.className = 'crumb-link';
    homeA.textContent = '主页';

    const sep1 = document.createElement('span');
    sep1.className = 'crumb-sep';
    sep1.textContent = '/';

    const blogA = document.createElement('a');
    blogA.href = 'blog.html';
    blogA.className = 'crumb-link';
    blogA.textContent = '知识库';

    const sep2 = document.createElement('span');
    sep2.className = 'crumb-sep';
    sep2.textContent = '/';

    const catSpan = document.createElement('span');
    catSpan.className = 'crumb-cat';
    catSpan.textContent = note.category;

    const sep3 = document.createElement('span');
    sep3.className = 'crumb-sep';
    sep3.textContent = '/';

    const curSpan = document.createElement('span');
    curSpan.className = 'crumb-current';
    curSpan.textContent = note.title;

    el.breadcrumbs.append(homeA, sep1, blogA, sep2, catSpan, sep3, curSpan);
  }

  function renderPrevNext(note) {
    const idx = state.notes.findIndex(n => n.id === note.id);
    const prevNote = idx > 0 ? state.notes[idx - 1] : null;
    const nextNote = idx < state.notes.length - 1 ? state.notes[idx + 1] : null;

    if (prevNote) {
      el.navPrev.hidden = false;
      el.navPrevTitle.textContent = prevNote.title;
      el.navPrev.onclick = (e) => {
        e.preventDefault();
        loadNote(prevNote);
      };
    } else {
      el.navPrev.hidden = true;
    }

    if (nextNote) {
      el.navNext.hidden = false;
      el.navNextTitle.textContent = nextNote.title;
      el.navNext.onclick = (e) => {
        e.preventDefault();
        loadNote(nextNote);
      };
    } else {
      el.navNext.hidden = true;
    }
  }

  function renderRelatedNotes(note) {
    el.relatedList.replaceChildren();
    const sameCat = state.notes
      .filter(n => n.category === note.category && n.id !== note.id)
      .slice(0, 5);

    if (sameCat.length === 0) {
      const emptyLi = document.createElement('li');
      emptyLi.className = 'related-empty';
      emptyLi.textContent = '暂无同分类笔记';
      el.relatedList.appendChild(emptyLi);
      return;
    }

    const fragment = document.createDocumentFragment();
    sameCat.forEach(n => {
      const li = document.createElement('li');
      li.className = 'related-item';
      const a = document.createElement('a');
      a.href = `#note=${encodeURIComponent(n.relPath)}`;
      a.textContent = n.title;
      a.addEventListener('click', (e) => {
        e.preventDefault();
        loadNote(n);
      });
      li.appendChild(a);
      fragment.appendChild(li);
    });
    el.relatedList.appendChild(fragment);
  }

  // 10. 实时全文搜索功能
  function handleSearch(query) {
    const q = query.trim().toLowerCase();
    state.searchQuery = q;

    if (!q) {
      el.searchResultsPanel.hidden = true;
      el.searchClear.hidden = true;
      return;
    }

    el.searchClear.hidden = false;
    el.searchResultsPanel.hidden = false;

    const matches = state.notes.filter(n => {
      const matchTitle = n.title.toLowerCase().includes(q);
      const matchPath = n.relPath.toLowerCase().includes(q);
      const matchExcerpt = (n.excerpt || '').toLowerCase().includes(q);
      const matchTags = (n.tags || []).some(t => t.toLowerCase().includes(q));
      return matchTitle || matchPath || matchExcerpt || matchTags;
    });

    el.searchResultsCount.textContent = `找到 ${matches.length} 篇相关笔记`;
    el.searchResultsList.replaceChildren();

    if (matches.length === 0) {
      const emptyDiv = document.createElement('div');
      emptyDiv.className = 'search-empty';
      emptyDiv.textContent = '没有匹配到相关笔记';
      el.searchResultsList.appendChild(emptyDiv);
      return;
    }

    const fragment = document.createDocumentFragment();
    matches.forEach(n => {
      const item = document.createElement('div');
      item.className = 'search-item';

      const catDiv = document.createElement('div');
      catDiv.className = 'search-item-cat';
      catDiv.textContent = n.category;

      const titleH4 = document.createElement('h4');
      titleH4.className = 'search-item-title';
      titleH4.textContent = n.title;

      const excerptP = document.createElement('p');
      excerptP.className = 'search-item-excerpt';
      excerptP.textContent = n.excerpt || '';

      item.append(catDiv, titleH4, excerptP);

      item.addEventListener('click', () => {
        loadNote(n);
        el.searchResultsPanel.hidden = true;
        el.searchInput.value = '';
        el.searchClear.hidden = true;
        if (window.innerWidth <= 900) toggleSidebar(false);
      });
      fragment.appendChild(item);
    });

    el.searchResultsList.appendChild(fragment);
  }

  // 11. 视图切换（侧栏 / 大纲 / 沉浸模式）
  function toggleSidebar(forceState) {
    state.sidebarOpen = typeof forceState === 'boolean' ? forceState : !state.sidebarOpen;
    el.sidebar.classList.toggle('collapsed', !state.sidebarOpen);
    if (el.backdrop) {
      el.backdrop.classList.toggle('active', state.sidebarOpen && window.innerWidth <= 900);
    }
  }

  function toggleOutline(forceState) {
    state.outlineOpen = typeof forceState === 'boolean' ? forceState : !state.outlineOpen;
    el.outline.classList.toggle('collapsed', !state.outlineOpen);
  }

  function toggleFocusMode() {
    state.focusMode = !state.focusMode;
    el.app.classList.toggle('focus-mode', state.focusMode);
    if (state.focusMode) {
      toggleSidebar(false);
      toggleOutline(false);
      showToast('已进入沉浸阅读模式 (按 Alt+F 退出)');
    } else {
      toggleSidebar(window.innerWidth > 900);
      toggleOutline(window.innerWidth > 1200);
    }
  }

  // 12. 路由处理
  function routeInitialNote() {
    const hash = window.location.hash.slice(1);
    const searchParams = new URLSearchParams(window.location.search);
    const noteParam = searchParams.get('note') || searchParams.get('p');

    let targetRelPath = null;

    if (hash.startsWith('note=')) {
      targetRelPath = decodeURIComponent(hash.slice(5));
    } else if (noteParam) {
      targetRelPath = decodeURIComponent(noteParam);
    }

    if (targetRelPath) {
      const found = state.notes.find(n => n.relPath === targetRelPath || n.relPath.replace(/\.md$/, '') === targetRelPath);
      if (found) {
        loadNote(found);
        return;
      }
    }

    // 默认加载：优先加载个人简介或第一篇
    const defaultNote = state.notes.find(n => n.title.includes('个人简介') || n.relPath.includes('个人简介')) || state.notes[0];
    if (defaultNote) {
      loadNote(defaultNote);
    }
  }

  // 13. 事件监听绑定
  function setupEventListeners() {
    // 侧栏开关
    el.btnToggleSidebar.addEventListener('click', () => toggleSidebar());
    el.sidebarCloseBtn.addEventListener('click', () => toggleSidebar(false));
    if (el.mobileTreeBtn) el.mobileTreeBtn.addEventListener('click', () => toggleSidebar(true));

    // 大纲开关
    el.btnToggleOutline.addEventListener('click', () => toggleOutline());
    el.outlineCloseBtn.addEventListener('click', () => toggleOutline(false));
    if (el.mobileOutlineBtn) el.mobileOutlineBtn.addEventListener('click', () => toggleOutline(true));

    // 移动端背景遮罩
    if (el.backdrop) {
      el.backdrop.addEventListener('click', () => {
        toggleSidebar(false);
        toggleOutline(false);
      });
    }

    // 沉浸全屏阅读模式
    el.btnFocusMode.addEventListener('click', toggleFocusMode);

    // 搜索输入
    el.searchInput.addEventListener('input', (e) => handleSearch(e.target.value));
    el.searchClear.addEventListener('click', () => {
      el.searchInput.value = '';
      handleSearch('');
      el.searchInput.focus();
    });

    // 复制链接
    el.btnCopyLink.addEventListener('click', () => {
      navigator.clipboard.writeText(window.location.href).then(() => {
        showToast('文章链接已复制到剪贴板！');
      });
    });

    // 复制 Markdown 原文
    el.btnCopyMarkdown.addEventListener('click', () => {
      if (state.currentNote) {
        const raw = state.rawMarkdownMap.get(state.currentNote.relPath) || '';
        navigator.clipboard.writeText(raw).then(() => {
          showToast('Markdown 原文已复制！');
        });
      }
    });

    // 代码块复制代理
    document.addEventListener('click', (e) => {
      const btn = e.target.closest('.btn-copy-code');
      if (btn) {
        const code = decodeURIComponent(btn.dataset.code || '');
        navigator.clipboard.writeText(code).then(() => {
          const original = btn.textContent;
          btn.textContent = '已复制 ✓';
          setTimeout(() => { btn.textContent = original; }, 2000);
        });
      }
    });

    // 全局快捷键
    window.addEventListener('keydown', (e) => {
      // ⌘K 或 Ctrl+K 打开搜索
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        el.searchInput.focus();
        el.searchInput.select();
      }
      // Alt+B 切换目录
      if (e.altKey && e.key.toLowerCase() === 'b') {
        e.preventDefault();
        toggleSidebar();
      }
      // Alt+O 切换大纲
      if (e.altKey && e.key.toLowerCase() === 'o') {
        e.preventDefault();
        toggleOutline();
      }
      // Alt+F 切换沉浸模式
      if (e.altKey && e.key.toLowerCase() === 'f') {
        e.preventDefault();
        toggleFocusMode();
      }
      // Esc 关闭搜索面板
      if (e.key === 'Escape') {
        if (!el.searchResultsPanel.hidden) {
          el.searchResultsPanel.hidden = true;
        }
      }
    });

    // 监听 Hash 变化
    window.addEventListener('hashchange', () => {
      const hash = window.location.hash.slice(1);
      if (hash.startsWith('note=')) {
        const path = decodeURIComponent(hash.slice(5));
        if (state.currentNote && state.currentNote.relPath === path) return;
        const found = state.notes.find(n => n.relPath === path);
        if (found) loadNote(found);
      }
    });
  }

  // 14. 辅助函数
  function showToast(message) {
    if (!el.toast) return;
    el.toast.textContent = message;
    el.toast.hidden = false;
    el.toast.classList.add('visible');
    clearTimeout(el.toast._timer);
    el.toast._timer = setTimeout(() => {
      el.toast.classList.remove('visible');
      setTimeout(() => { el.toast.hidden = true; }, 300);
    }, 2500);
  }

  // 启动应用
  document.addEventListener('DOMContentLoaded', init);
})();
