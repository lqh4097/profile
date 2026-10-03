#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '..');
const defaultVaultPath = '/Users/gureiratto/Library/Mobile Documents/iCloud~md~obsidian/Documents/笔记';
const vaultPath = process.env.OBSIDIAN_VAULT_PATH || defaultVaultPath;
const targetNotesDir = path.join(repoRoot, 'content', 'notes');
const targetManifestPath = path.join(repoRoot, 'content', 'notes-manifest.json');

console.log('🔄 正在同步 Obsidian 笔记库...');
console.log(`📂 源目录: ${vaultPath}`);
console.log(`🎯 目标目录: ${targetNotesDir}`);

if (!fs.existsSync(vaultPath)) {
  console.warn(`⚠️ 无法找到 Obsidian 目录: ${vaultPath}`);
  if (fs.existsSync(targetNotesDir)) {
    console.log('ℹ️ 使用仓库中已有的 content/notes 重新构建索引清单...');
    buildManifestFromDir(targetNotesDir);
    process.exit(0);
  } else {
    console.error('❌ 未找到源笔记文件，无法继续。');
    process.exit(1);
  }
}

// 递归扫描目录
function scanDir(dir, base = '') {
  let list = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const fullPath = path.join(dir, entry.name);
    const relPath = path.join(base, entry.name);
    if (entry.isDirectory()) {
      list = list.concat(scanDir(fullPath, relPath));
    } else if (entry.isFile() && entry.name.endsWith('.md')) {
      list.push({ relPath, fullPath });
    }
  }
  return list;
}

// 递归创建目录
function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

// 复制文件
function copyRecursive(srcDir, destDir) {
  ensureDir(destDir);
  const items = scanDir(srcDir);
  for (const item of items) {
    const destFile = path.join(destDir, item.relPath);
    ensureDir(path.dirname(destFile));
    fs.copyFileSync(item.fullPath, destFile);
  }
}

// 提取标题、摘要、标签、字数、标题目录
function parseNote(relPath, content, stat) {
  let text = content;
  let frontmatter = {};

  // 解析 frontmatter (YAML)
  if (text.startsWith('---')) {
    const endIdx = text.indexOf('---', 3);
    if (endIdx !== -1) {
      const yamlStr = text.slice(3, endIdx).trim();
      text = text.slice(endIdx + 3).trim();
      yamlStr.split('\n').forEach(line => {
        const colonIdx = line.indexOf(':');
        if (colonIdx !== -1) {
          const key = line.slice(0, colonIdx).trim();
          const val = line.slice(colonIdx + 1).trim();
          frontmatter[key] = val.replace(/^["']|["']$/g, '');
        }
      });
    }
  }

  // 提取标题：优先取 H1，其次文件名
  let title = frontmatter.title;
  if (!title) {
    const h1Match = text.match(/^#\s+(.+)$/m);
    if (h1Match) {
      title = h1Match[1].trim();
    } else {
      title = path.basename(relPath, '.md');
    }
  }

  // 分类：由上级目录决定
  const dirName = path.dirname(relPath);
  const category = dirName === '.' ? '综合随笔' : dirName.replace(/\\/g, '/');

  // 标签提取
  const tagsSet = new Set();
  if (frontmatter.tags) {
    frontmatter.tags.split(/[\s,]+/).forEach(t => t && tagsSet.add(t.replace(/^#/, '')));
  }
  if (frontmatter.tag) {
    tagsSet.add(String(frontmatter.tag).replace(/^#/, ''));
  }
  // 提取正文里的 #tag
  const inlineTags = text.match(/(?:^|\s)#([\u4e00-\u9fa5a-zA-Z0-9_\-]+)(?=\s|$)/g);
  if (inlineTags) {
    inlineTags.forEach(t => {
      const clean = t.trim().replace(/^#/, '');
      if (clean && !clean.match(/^\d+$/)) tagsSet.add(clean);
    });
  }

  // 统计字数（中文按单字计，英文按词计）
  const cleanForCount = text.replace(/```[\s\S]*?```/g, '').replace(/`.*?`/g, '').replace(/\[.*?\]\(.*?\)/g, '');
  const cjkCount = (cleanForCount.match(/[\u4e00-\u9fa5\u3040-\u30ff]/g) || []).length;
  const enWords = (cleanForCount.match(/[a-zA-Z0-9_\-]+/g) || []).length;
  const wordCount = cjkCount + enWords;
  const readingTime = Math.max(1, Math.ceil(wordCount / 350));

  // 提取纯文本摘要
  const plainText = text
    .replace(/^#+.*$/gm, '')
    .replace(/```[\s\S]*?```/g, '')
    .replace(/`.*?`/g, '')
    .replace(/!\[.*?\]\(.*?\)/g, '')
    .replace(/\[(.*?)\]\(.*?\)/g, '$1')
    .replace(/[>*_\-#~]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  const excerpt = plainText.slice(0, 160) + (plainText.length > 160 ? '...' : '');

  // 提取 headings (H2, H3, H4) 用于目录
  const headings = [];
  const headingRegex = /^(#{2,4})\s+(.+)$/gm;
  let match;
  while ((match = headingRegex.exec(text)) !== null) {
    const level = match[1].length;
    const rawText = match[2].trim().replace(/[*`_]/g, '');
    const id = rawText.toLowerCase().replace(/[^\w\u4e00-\u9fa5\-]+/g, '-');
    headings.push({ level, text: rawText, id });
  }

  // 日期
  let date = frontmatter.date;
  if (!date && stat) {
    date = stat.mtime.toISOString().slice(0, 10);
  } else if (!date) {
    date = new Date().toISOString().slice(0, 10);
  }

  const id = relPath
    .replace(/\.md$/, '')
    .replace(/\//g, '--')
    .replace(/[^\w\u4e00-\u9fa5\-]/g, '');

  return {
    id,
    relPath: relPath.replace(/\\/g, '/'),
    title,
    category,
    tags: Array.from(tagsSet),
    wordCount,
    readingTime,
    excerpt,
    headings,
    date,
    mtime: stat ? stat.mtime.toISOString() : new Date().toISOString()
  };
}

function buildManifestFromDir(notesDir) {
  const items = scanDir(notesDir);
  const notes = [];
  const categoriesMap = {};
  const tagsMap = {};

  for (const item of items) {
    const fullPath = path.join(notesDir, item.relPath);
    const content = fs.readFileSync(fullPath, 'utf8');
    const stat = fs.statSync(fullPath);
    const note = parseNote(item.relPath, content, stat);
    notes.push(note);

    // 统计分类
    categoriesMap[note.category] = (categoriesMap[note.category] || 0) + 1;

    // 统计标签
    note.tags.forEach(t => {
      tagsMap[t] = (tagsMap[t] || 0) + 1;
    });
  }

  // 默认排序：按更新时间或特定顺序
  notes.sort((a, b) => new Date(b.mtime) - new Date(a.mtime));

  const manifest = {
    updatedAt: new Date().toISOString(),
    totalNotes: notes.length,
    categories: Object.entries(categoriesMap).map(([name, count]) => ({ name, count })),
    tags: Object.entries(tagsMap)
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count),
    notes
  };

  ensureDir(path.dirname(targetManifestPath));
  fs.writeFileSync(targetManifestPath, JSON.stringify(manifest, null, 2), 'utf8');
  console.log(`✅ 成功索引 ${notes.length} 篇笔记，清单已写入: ${targetManifestPath}`);
  return manifest;
}

// 执行复制
ensureDir(targetNotesDir);
copyRecursive(vaultPath, targetNotesDir);
console.log('✅ 已将 iCloud 中的笔记复制至 content/notes/');

// 生成索引清单
const manifest = buildManifestFromDir(targetNotesDir);

console.log('\n📊 统计汇总:');
console.log(`- 笔记总数: ${manifest.totalNotes}`);
console.log(`- 分类分布:`);
manifest.categories.forEach(c => console.log(`  • ${c.name}: ${c.count} 篇`));
console.log(`- 标签总数: ${manifest.tags.length}`);
