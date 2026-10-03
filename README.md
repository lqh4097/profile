# lqh4097 - 个人主页 (Personal Website)

欢迎访问我的个人主页代码仓库！

## 🚀 简介
本项目是基于纯原生 HTML/CSS 编写的极简现代风个人主页，专为 **Cloudflare Pages** 托管部署优化，无需复杂的构建环境，轻量、秒开、自适应深色/浅色主题。

## 🛠️ 本地预览
直接在浏览器中打开 `index.html` 即可预览：
```bash
# macOS
open index.html
```

## 🌐 部署方式 (Cloudflare Pages)
1. 登录 [Cloudflare 控制台](https://dash.cloudflare.com/)。
2. 进入 **Compute (Workers) > Workers & Pages**，点击 **Create**。
3. 选择 **Pages** -> **Connect to Git**，选中此仓库 `profile`。
4. 构建设置：
   - **Framework preset**: `None`
   - **Build command**: *(留空)*
   - **Build output directory**: `.` (根目录)
5. 点击 **Save and Deploy**，即可在几十秒内完成全球 CDN 部署。
