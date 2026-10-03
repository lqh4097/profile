# lqh4097 - 个人主页 (Personal Website)

欢迎访问我的个人主页代码仓库！

## 🚀 简介
本项目是一个基于 HTML、CSS 和 JavaScript 的个人网站，部署在 **Cloudflare Pages**。番剧、日语资料和友情链接通过 Pages Functions 从 Cloudflare D1 读取。

## 🗃️ D1 内容数据

- Pages Functions 提供只读接口：`/api/anime`、`/api/japanese`、`/api/friends`。
- D1 绑定变量名使用 `DB` 或 `db`。
- `wrangler.jsonc` 保存 D1 数据库名称、ID 和迁移目录；Cloudflare 页面函数仍使用控制台里已经配置的 `DB` 绑定。
- 首次部署时，Pages 会通过 [`migrations/0001_initial_schema.sql`](migrations/0001_initial_schema.sql) 自动创建三张数据表。
- 迁移只在 `main` 分支的生产部署中运行，预览部署不会修改生产数据库。
- 添加或修改条目可以在 D1 SQL Console 操作；提交新记录后，页面会自动读取并显示。
- 如果之前已经手动创建过表，初始迁移使用了 `IF NOT EXISTS`，可以安全地补记迁移状态。

### 自动运行数据库迁移

在 Cloudflare Pages 项目中做一次配置：

1. 在 **Settings → Environment variables** 的生产环境添加 `D1_DATABASE_NAME`，值为 D1 数据库名称。
2. 添加 `CLOUDFLARE_ACCOUNT_ID`，值为 Cloudflare 账户 ID；再添加 `CLOUDFLARE_API_TOKEN`，使用仅授予 **D1 Edit** 权限的 API Token，并将它设为 Secret。不要把 Token 写进仓库或发到聊天里。
3. 在 **Settings → Builds & deployments** 中将 Build command 设为 `bash build.sh`，Build output directory 设为 `.`。生产分支为 `main`。本仓库的 `wrangler.jsonc` 只为 Wrangler 数据库迁移提供配置，不覆盖 Pages 控制台中的构建输出目录或函数绑定。
4. 推送到 `main` 后，Cloudflare 会先应用尚未运行的迁移，再部署网站。之后新增表结构时，只需添加下一个编号的 `.sql` 文件（例如 `migrations/0002_add_tags.sql`）并推送。

普通新增番剧、日语资料或友链内容不需要新建表；当前可在 D1 SQL Console 中向现有表添加记录。网站目前提供只读 API，尚未提供网页管理界面。

数据字段：

- `anime`：`title`、`release_year`、`score`、`review`、`cover_url`、`sort_order`
- `japanese_resources`：`title`、`category`、`description`、`url`、`sort_order`
- `friend_links`：`name`、`url`、`description`、`avatar_url`、`sort_order`

图片和资料文件可以放在静态资源目录中，并将路径填入对应记录。较大的文件可另行接入 R2。

## 🛠️ 本地预览

```bash
# macOS
open index.html
```

直接打开文件可以预览静态页面；D1 数据接口需在 Cloudflare Pages 部署环境中运行。

## 🌐 部署方式 (Cloudflare Pages)
1. 登录 [Cloudflare 控制台](https://dash.cloudflare.com/)。
2. 进入 **Compute (Workers) > Workers & Pages**，点击 **Create**。
3. 选择 **Pages** -> **Connect to Git**，选中此仓库 `profile`。
4. 构建设置：
   - **Framework preset**: `None`
   - **Build command**: `bash build.sh`
   - **Build output directory**: `.` (根目录)
5. 点击 **Save and Deploy**，即可在几十秒内完成全球 CDN 部署。
