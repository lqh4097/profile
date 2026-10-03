# Antigravity & Codex 代理配置指南（免全局 TUN 模式）

本指南介绍如何在不同系统和环境下，为 **Antigravity** 和 **Codex** 配置独立的网络代理，从而**避免开启全局 TUN / 增强模式**，节省代理流量并保障连接稳定性。

---

## 一、 Antigravity 代理配置

### 1. macOS 篇：编译独立启动器 App

#### 核心原理
Antigravity 客户端基于 Electron 开发，且在运行时会在后台拉取名为 `language_server` 的后台进程（负责处理模型请求与终端命令）。
通过在启动时注入 `HTTP_PROXY`、`HTTPS_PROXY` 和 `ALL_PROXY` 环境变量，可以让主程序及所有子进程强制走代理，而系统的其他软件依然走直连。

#### 步骤 1：编译制作独立的启动器 App
1. 打开 macOS 的 **终端 (Terminal)**。
2. 运行以下命令，在桌面生成一个名为 `Antigravity-Proxy.app` 的启动器：
   ```bash
   osacompile -o ~/Desktop/Antigravity-Proxy.app -e 'do shell script "export HTTP_PROXY=http://127.0.0.1:7879; export HTTPS_PROXY=http://127.0.0.1:7879; export ALL_PROXY=socks5://127.0.0.1:7879; /Applications/Antigravity.app/Contents/MacOS/Antigravity >/dev/null 2>&1 &"'
   ```
   *(注：请根据你本地 Clash/Verge 的实际端口替换命令中的 `7879`。)*

#### 步骤 2：为启动器换上官方图标
在终端中继续运行以下 Swift 命令，自动提取原版图标并刷新 Finder 缓存：
```bash
# 1. 使用 Swift 调用 Cocoa 接口写入自定义图标
swift -e 'import Cocoa; NSWorkspace.shared.setIcon(NSImage(contentsOfFile: "/Applications/Antigravity.app/Contents/Resources/icon.icns"), forFile: NSHomeDirectory() + "/Desktop/Antigravity-Proxy.app", options: [])'

# 2. 重启 Finder 刷新桌面缓存
killall Finder
```

#### 步骤 3：使用与整理
1. **直接双击运行**：双击桌面上的 `Antigravity-Proxy.app` 即可静默拉起带代理的客户端。
2. **整理归档**：可将其拖入 `应用程序 (Applications)` 文件夹，或拖至屏幕下方的 Dock 栏一键启动。

---

### 2. Windows 篇：透明代理注入工具

#### 核心原理
在 Windows 上通过在 Antigravity 安装目录下部署特定的 `version.dll` 和配置文件 `config.json`。程序启动时会自动加载该 DLL 并 Hook 网络 API，将网络连接强制透明重定向到指定的本地代理端口。

#### 步骤 1：下载工具
前往 GitHub [yuaotian/antigravity-proxy](https://github.com/yuaotian/antigravity-proxy/releases) 下载最新版本的压缩包（如 `win-x64.zip`）。

#### 步骤 2：解压与部署
1. 将下载的压缩包解压，得到 `version.dll` 和 `config.json`。
2. 打开 Antigravity 的安装目录（包含 `Antigravity.exe` 的文件夹）。
   * *默认路径通常为：`C:\Users\<你的用户名>\AppData\Local\Programs\Antigravity`*
3. 将解压出来的两个文件复制粘贴到该目录下。

#### 步骤 3：配置本地代理
用文本编辑器打开安装目录下的 `config.json`，修改 `"proxy"` 键值为你本地代理软件的地址与端口，保存即可。例如：
```json
{
  "proxy": "http://127.0.0.1:7890"
}
```

#### 步骤 4：运行验证
直接双击运行 `Antigravity.exe`，软件及其后台服务将自动通过配置的本地代理联网，且不影响其他软件。

---

## 二、 Codex 代理配置（解决重连问题）

### 核心原理
在进行本地开发或使用 Codex 相关服务（如 CLI 客户端、IDE 插件或后台 agent）时，如果遇到网络连接超时、WebSocket 握手失败或频繁重连的问题，可以通过在项目或服务的根目录下配置 `.env` 文件来强制指定代理，无需开启全局代理。

### 配置步骤

#### 1. 创建或编辑 `.env` 文件
在你的 Codex 项目根目录下，或者在全局配置目录（如用户家目录下的 `~/.codex/` 文件夹内）创建或编辑一个名为 `.env` 的文件。

#### 2. 写入代理配置
在 `.env` 文件中添加以下环境变量，指向你本地代理软件的 HTTP 端口（以 `7890` 端口为例）：

```env
# 开启 HTTP 代理
HTTP_PROXY="http://127.0.0.1:7890"

# 开启 HTTPS 代理
HTTPS_PROXY="http://127.0.0.1:7890"
```

> **注意**：
> - 请根据你的代理软件实际开放的端口修改上面的 `7890`。
> - 在 `.env` 中修改并保存后，通常需要重启 Codex 相关进程、终端会话或 IDE 编辑器以使配置生效。

---

## 三、 常见问题与排查
1. **配置代理后仍然连接失败？**
   - 检查本地代理软件是否开启了“允许局域网连接”（Allow LAN）或本地端口是否有变动。
   - 在终端使用 `curl -I https://www.google.com` 命令配合临时环境变量测试代理网络本身是否通畅。
2. **账号或地区风控？**
   - 代理节点建议选择网络纯净度较高的原生节点，频繁更换代理 IP 或使用垃圾机房节点容易触发账号风控。
