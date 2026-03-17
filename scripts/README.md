# Userscripts (Greasy Fork)

用户脚本目录，用于托管发布到 [Greasy Fork](https://greasyfork.org) 的油猴脚本。

## 约定

| 项目 | 格式 | 示例 |
|------|------|------|
| 文件 | `scripts/{name}/{name}.user.js` 或 `scripts/{name}.user.js` | `scripts/totp-auto-paste/totp-auto-paste.user.js` |
| Secret | `GREASYFORK_URL_{NAME}` | `GREASYFORK_URL_TOTP_AUTO_PASTE` |

## 发布流程

1. 修改脚本文件（更新 `@version`）→ merge 到 `main`
2. CI 自动检测变更并通知 Greasy Fork 同步

## 新增脚本

1. 在 `scripts/` 下添加 `{name}/{name}.user.js` 或 `{name}.user.js`
2. 在 Greasy Fork 配置 Sync URL（raw 链接）：
   ```
   https://raw.githubusercontent.com/heliannuuthus/blacksmith/main/scripts/{name}/{name}.user.js
   ```
3. 在 Greasy Fork 脚本管理页配置 webhook，复制 webhook URL
4. 在 GitHub Settings → Secrets → Actions 添加：
   - `GREASYFORK_URL_{NAME}`：webhook URL（必需，连字符转下划线，如 `TOTP_AUTO_PASTE`）
   - `GREASYFORK_SECRET_{NAME}`：webhook secret（可选）
5. 在 `.github/workflows/greasyfork-sync.yml` 的 matrix 中添加一项：
   ```yaml
   - script: {name}
     suffix: {NAME}
   ```
