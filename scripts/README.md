# Userscripts (Greasy Fork)

用户脚本目录，用于托管发布到 [Greasy Fork](https://greasyfork.org) 的油猴脚本。

## 约定

| 项目 | 格式 | 示例 |
|------|------|------|
| 文件 | `scripts/{name}.user.js` | `scripts/my-tool.user.js` |
| Tag | `scripts/{name}/v{version}` | `scripts/my-tool/v1.0.0` |
| Secret | `GREASYFORK_WEBHOOK_{NAME}` | `GREASYFORK_WEBHOOK_MY_TOOL` |

## 发布流程

1. 修改脚本文件 → merge 到 `main`
2. 打 tag：`git tag scripts/my-tool/v1.0.0 && git push origin scripts/my-tool/v1.0.0`
3. CI 自动解析 tag，通过对应 webhook 通知 Greasy Fork 拉取最新脚本

## 新增脚本

1. 在 `scripts/` 下添加 `{name}.user.js`
2. 在 Greasy Fork 配置 Sync URL：
   ```
   https://raw.githubusercontent.com/heliannuuthus/blacksmith/main/scripts/{name}.user.js
   ```
3. 在 GitHub Settings → Secrets → Actions 添加 `GREASYFORK_WEBHOOK_{NAME}`
4. 在 `.github/workflows/greasyfork-sync.yml` 的 `env` 中添加映射：
   ```yaml
   WEBHOOK_{name}: ${{ secrets.GREASYFORK_WEBHOOK_{NAME} }}
   ```
5. 打 tag 发布即可
