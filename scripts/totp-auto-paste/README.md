# TOTP Auto-Paste

聚焦验证码输入框时，自动从剪贴板读取 TOTP 并填入。配合 Bitwarden / Vaultwarden 的「自动复制 TOTP」功能使用。

## 工作流程

1. Bitwarden 自动填充账密 → TOTP 自动复制到剪贴板
2. 页面跳转到验证码输入页（或出现验证码输入框）
3. 点击 / Tab 聚焦到输入框 → 脚本自动读取剪贴板并填入 6 位验证码

## 安装

### Tampermonkey / Violentmonkey

1. 安装 [Tampermonkey](https://www.tampermonkey.net/) 或 [Violentmonkey](https://violentmonkey.github.io/)
2. 点击 [totp-auto-paste.user.js](./totp-auto-paste.user.js) → Raw → 浏览器会弹出安装提示
3. 确认安装

### Greasy Fork

TODO: 发布后补充链接

## 识别规则

脚本通过以下选择器匹配验证码输入框：

| 选择器 | 场景 |
|--------|------|
| `autocomplete="one-time-code"` | 标准 HTML 语义 |
| `name` 包含 `totp / otp / mfa / 2fa / verification / verify` | 常见命名 |
| `placeholder` 包含 `验证码 / verification` | 占位符匹配 |
| `aria-label` 包含 `verification / code` | 无障碍标签 |

## 限制

- 浏览器剪贴板 API 要求用户交互触发（focus 事件满足此条件），纯被动场景下无法读取
- 仅处理 6 位纯数字验证码
- 不会覆盖已填入的相同值

## License

MIT
