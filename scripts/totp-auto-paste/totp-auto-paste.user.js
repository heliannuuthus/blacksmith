// ==UserScript==
// @name         TOTP Auto-Paste | 验证码自动填入
// @namespace    https://github.com/heliannuuthus/blacksmith
// @version      1.1.0
// @description  Auto-paste TOTP verification code from clipboard when focusing on OTP input fields. Works with Bitwarden / Vaultwarden auto-copy.
// @description:zh-CN  聚焦验证码输入框时自动从剪贴板粘贴 TOTP，配合 Bitwarden / Vaultwarden 的自动复制功能使用
// @author       heliannuuthus
// @license      MIT
// @homepage     https://github.com/heliannuuthus/blacksmith/tree/main/scripts/totp-auto-paste
// @supportURL   https://github.com/heliannuuthus/blacksmith/issues
// @match        *://*/*
// @grant        none
// @run-at       document-idle
// ==/UserScript==

(function () {
  'use strict';

  const SELECTORS = [
    'input[autocomplete="one-time-code"]',
    'input[name*="totp" i]',
    'input[name*="otp" i]',
    'input[name*="mfa" i]',
    'input[name*="2fa" i]',
    'input[name*="verification" i]',
    'input[name*="verify" i]',
    'input[name*="passcode" i]',
    'input[name*="pin_code" i]',
    'input[name*="auth_code" i]',
    'input[name*="security_code" i]',
    'input[name*="two_factor" i]',
    'input[id*="otp" i]',
    'input[id*="totp" i]',
    'input[id*="2fa" i]',
    'input[id*="two-factor" i]',
    'input[id*="mfa" i]',
    'input[placeholder*="验证码"]',
    'input[placeholder*="安全码"]',
    'input[placeholder*="动态口令"]',
    'input[placeholder*="verification" i]',
    'input[placeholder*="one-time" i]',
    'input[placeholder*="6-digit" i]',
    'input[placeholder*="auth code" i]',
    'input[placeholder*="passcode" i]',
    'input[aria-label*="verification" i]',
    'input[aria-label*="code" i]',
    'input[aria-label*="one-time" i]',
    'input[aria-label*="two-factor" i]',
    'input[aria-label*="passcode" i]',
    'input[data-testid*="otp" i]',
    'input[data-testid*="totp" i]',
    'input[data-testid*="2fa" i]',
    'input[data-testid*="two-fa" i]',
    'input[data-testid*="mfa" i]',
    'input[inputmode="numeric"][maxlength="6"]',
  ];

  function findTotpInput() {
    for (const sel of SELECTORS) {
      const el = document.querySelector(sel);
      if (el && el.offsetParent !== null) return el;
    }
    return null;
  }

  async function tryPaste(input) {
    try {
      const text = await navigator.clipboard.readText();
      const code = text.replace(/[^0-9]/g, '').slice(0, 6);
      if (code.length === 6 && input.value !== code) {
        input.value = code;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
      }
    } catch (_) {
      // clipboard permission denied — silent
    }
  }

  function watch(input) {
    if (input.dataset.totpWatched) return;
    input.dataset.totpWatched = 'true';
    input.addEventListener('focus', () => tryPaste(input));
    input.addEventListener('click', () => tryPaste(input));
    if (document.activeElement === input) tryPaste(input);
  }

  const observer = new MutationObserver(() => {
    const input = findTotpInput();
    if (input) watch(input);
  });

  observer.observe(document.body, { childList: true, subtree: true });

  const existing = findTotpInput();
  if (existing) watch(existing);
})();
