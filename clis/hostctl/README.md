# hostctl

一个精美的 Rust 版本的 hosts 文件管理工具，支持交互式 TUI 界面。

## 功能特性

- 🎨 **精美的 TUI 界面** - 使用 ratatui 构建的现代化交互界面
- 📁 **配置文件管理** - 创建、编辑、删除多个 hosts 配置
- 🔄 **快速切换** - 一键启用/禁用不同的 hosts 配置
- 💾 **自动备份** - 每次切换时自动备份当前配置
- 🛡️ **安全可靠** - 使用 Rust 编写，内存安全

## 安装

### 使用 Cargo 安装（推荐）

```bash
cd clis/hostctl
cargo install --path .
```

这会安装到 `~/.cargo/bin/hostctl`（确保 `~/.cargo/bin` 在你的 PATH 中）。

### 手动构建

```bash
# 构建
cd clis/hostctl
cargo build --release

# 使用构建的二进制文件
./target/release/hostctl

# 或安装到系统路径（需要 sudo）
sudo cp target/release/hostctl /usr/local/bin/

# 或安装到用户目录
cp target/release/hostctl ~/.local/bin/
```

### 从 GitHub Releases 下载（如果已发布）

如果项目在 GitHub 上发布了预编译的二进制文件，可以直接下载：

```bash
# 下载对应平台的二进制文件
# 例如 Linux x86_64
wget https://github.com/your-org/blacksmith/releases/latest/download/hostctl-x86_64-unknown-linux-gnu
chmod +x hostctl-x86_64-unknown-linux-gnu
sudo mv hostctl-x86_64-unknown-linux-gnu /usr/local/bin/hostctl
```

## 使用方法

### 交互式界面（推荐）

```bash
hostctl
# 或
hostctl --interactive
```

在交互式界面中：
- `↑↓`: 选择配置文件
- `n`: 创建新配置
- `d`: 删除配置
- `e` 或 `Enter`: 启用配置
- `q`: 退出

### 命令行模式

```bash
# 列出所有配置
hostctl list

# 创建新配置
hostctl create dev

# 添加 hosts 条目
hostctl add dev 127.0.0.1 localhost.example.com

# 启用配置（需要 sudo）
sudo hostctl enable dev

# 禁用配置（恢复默认）
sudo hostctl disable

# 查看配置内容
hostctl show dev

# 编辑配置（使用 $EDITOR）
hostctl edit dev

# 备份当前 hosts
sudo hostctl backup

# 恢复备份
sudo hostctl restore

# 查看当前状态
hostctl status
```

## 配置文件格式

配置文件保存在 `~/.config/hostctl/profiles/` 目录，格式为 TOML：

```toml
name = "dev"
description = "开发环境配置"

[[entries]]
ip = "127.0.0.1"
hosts = ["localhost", "example.local"]
comment = "本地开发"

[[entries]]
ip = "192.168.1.100"
hosts = ["api.example.com"]
```

## 目录结构

- `~/.config/hostctl/profiles/` - 配置文件目录
- `~/.config/hostctl/backups/` - 备份文件目录

## 注意事项

- 修改 `/etc/hosts` 需要 root 权限，使用 `sudo` 运行相关命令
- 每次切换配置时会自动备份当前 hosts 文件
- 配置文件使用 TOML 格式，易于编辑

## 开发

```bash
# 运行开发版本
cargo run

# 运行测试
cargo test

# 格式化代码
cargo fmt

# 代码检查
cargo clippy
```

## 许可证

MIT
