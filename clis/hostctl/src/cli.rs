use clap::{Parser, Subcommand};
use clap_complete::Shell;

#[derive(Parser)]
#[command(name = "hostctl")]
#[command(about = "一个精美的 hosts 文件管理工具", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// 启用交互式 TUI 界面
    #[arg(short, long)]
    pub interactive: bool,
}

#[derive(Subcommand)]
pub enum Commands {
    /// 列出所有配置文件
    List,
    /// 创建新的配置文件
    Create {
        /// 配置文件名
        name: String,
    },
    /// 删除配置文件
    Delete {
        /// 配置文件名
        name: String,
    },
    /// 启用配置文件（应用到 /etc/hosts）
    Enable {
        /// 配置文件名
        name: String,
    },
    /// 禁用当前配置（恢复默认）
    Disable,
    /// 显示当前启用的配置
    Status,
    /// 编辑配置文件
    Edit {
        /// 配置文件名
        name: String,
    },
    /// 备份当前 hosts 文件
    Backup,
    /// 恢复 hosts 文件
    Restore,
    /// 显示配置文件内容
    Show {
        /// 配置文件名
        name: String,
    },
    /// 添加 hosts 条目到配置文件
    Add {
        /// 配置文件名
        profile: String,
        /// IP 地址
        ip: String,
        /// 主机名（多个用空格分隔）
        hosts: Vec<String>,
    },
    /// 从配置文件移除 hosts 条目
    Remove {
        /// 配置文件名
        profile: String,
        /// IP 地址或主机名
        target: String,
    },
    /// 生成 shell 补全脚本
    Completion {
        /// 目标 shell 类型
        shell: Shell,
    },
    /// 检查运行环境和依赖状态
    Doctor,
}
