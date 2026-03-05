use anyhow::{Context, Result};
use colored::*;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[derive(Clone)]
pub struct HostsFile {
    path: PathBuf,
}

impl HostsFile {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn read(&self) -> Result<String> {
        fs::read_to_string(&self.path)
            .with_context(|| format!("无法读取 hosts 文件: {}", self.path.display()))
    }

    pub async fn write(&self, content: &str) -> Result<()> {
        // 检查是否有写入权限
        if !self.has_write_permission()? {
            // 尝试使用 polkit 提升权限
            return self.write_with_polkit(content).await;
        }
        
        fs::write(&self.path, content)
            .with_context(|| format!("无法写入 hosts 文件: {}", self.path.display()))?;
        Ok(())
    }

    fn has_write_permission(&self) -> Result<bool> {
        if !self.path.exists() {
            return Ok(false);
        }
        
        // 尝试以只读方式打开文件来测试权限
        // 更简单的方法：直接尝试写入一个临时文件，如果失败则没有权限
        match fs::OpenOptions::new()
            .write(true)
            .open(&self.path)
        {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    async fn write_with_polkit(&self, content: &str) -> Result<()> {
        // 检查是否在交互式终端中
        if !std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            anyhow::bail!(
                "需要权限来写入 {}，但不在交互式终端中。\n请使用: sudo hostctl enable <profile>",
                self.path.display()
            );
        }
        
        println!("{}", "需要权限来修改 /etc/hosts".yellow());
        println!("{}", "正在通过 polkit 请求权限...".yellow());
        
        // 将内容写入临时文件
        let temp_file = std::env::temp_dir().join(format!("hostctl_{}.tmp", std::process::id()));
        fs::write(&temp_file, content)
            .context("无法写入临时文件")?;
        
        // 使用 pkexec 执行写入操作
        // pkexec 会自动通过 polkit 请求权限并显示认证对话框
        let output = Command::new("pkexec")
            .arg("sh")
            .arg("-c")
            .arg(format!("cat {} > {}", temp_file.display(), self.path.display()))
            .output()
            .context("无法启动 pkexec 命令，请确保已安装 polkit-1")?;
        
        // 清理临时文件
        let _ = fs::remove_file(&temp_file);
        
        if !output.status.success() {
            let error_msg = String::from_utf8_lossy(&output.stderr);
            if output.status.code() == Some(126) {
                anyhow::bail!("权限被拒绝：无法修改 /etc/hosts");
            } else if output.status.code() == Some(127) {
                anyhow::bail!("pkexec 未找到，请安装 polkit-1: sudo apt-get install policykit-1");
            } else {
                anyhow::bail!("pkexec 写入失败: {}", error_msg);
            }
        }
        
        Ok(())
    }

    pub fn backup(&self, backup_path: &PathBuf) -> Result<()> {
        let content = self.read()?;
        fs::write(backup_path, content)
            .with_context(|| format!("无法创建备份: {}", backup_path.display()))?;
        Ok(())
    }

    pub async fn restore(&self, backup_path: &PathBuf) -> Result<()> {
        let content = fs::read_to_string(backup_path)
            .with_context(|| format!("无法读取备份文件: {}", backup_path.display()))?;
        self.write(&content).await?;
        Ok(())
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}
