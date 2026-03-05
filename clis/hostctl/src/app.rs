use crate::cli::{Cli, Commands};
use crate::config::Config;
use crate::hosts::HostsFile;
use crate::profile::Profile;
use crate::tui::Tui;
use anyhow::{Context, Result};
use colored::*;
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

pub struct App {
    config: Config,
}

impl App {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    pub async fn run(&self, cli: Cli) -> Result<()> {
        // 确保目录存在
        self.config.ensure_dirs()?;

        // 如果没有命令或指定了 interactive，启动 TUI
        if cli.command.is_none() || cli.interactive {
            return Tui::new(self.config.clone()).run().await;
        }

        // 执行 CLI 命令
        match cli.command.unwrap() {
            Commands::List => self.list_profiles(),
            Commands::Create { name } => self.create_profile(name),
            Commands::Delete { name } => self.delete_profile(name),
            Commands::Enable { name } => self.enable_profile(name).await,
            Commands::Disable => self.disable_profile().await,
            Commands::Status => self.show_status(),
            Commands::Edit { name } => self.edit_profile(name),
            Commands::Backup => self.backup_hosts(),
            Commands::Restore => self.restore_hosts().await,
            Commands::Show { name } => self.show_profile(name),
            Commands::Add {
                profile,
                ip,
                hosts,
            } => self.add_entry(profile, ip, hosts),
            Commands::Remove { profile, target } => self.remove_entry(profile, target),
        }
    }

    fn list_profiles(&self) -> Result<()> {
        let profiles = self.get_profiles()?;
        if profiles.is_empty() {
            println!("{}", "没有找到配置文件".yellow());
            return Ok(());
        }

        println!("{}", "配置文件列表:".green().bold());
        for profile in profiles {
            println!("  • {}", profile.green());
        }
        Ok(())
    }

    fn create_profile(&self, name: String) -> Result<()> {
        let profile_path = self.profile_path(&name);
        if profile_path.exists() {
            anyhow::bail!("配置文件已存在: {}", name);
        }

        let profile = Profile::new(name.clone());
        profile.save(&profile_path)?;
        println!("{} {}", "✓".green(), format!("配置文件已创建: {}", name).green());
        Ok(())
    }

    fn delete_profile(&self, name: String) -> Result<()> {
        let profile_path = self.profile_path(&name);
        if !profile_path.exists() {
            anyhow::bail!("配置文件不存在: {}", name);
        }

        fs::remove_file(&profile_path)?;
        println!("{} {}", "✓".green(), format!("配置文件已删除: {}", name).green());
        Ok(())
    }

    async fn enable_profile(&self, name: String) -> Result<()> {
        let profile_path = self.profile_path(&name);
        if !profile_path.exists() {
            anyhow::bail!("配置文件不存在: {}", name);
        }

        let profile = Profile::load(&profile_path)?;
        let hosts_file = HostsFile::new(self.config.hosts_file.clone());

        // 备份当前 hosts
        let backup_name = format!(
            "backup_{}.txt",
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
        let backup_path = self.config.backup_dir.join(&backup_name);
        hosts_file.backup(&backup_path)?;

        // 写入新配置
        let content = profile.to_hosts_content();
        hosts_file.write(&content).await?;

        println!("{} {}", "✓".green(), format!("已启用配置: {}", name).green());
        println!("{} {}", "✓".green(), format!("备份已保存: {}", backup_name).bright_black());
        Ok(())
    }

    async fn disable_profile(&self) -> Result<()> {
        let hosts_file = HostsFile::new(self.config.hosts_file.clone());
        
        // 备份当前 hosts
        let backup_name = format!(
            "backup_{}.txt",
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
        let backup_path = self.config.backup_dir.join(&backup_name);
        hosts_file.backup(&backup_path)?;

        // 恢复默认 hosts（只保留基本内容）
        let default_content = "# /etc/hosts\n127.0.0.1\tlocalhost\n::1\t\tlocalhost\n";
        hosts_file.write(default_content).await?;

        println!("{} {}", "✓".green(), "已禁用配置".green());
        println!("{} {}", "✓".green(), format!("备份已保存: {}", backup_name).bright_black());
        Ok(())
    }

    fn show_status(&self) -> Result<()> {
        let hosts_file = HostsFile::new(self.config.hosts_file.clone());
        let content = hosts_file.read()?;
        
        // 尝试从内容中提取当前配置名称
        if content.contains("# Profile:") {
            for line in content.lines() {
                if line.starts_with("# Profile:") {
                    let profile_name = line.replace("# Profile:", "").trim().to_string();
                    println!("{} {}", "当前配置:".green().bold(), profile_name.green());
                    return Ok(());
                }
            }
        }

        println!("{}", "当前使用默认配置".yellow());
        Ok(())
    }

    fn edit_profile(&self, name: String) -> Result<()> {
        let profile_path = self.profile_path(&name);
        if !profile_path.exists() {
            anyhow::bail!("配置文件不存在: {}", name);
        }

        // 使用系统编辑器
        let editor = std::env::var("EDITOR").unwrap_or_else(|_| "nano".to_string());
        std::process::Command::new(editor)
            .arg(&profile_path)
            .status()
            .context("无法启动编辑器")?;

        println!("{} {}", "✓".green(), format!("配置文件已编辑: {}", name).green());
        Ok(())
    }

    fn backup_hosts(&self) -> Result<()> {
        let hosts_file = HostsFile::new(self.config.hosts_file.clone());
        let backup_name = format!(
            "backup_{}.txt",
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
        let backup_path = self.config.backup_dir.join(&backup_name);
        hosts_file.backup(&backup_path)?;
        println!("{} {}", "✓".green(), format!("备份已保存: {}", backup_name).green());
        Ok(())
    }

    async fn restore_hosts(&self) -> Result<()> {
        let backups = self.get_backups()?;
        if backups.is_empty() {
            anyhow::bail!("没有找到备份文件");
        }

        println!("{}", "可用的备份:".green().bold());
        for (i, backup) in backups.iter().enumerate() {
            println!("  {}: {}", i + 1, backup);
        }

        // 这里应该让用户选择，简化处理，使用最新的
        let latest = backups.last().unwrap();
        let backup_path = self.config.backup_dir.join(latest);
        let hosts_file = HostsFile::new(self.config.hosts_file.clone());
        hosts_file.restore(&backup_path).await?;
        println!("{} {}", "✓".green(), format!("已恢复备份: {}", latest).green());
        Ok(())
    }

    fn show_profile(&self, name: String) -> Result<()> {
        let profile_path = self.profile_path(&name);
        if !profile_path.exists() {
            anyhow::bail!("配置文件不存在: {}", name);
        }

        let profile = Profile::load(&profile_path)?;
        println!("{}", profile.to_hosts_content());
        Ok(())
    }

    fn add_entry(&self, profile: String, ip: String, hosts: Vec<String>) -> Result<()> {
        let profile_path = self.profile_path(&profile);
        if !profile_path.exists() {
            anyhow::bail!("配置文件不存在: {}", profile);
        }

        let mut profile_obj = Profile::load(&profile_path)?;
        profile_obj.add_entry(ip.clone(), hosts.clone(), None);
        profile_obj.save(&profile_path)?;

        println!("{} {}", "✓".green(), format!("已添加条目到 {}: {} -> {:?}", profile, ip, hosts).green());
        Ok(())
    }

    fn remove_entry(&self, profile: String, target: String) -> Result<()> {
        let profile_path = self.profile_path(&profile);
        if !profile_path.exists() {
            anyhow::bail!("配置文件不存在: {}", profile);
        }

        let mut profile_obj = Profile::load(&profile_path)?;
        if profile_obj.remove_entry(&target) {
            profile_obj.save(&profile_path)?;
            println!("{} {}", "✓".green(), format!("已从 {} 移除条目: {}", profile, target).green());
        } else {
            println!("{} {}", "✗".red(), format!("未找到条目: {}", target).red());
        }
        Ok(())
    }

    fn get_profiles(&self) -> Result<Vec<String>> {
        let mut profiles = Vec::new();
        if !self.config.profiles_dir.exists() {
            return Ok(profiles);
        }

        for entry in fs::read_dir(&self.config.profiles_dir)? {
            let entry = entry?;
            if entry.path().is_file() && entry.path().extension().and_then(|s| s.to_str()) == Some("toml") {
                if let Some(name) = entry.path().file_stem().and_then(|s| s.to_str()) {
                    profiles.push(name.to_string());
                }
            }
        }

        profiles.sort();
        Ok(profiles)
    }

    fn get_backups(&self) -> Result<Vec<String>> {
        let mut backups = Vec::new();
        if !self.config.backup_dir.exists() {
            return Ok(backups);
        }

        for entry in fs::read_dir(&self.config.backup_dir)? {
            let entry = entry?;
            if entry.path().is_file() {
                if let Some(name) = entry.path().file_name().and_then(|s| s.to_str()) {
                    backups.push(name.to_string());
                }
            }
        }

        backups.sort();
        Ok(backups)
    }

    fn profile_path(&self, name: &str) -> PathBuf {
        self.config.profiles_dir.join(format!("{}.toml", name))
    }
}
