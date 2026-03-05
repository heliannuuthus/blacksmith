use anyhow::{Context, Result};
use dirs::home_dir;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub profiles_dir: PathBuf,
    pub backup_dir: PathBuf,
    pub hosts_file: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        let home = home_dir().unwrap_or_else(|| PathBuf::from("~"));
        Self {
            profiles_dir: home.join(".config").join("hostctl").join("profiles"),
            backup_dir: home.join(".config").join("hostctl").join("backups"),
            hosts_file: PathBuf::from("/etc/hosts"),
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let config = Self::default();
        
        // 创建必要的目录
        fs::create_dir_all(&config.profiles_dir)
            .context("无法创建配置目录")?;
        fs::create_dir_all(&config.backup_dir)
            .context("无法创建备份目录")?;
        
        Ok(config)
    }

    pub fn ensure_dirs(&self) -> Result<()> {
        fs::create_dir_all(&self.profiles_dir)
            .context("无法创建配置目录")?;
        fs::create_dir_all(&self.backup_dir)
            .context("无法创建备份目录")?;
        Ok(())
    }
}
