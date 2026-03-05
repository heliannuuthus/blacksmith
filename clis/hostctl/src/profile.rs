use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub name: String,
    pub description: Option<String>,
    pub entries: Vec<HostEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostEntry {
    pub ip: String,
    pub hosts: Vec<String>,
    pub comment: Option<String>,
}

impl Profile {
    pub fn new(name: String) -> Self {
        Self {
            name,
            description: None,
            entries: Vec::new(),
        }
    }

    pub fn load(path: &PathBuf) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("无法读取配置文件: {}", path.display()))?;
        let profile: Profile = toml::from_str(&content)
            .with_context(|| format!("无法解析配置文件 {}: 请检查 TOML 格式是否正确", path.display()))?;
        Ok(profile)
    }

    pub fn save(&self, path: &PathBuf) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .context("无法序列化配置")?;
        fs::write(path, content)
            .with_context(|| format!("无法保存配置文件: {}", path.display()))?;
        Ok(())
    }

    pub fn to_hosts_content(&self) -> String {
        let mut lines = Vec::new();
        
        if let Some(desc) = &self.description {
            lines.push(format!("# {}", desc));
        }
        lines.push(format!("# Profile: {}", self.name));
        lines.push("".to_string());

        for entry in &self.entries {
            if let Some(comment) = &entry.comment {
                lines.push(format!("# {}", comment));
            }
            let hosts_str = entry.hosts.join(" ");
            lines.push(format!("{}\t{}", entry.ip, hosts_str));
        }

        lines.join("\n")
    }

    pub fn add_entry(&mut self, ip: String, hosts: Vec<String>, comment: Option<String>) {
        self.entries.push(HostEntry {
            ip,
            hosts,
            comment,
        });
    }

    pub fn remove_entry(&mut self, target: &str) -> bool {
        let initial_len = self.entries.len();
        self.entries.retain(|entry| {
            entry.ip != target && !entry.hosts.iter().any(|h| h == target)
        });
        self.entries.len() < initial_len
    }
}
