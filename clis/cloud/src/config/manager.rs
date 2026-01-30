//! Configuration manager for cloud CLI.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::types::Provider;

/// Global configuration structure.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    /// Current active provider name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current: Option<String>,
    /// Configured providers
    #[serde(default)]
    pub providers: HashMap<String, Provider>,
}

/// Configuration manager.
pub struct ConfigManager {
    path: PathBuf,
}

impl ConfigManager {
    /// Create a new configuration manager.
    ///
    /// # Errors
    /// Returns error if home directory cannot be determined or config directory cannot be created.
    pub fn new() -> Result<Self> {
        let config_dir = dirs::home_dir()
            .context("cannot determine home directory")?
            .join(".config")
            .join("blacksmith")
            .join("cloud");

        fs::create_dir_all(&config_dir)
            .with_context(|| format!("cannot create config directory: {}", config_dir.display()))?;

        Ok(Self {
            path: config_dir.join("config.toml"),
        })
    }

    /// Get the configuration file path.
    #[must_use]
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    /// Load configuration from file.
    ///
    /// # Errors
    /// Returns error if file cannot be read or parsed.
    pub fn load(&self) -> Result<Config> {
        if !self.path.exists() {
            return Ok(Config::default());
        }

        let content = fs::read_to_string(&self.path)
            .with_context(|| format!("cannot read config file: {}", self.path.display()))?;

        toml::from_str(&content).context("cannot parse config file")
    }

    /// Save configuration to file.
    ///
    /// # Errors
    /// Returns error if file cannot be written.
    pub fn save(&self, config: &Config) -> Result<()> {
        let content = toml::to_string_pretty(config).context("cannot serialize config")?;

        fs::write(&self.path, content)
            .with_context(|| format!("cannot write config file: {}", self.path.display()))
    }

    /// Get a provider by name.
    ///
    /// # Errors
    /// Returns error if config cannot be loaded.
    pub fn get_provider(&self, name: &str) -> Result<Option<Provider>> {
        let config = self.load()?;
        Ok(config.providers.get(name).cloned())
    }

    /// Add or update a provider.
    ///
    /// # Errors
    /// Returns error if config cannot be loaded or saved.
    pub fn set_provider(&self, name: &str, provider: Provider) -> Result<()> {
        let mut config = self.load()?;
        config.providers.insert(name.to_owned(), provider);
        self.save(&config)
    }

    /// Remove a provider.
    ///
    /// # Errors
    /// Returns error if config cannot be loaded or saved.
    pub fn remove_provider(&self, name: &str) -> Result<bool> {
        let mut config = self.load()?;
        let removed = config.providers.remove(name).is_some();
        
        // Clear current if it was the removed provider
        if config.current.as_deref() == Some(name) {
            config.current = None;
        }
        
        self.save(&config)?;
        Ok(removed)
    }

    /// Rename a provider.
    ///
    /// # Errors
    /// Returns error if config cannot be loaded or saved, or if provider doesn't exist.
    pub fn rename_provider(&self, old_name: &str, new_name: &str) -> Result<()> {
        let mut config = self.load()?;
        
        if !config.providers.contains_key(old_name) {
            anyhow::bail!("provider '{old_name}' not found");
        }
        
        if config.providers.contains_key(new_name) {
            anyhow::bail!("provider '{new_name}' already exists");
        }
        
        if let Some(provider) = config.providers.remove(old_name) {
            config.providers.insert(new_name.to_owned(), provider);
            
            // Update current if it was the renamed provider
            if config.current.as_deref() == Some(old_name) {
                config.current = Some(new_name.to_owned());
            }
        }
        
        self.save(&config)
    }

    /// List all configured providers.
    ///
    /// # Errors
    /// Returns error if config cannot be loaded.
    #[allow(dead_code)] // May be used in future commands
    pub fn list_providers(&self) -> Result<Vec<String>> {
        let config = self.load()?;
        Ok(config.providers.keys().cloned().collect())
    }

    /// Get the current provider name.
    ///
    /// # Errors
    /// Returns error if config cannot be loaded.
    pub fn current(&self) -> Result<Option<String>> {
        let config = self.load()?;
        Ok(config.current)
    }

    /// Get the current provider configuration.
    ///
    /// # Errors
    /// Returns error if config cannot be loaded.
    pub fn current_provider(&self) -> Result<Option<(String, Provider)>> {
        let config = self.load()?;
        
        if let Some(name) = &config.current {
            if let Some(provider) = config.providers.get(name) {
                return Ok(Some((name.clone(), provider.clone())));
            }
        }
        
        Ok(None)
    }

    /// Set the current provider.
    ///
    /// # Errors
    /// Returns error if config cannot be loaded or saved, or if provider doesn't exist.
    pub fn set_current(&self, name: &str) -> Result<()> {
        let mut config = self.load()?;
        
        if !config.providers.contains_key(name) {
            anyhow::bail!("provider '{name}' not found");
        }
        
        config.current = Some(name.to_owned());
        self.save(&config)
    }

    /// Unset the current provider.
    ///
    /// # Errors
    /// Returns error if config cannot be loaded or saved.
    #[allow(dead_code)] // May be used in future commands
    pub fn unset_current(&self) -> Result<()> {
        let mut config = self.load()?;
        config.current = None;
        self.save(&config)
    }

    /// Resolve provider: use specified name or fall back to current.
    ///
    /// # Errors
    /// Returns error if no provider is specified and no current provider is set.
    pub fn resolve_provider(&self, name: Option<&str>) -> Result<(String, Provider)> {
        if let Some(name) = name {
            let provider = self
                .get_provider(name)?
                .with_context(|| format!("provider '{name}' not found"))?;
            return Ok((name.to_owned(), provider));
        }
        
        self.current_provider()?
            .context("no provider specified and no current provider set. Use 'cloud use <provider>' or specify --provider")
    }
}
