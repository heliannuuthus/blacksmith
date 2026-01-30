//! Configuration management commands.

use std::process::Command;

use anyhow::{Context, Result};
use clap::Subcommand;

use crate::config::ConfigManager;

/// Config subcommands.
#[derive(Subcommand)]
pub enum Commands {
    /// Show configuration
    Show,
    /// Edit configuration file
    Edit {
        /// Editor to use (defaults to $EDITOR or 'vi')
        #[arg(long)]
        editor: Option<String>,
    },
    /// Show configuration file path
    Path,
}

/// Execute config command.
///
/// # Errors
/// Returns error if command execution fails.
pub fn execute(manager: &ConfigManager, cmd: Commands) -> Result<()> {
    match cmd {
        Commands::Show => {
            let config = manager.load()?;

            if let Some(ref current) = config.current {
                println!("Current provider: {current}\n");
            } else {
                println!("No provider selected\n");
            }

            if config.providers.is_empty() {
                println!("No providers configured");
            } else {
                println!("Providers:");
                for (name, provider) in &config.providers {
                    let marker = if config.current.as_deref() == Some(name) {
                        " *"
                    } else {
                        ""
                    };
                    println!(
                        "  {}{} [{}] {}",
                        name,
                        marker,
                        provider.provider_type,
                        provider.region.as_deref().unwrap_or("-")
                    );
                }
            }

            println!("\nConfiguration file: {}", manager.path().display());
        }
        Commands::Edit { editor } => {
            let editor = editor
                .or_else(|| std::env::var("EDITOR").ok())
                .unwrap_or_else(|| "vi".to_owned());

            let config_path = manager.path();

            // Ensure config file exists
            if !config_path.exists() {
                manager.save(&crate::config::Config::default())?;
            }

            let status = Command::new(&editor)
                .arg(config_path)
                .status()
                .with_context(|| format!("failed to execute editor: {editor}"))?;

            if !status.success() {
                anyhow::bail!("editor exited with non-zero status");
            }

            // Validate config after editing
            manager
                .load()
                .context("configuration file is invalid after editing")?;

            println!("Configuration updated successfully");
        }
        Commands::Path => {
            println!("{}", manager.path().display());
        }
    }
    Ok(())
}
