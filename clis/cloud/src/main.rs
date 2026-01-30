//! Cloud CLI - Cloud resource management tool.

mod backends;
mod commands;
mod config;
mod resources;
mod services;

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::env::CompleteEnv;

use crate::config::ConfigManager;
use crate::resources::{OutputFormat, ResourceKind};

/// Get available provider names for shell completion.
fn complete_provider_names(current: &std::ffi::OsStr) -> Vec<clap_complete::CompletionCandidate> {
    let current = current.to_string_lossy();
    ConfigManager::new()
        .and_then(|m| m.list_providers())
        .unwrap_or_default()
        .into_iter()
        .filter(|name| name.starts_with(current.as_ref()))
        .map(clap_complete::CompletionCandidate::new)
        .collect()
}

/// Get available resource type names for shell completion.
fn complete_resource_types(current: &std::ffi::OsStr) -> Vec<clap_complete::CompletionCandidate> {
    let current = current.to_string_lossy();
    ResourceKind::all_names()
        .into_iter()
        .filter(|name| name.starts_with(current.as_ref()))
        .map(clap_complete::CompletionCandidate::new)
        .collect()
}

#[derive(Parser)]
#[command(name = "cloud")]
#[command(about = "Cloud resource management CLI", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Get/list resources (kubectl-style)
    #[command(visible_alias = "ls")]
    Get {
        /// Resource type (cpfs, fs, ...)
        #[arg(add = clap_complete::ArgValueCompleter::new(complete_resource_types))]
        resource: String,
        /// Resource name (if not provided, lists all)
        name: Option<String>,
        /// Provider name
        #[arg(long, short, add = clap_complete::ArgValueCompleter::new(complete_provider_names))]
        provider: Option<String>,
        /// Region override
        #[arg(long, short)]
        region: Option<String>,
        /// Output format
        #[arg(long, short, value_enum, default_value = "table")]
        output: OutputFormat,
        /// Page number
        #[arg(long, default_value = "1")]
        page: i32,
        /// Page size
        #[arg(long, default_value = "10")]
        page_size: i32,
    },
    /// Show detailed resource information
    Describe {
        /// Resource type (cpfs, fs, ...)
        #[arg(add = clap_complete::ArgValueCompleter::new(complete_resource_types))]
        resource: String,
        /// Resource name
        name: String,
        /// Provider name
        #[arg(long, short, add = clap_complete::ArgValueCompleter::new(complete_provider_names))]
        provider: Option<String>,
        /// Region override
        #[arg(long, short)]
        region: Option<String>,
        /// Output format
        #[arg(long, short, value_enum, default_value = "table")]
        output: OutputFormat,
    },
    /// Create a resource
    Create {
        /// Resource type (cpfs, fs, ...)
        #[arg(add = clap_complete::ArgValueCompleter::new(complete_resource_types))]
        resource: Option<String>,
        /// Create from file (JSON/YAML)
        #[arg(long, short)]
        file: Option<PathBuf>,
        /// Provider name
        #[arg(long, short, add = clap_complete::ArgValueCompleter::new(complete_provider_names))]
        provider: Option<String>,
        /// Region override
        #[arg(long, short)]
        region: Option<String>,
        /// Resource options (key=value pairs)
        #[arg(long = "set", short = 's', value_parser = parse_key_value)]
        options: Vec<(String, String)>,
    },
    /// Delete a resource
    Delete {
        /// Resource type (cpfs, fs, ...)
        #[arg(add = clap_complete::ArgValueCompleter::new(complete_resource_types))]
        resource: String,
        /// Resource name
        name: String,
        /// Provider name
        #[arg(long, short, add = clap_complete::ArgValueCompleter::new(complete_provider_names))]
        provider: Option<String>,
        /// Region override
        #[arg(long, short)]
        region: Option<String>,
        /// Skip confirmation
        #[arg(long, short = 'y')]
        force: bool,
    },
    /// Interactive list with pagination
    #[command(visible_alias = "l")]
    List {
        /// Resource type (cpfs, fs, ...)
        #[arg(add = clap_complete::ArgValueCompleter::new(complete_resource_types))]
        resource: String,
        /// Provider name
        #[arg(long, short, add = clap_complete::ArgValueCompleter::new(complete_provider_names))]
        provider: Option<String>,
        /// Region override
        #[arg(long, short)]
        region: Option<String>,
        /// Page size
        #[arg(long, default_value = "10")]
        page_size: i32,
    },
    /// Show and manage configuration
    #[command(visible_alias = "cfg")]
    Config {
        #[command(subcommand)]
        cmd: commands::config::Commands,
    },
    /// Manage providers
    #[command(visible_alias = "prov")]
    Provider {
        #[command(subcommand)]
        cmd: commands::provider::Commands,
    },
    /// Switch current provider (shortcut for 'provider use')
    Use {
        /// Provider name (if not provided, shows current)
        #[arg(add = clap_complete::ArgValueCompleter::new(complete_provider_names))]
        name: Option<String>,
    },
    /// Manage CPFS file systems (legacy, use 'get cpfs' instead)
    #[command(hide = true)]
    Cpfs {
        #[command(subcommand)]
        cmd: commands::cpfs::Commands,
    },
    /// Generate shell completion script
    Completion {
        /// Shell type
        #[arg(value_enum)]
        shell: clap_complete::Shell,
        /// Install completion script
        #[arg(long)]
        install: bool,
    },
}

/// Parse key=value pairs.
fn parse_key_value(s: &str) -> Result<(String, String), String> {
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid format: '{}' (expected key=value)", s))?;
    Ok((s[..pos].to_string(), s[pos + 1..].to_string()))
}

#[tokio::main]
async fn main() -> Result<()> {
    // Handle dynamic shell completion
    CompleteEnv::with_factory(Cli::command).complete();

    let cli = Cli::parse();
    let config_manager = ConfigManager::new()?;

    let Some(command) = cli.command else {
        Cli::command().print_help()?;
        return Ok(());
    };

    match command {
        Commands::Get {
            resource,
            name,
            provider,
            region,
            output,
            page,
            page_size,
        } => {
            commands::get::execute(
                &config_manager,
                &resource,
                name.as_deref(),
                provider.as_deref(),
                region.as_deref(),
                output,
                Some(page),
                Some(page_size),
            )
            .await?;
        }
        Commands::Describe {
            resource,
            name,
            provider,
            region,
            output,
        } => {
            commands::describe::execute(
                &config_manager,
                &resource,
                &name,
                provider.as_deref(),
                region.as_deref(),
                output,
            )
            .await?;
        }
        Commands::Create {
            resource,
            file,
            provider,
            region,
            options,
        } => {
            if let Some(file_path) = file {
                commands::create::execute_from_file(
                    &config_manager,
                    &file_path,
                    provider.as_deref(),
                    region.as_deref(),
                )
                .await?;
            } else if let Some(resource_type) = resource {
                let opts: HashMap<String, String> = options.into_iter().collect();
                commands::create::execute(
                    &config_manager,
                    &resource_type,
                    provider.as_deref(),
                    region.as_deref(),
                    opts,
                )
                .await?;
            } else {
                anyhow::bail!("either resource type or --file must be specified");
            }
        }
        Commands::Delete {
            resource,
            name,
            provider,
            region,
            force,
        } => {
            commands::delete::execute(
                &config_manager,
                &resource,
                &name,
                provider.as_deref(),
                region.as_deref(),
                force,
            )
            .await?;
        }
        Commands::List {
            resource,
            provider,
            region,
            page_size,
        } => {
            commands::list::execute(
                &config_manager,
                &resource,
                provider.as_deref(),
                region.as_deref(),
                page_size,
            )
            .await?;
        }
        Commands::Config { cmd } => commands::config::execute(&config_manager, cmd)?,
        Commands::Provider { cmd } => commands::provider::execute(&config_manager, cmd)?,
        Commands::Use { name } => commands::use_cmd::execute(&config_manager, name)?,
        Commands::Cpfs { cmd } => commands::cpfs::execute(&config_manager, cmd).await?,
        Commands::Completion { shell, install } => {
            if install {
                install_completion(shell)?;
            } else {
                print_completion_instructions(shell);
            }
        }
    }

    Ok(())
}

/// Print instructions for enabling dynamic shell completion.
fn print_completion_instructions(shell: clap_complete::Shell) {
    match shell {
        clap_complete::Shell::Zsh => {
            println!("# Add this to your ~/.zshrc:");
            println!("source <(COMPLETE=zsh cloud)");
        }
        clap_complete::Shell::Bash => {
            println!("# Add this to your ~/.bashrc:");
            println!("source <(COMPLETE=bash cloud)");
        }
        clap_complete::Shell::Fish => {
            println!("# Add this to your ~/.config/fish/config.fish:");
            println!("source (COMPLETE=fish cloud | psub)");
        }
        _ => {
            println!("Dynamic completion for {:?} is not supported.", shell);
            println!("Use: COMPLETE=<shell> cloud");
        }
    }
}

/// Install completion script to appropriate location based on shell.
fn install_completion(shell: clap_complete::Shell) -> Result<()> {
    let completion_line = match shell {
        clap_complete::Shell::Zsh => "source <(COMPLETE=zsh cloud)",
        clap_complete::Shell::Bash => "source <(COMPLETE=bash cloud)",
        clap_complete::Shell::Fish => "source (COMPLETE=fish cloud | psub)",
        _ => anyhow::bail!("--install is only supported for zsh, bash, and fish shells"),
    };

    let (rc_file, rc_name) = match shell {
        clap_complete::Shell::Zsh => {
            let rc = dirs::home_dir()
                .map(|h| h.join(".zshrc"))
                .ok_or_else(|| anyhow::anyhow!("cannot determine home directory"))?;
            (rc, ".zshrc")
        }
        clap_complete::Shell::Bash => {
            let rc = dirs::home_dir()
                .map(|h| h.join(".bashrc"))
                .ok_or_else(|| anyhow::anyhow!("cannot determine home directory"))?;
            (rc, ".bashrc")
        }
        clap_complete::Shell::Fish => {
            let rc = dirs::home_dir()
                .map(|h| h.join(".config").join("fish").join("config.fish"))
                .ok_or_else(|| anyhow::anyhow!("cannot determine home directory"))?;
            if let Some(parent) = rc.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("cannot create directory: {}", parent.display()))?;
            }
            (rc, "config.fish")
        }
        _ => unreachable!(),
    };

    // Check if already installed
    if rc_file.exists() {
        let content = fs::read_to_string(&rc_file)
            .with_context(|| format!("cannot read {}", rc_file.display()))?;
        if content.contains("COMPLETE=") && content.contains("cloud") {
            println!("Completion already installed in {}", rc_name);
            println!("Run 'source ~/{0}' or restart your shell to enable.", rc_name);
            return Ok(());
        }
    }

    // Append completion line
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&rc_file)
        .with_context(|| format!("cannot open {}", rc_file.display()))?;

    use std::io::Write;
    writeln!(file, "\n# cloud CLI completion")?;
    writeln!(file, "{}", completion_line)?;

    println!("Completion installed to {}", rc_file.display());
    println!("Run 'source ~/{0}' or restart your shell to enable.", rc_name);

    Ok(())
}
