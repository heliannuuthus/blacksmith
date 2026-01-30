//! Provider management commands.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Subcommand;
use dialoguer::{Input, Select};

use crate::config::{Auth, ConfigManager, Provider, ProviderType};

/// Provider subcommands.
#[derive(Subcommand)]
pub enum Commands {
    /// Add a new provider
    Add {
        /// Provider name
        name: String,
        /// Configuration file path (if not provided, interactive mode will be used)
        #[arg(long)]
        file: Option<PathBuf>,
    },
    /// Remove a provider
    Remove {
        /// Provider name
        name: String,
    },
    /// Rename a provider
    Rename {
        /// Old name
        old: String,
        /// New name
        new: String,
    },
    /// Show provider details
    Show {
        /// Provider name
        name: String,
    },
    /// List all providers
    List,
    /// Create example provider configuration file
    CreateExample {
        /// Cloud provider type
        #[arg(long, default_value = "aliyun")]
        provider_type: String,
        /// Output file path
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

/// Execute provider command.
pub fn execute(manager: &ConfigManager, cmd: Commands) -> Result<()> {
    match cmd {
        Commands::Add { name, file } => {
            if manager.get_provider(&name)?.is_some() {
                anyhow::bail!("provider '{}' already exists", name);
            }

            let provider = if let Some(file_path) = file {
                load_provider_from_file(&file_path)?
            } else {
                interactive_add_provider()?
            };

            manager.set_provider(&name, provider)?;
            println!("Provider '{}' added successfully", name);
        }
        Commands::Remove { name } => {
            if manager.remove_provider(&name)? {
                println!("Provider '{}' removed", name);
            } else {
                println!("Provider '{}' not found", name);
            }
        }
        Commands::Rename { old, new } => {
            manager.rename_provider(&old, &new)?;
            println!("Provider renamed: {} -> {}", old, new);
        }
        Commands::Show { name } => {
            if let Some(provider) = manager.get_provider(&name)? {
                print_provider(&name, &provider, manager.current()?.as_deref());
            } else {
                println!("Provider '{}' not found", name);
            }
        }
        Commands::List => {
            let config = manager.load()?;
            if config.providers.is_empty() {
                println!("No providers configured");
                println!("\nAdd a provider with: cloud provider add <name>");
            } else {
                println!("Providers:\n");
                for (name, provider) in &config.providers {
                    let current_marker = if config.current.as_deref() == Some(name) {
                        " (current)"
                    } else {
                        ""
                    };
                    println!(
                        "  {}{} [{}] {}",
                        name,
                        current_marker,
                        provider.provider_type,
                        provider.region.as_deref().unwrap_or("-")
                    );
                }
            }
        }
        Commands::CreateExample { provider_type, output } => {
            create_example_file(&provider_type, output)?;
        }
    }
    Ok(())
}

fn load_provider_from_file(path: &PathBuf) -> Result<Provider> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("cannot read file: {}", path.display()))?;

    if let Ok(provider) = toml::from_str::<Provider>(&content) {
        return Ok(provider);
    }

    if let Ok(provider) = serde_json::from_str::<Provider>(&content) {
        return Ok(provider);
    }

    anyhow::bail!("cannot parse file as TOML or JSON: {}", path.display())
}

fn interactive_add_provider() -> Result<Provider> {
    println!("Adding new provider (interactive mode)\n");

    let type_names = ProviderType::display_names();
    let type_selection = Select::new()
        .with_prompt("Select cloud provider type")
        .items(&type_names)
        .default(0)
        .interact()
        .context("failed to get user input")?;

    let provider_type = ProviderType::all()[type_selection].clone();

    let auth_types = Auth::type_names();
    let auth_selection = Select::new()
        .with_prompt("Select authentication type")
        .items(auth_types)
        .default(0)
        .interact()
        .context("failed to get user input")?;

    let auth = match auth_selection {
        0 => prompt_access_key()?,
        1 => prompt_token()?,
        2 => prompt_oauth2()?,
        _ => unreachable!(),
    };

    let region: String = Input::new()
        .with_prompt("Default region (optional, press Enter to skip)")
        .allow_empty(true)
        .interact_text()
        .context("failed to get region")?;

    let region = if region.is_empty() { None } else { Some(region) };

    Ok(Provider::new(provider_type, auth, region))
}

fn prompt_access_key() -> Result<Auth> {
    let access_key_id: String = Input::new()
        .with_prompt("Access Key ID")
        .interact_text()
        .context("failed to get access key ID")?;

    let access_key_secret: String = Input::new()
        .with_prompt("Access Key Secret")
        .interact_text()
        .context("failed to get access key secret")?;

    Ok(Auth::AccessKey {
        access_key_id,
        access_key_secret,
    })
}

fn prompt_token() -> Result<Auth> {
    let token: String = Input::new()
        .with_prompt("Token")
        .interact_text()
        .context("failed to get token")?;

    Ok(Auth::Token { token })
}

fn prompt_oauth2() -> Result<Auth> {
    let client_id: String = Input::new()
        .with_prompt("Client ID")
        .interact_text()
        .context("failed to get client ID")?;

    let client_secret: String = Input::new()
        .with_prompt("Client Secret")
        .interact_text()
        .context("failed to get client secret")?;

    let refresh_token: String = Input::new()
        .with_prompt("Refresh Token")
        .interact_text()
        .context("failed to get refresh token")?;

    Ok(Auth::OAuth2 {
        client_id,
        client_secret,
        refresh_token,
    })
}

fn print_provider(name: &str, provider: &Provider, current: Option<&str>) {
    let current_marker = if current == Some(name) { " (current)" } else { "" };

    println!("Provider: {}{}", name, current_marker);
    println!("  Type: {}", provider.provider_type);
    if let Some(region) = &provider.region {
        println!("  Region: {}", region);
    }

    match &provider.auth {
        Auth::AccessKey { access_key_id, access_key_secret } => {
            println!("  Auth: AccessKey");
            println!("    Access Key ID: {}", access_key_id);
            println!("    Access Key Secret: {}", "*".repeat(access_key_secret.len()));
        }
        Auth::Token { token } => {
            println!("  Auth: Token");
            println!("    Token: {}", "*".repeat(token.len()));
        }
        Auth::OAuth2 { client_id, client_secret, refresh_token } => {
            println!("  Auth: OAuth2");
            println!("    Client ID: {}", client_id);
            println!("    Client Secret: {}", "*".repeat(client_secret.len()));
            println!("    Refresh Token: {}", "*".repeat(refresh_token.len()));
        }
    }
}

fn create_example_file(provider_type_str: &str, output: Option<PathBuf>) -> Result<()> {
    let provider_type = match provider_type_str.to_lowercase().as_str() {
        "aliyun" => ProviderType::Aliyun,
        "aws" => ProviderType::Aws,
        "tencent" => ProviderType::Tencent,
        "gcp" => ProviderType::Gcp,
        "azure" => ProviderType::Azure,
        _ => anyhow::bail!("unknown provider type: {}", provider_type_str),
    };

    let default_region = match provider_type {
        ProviderType::Aliyun => "cn-hangzhou",
        ProviderType::Aws => "us-east-1",
        ProviderType::Tencent => "ap-beijing",
        ProviderType::Gcp => "us-central1",
        ProviderType::Azure => "eastus",
    };

    let example = Provider::new(
        provider_type,
        Auth::AccessKey {
            access_key_id: "YOUR_ACCESS_KEY_ID".to_owned(),
            access_key_secret: "YOUR_ACCESS_KEY_SECRET".to_owned(),
        },
        Some(default_region.to_owned()),
    );

    let content = toml::to_string_pretty(&example).context("cannot serialize example")?;

    if let Some(path) = output {
        fs::write(&path, &content)
            .with_context(|| format!("cannot write file: {}", path.display()))?;
        println!("Example configuration created: {}", path.display());
        println!("\nEdit the file and use: cloud provider add <name> --file {}", path.display());
    } else {
        println!("{}", content);
    }

    Ok(())
}
