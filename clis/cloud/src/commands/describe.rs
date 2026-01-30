//! Describe command - show detailed resource information.

use anyhow::Result;

use crate::config::ConfigManager;
use crate::resources::{create_resource, parse_resource_kind, OutputFormat};

/// Execute describe command.
///
/// # Errors
/// Returns error if resource operations fail.
pub async fn execute(
    manager: &ConfigManager,
    resource_type: &str,
    name: &str,
    provider: Option<&str>,
    region: Option<&str>,
    output: OutputFormat,
) -> Result<()> {
    let kind = parse_resource_kind(resource_type)?;
    let (provider_name, provider_config) = manager.resolve_provider(provider)?;
    let resource = create_resource(kind, &provider_config, region)?;

    let item = resource.get(name).await?;

    match output {
        OutputFormat::Json => {
            if let Ok(json) = serde_json::to_string_pretty(&item) {
                println!("{}", json);
            }
        }
        OutputFormat::Yaml => {
            println!("kind: {}", item.kind);
            println!("name: {}", item.name);
            println!("status: {}", item.status);
            println!("region: {}", item.region);
            if let Some(ref created_at) = item.created_at {
                println!("created_at: {}", created_at);
            }
            for (k, v) in &item.extra {
                if let Some(s) = v.as_str() {
                    println!("{}: {}", k, s);
                } else {
                    println!("{}: {}", k, v);
                }
            }
        }
        OutputFormat::Table | OutputFormat::Wide => {
            println!("Name:        {}", item.name);
            println!("Kind:        {}", item.kind);
            println!("Status:      {}", item.status);
            println!("Region:      {}", item.region);
            println!("Provider:    {}", provider_name);
            
            if let Some(ref created_at) = item.created_at {
                println!("Created:     {}", created_at);
            }

            println!();
            println!("Details:");

            // Sort extra fields for consistent output
            let mut extras: Vec<_> = item.extra.iter().collect();
            extras.sort_by_key(|(k, _)| *k);

            for (k, v) in extras {
                let key = k.replace('_', " ");
                let key = key
                    .split_whitespace()
                    .map(|s| {
                        let mut c = s.chars();
                        match c.next() {
                            None => String::new(),
                            Some(f) => f.to_uppercase().chain(c).collect(),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ");

                if let Some(s) = v.as_str() {
                    println!("  {}: {}", key, s);
                } else {
                    println!("  {}: {}", key, v);
                }
            }
        }
    }

    Ok(())
}
