//! Delete command - delete resources.

use anyhow::Result;
use dialoguer::Confirm;

use crate::config::ConfigManager;
use crate::resources::{create_resource, parse_resource_kind};

/// Execute delete command.
///
/// # Errors
/// Returns error if resource deletion fails.
pub async fn execute(
    manager: &ConfigManager,
    resource_type: &str,
    name: &str,
    provider: Option<&str>,
    region: Option<&str>,
    force: bool,
) -> Result<()> {
    let kind = parse_resource_kind(resource_type)?;
    let (provider_name, provider_config) = manager.resolve_provider(provider)?;
    let resource = create_resource(kind, &provider_config, region)?;

    // Confirm deletion unless --force is specified
    if !force {
        let prompt = format!(
            "Are you sure you want to delete {} '{}' from provider '{}'?",
            resource.kind(),
            name,
            provider_name
        );

        let confirmed = Confirm::new()
            .with_prompt(prompt)
            .default(false)
            .interact()
            .unwrap_or(false);

        if !confirmed {
            println!("Deletion cancelled.");
            return Ok(());
        }
    }

    resource.delete(name).await?;

    println!("{} '{}' deleted (provider: {})", resource.kind(), name, provider_name);

    Ok(())
}
