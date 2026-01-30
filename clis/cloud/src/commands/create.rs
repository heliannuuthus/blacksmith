//! Create command - create resources.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use crate::config::ConfigManager;
use crate::resources::{create_resource, parse_resource_kind, ResourceSpec};

/// Execute create command with resource type and inline options.
///
/// # Errors
/// Returns error if resource creation fails.
pub async fn execute(
    manager: &ConfigManager,
    resource_type: &str,
    provider: Option<&str>,
    region: Option<&str>,
    options: HashMap<String, String>,
) -> Result<()> {
    let kind = parse_resource_kind(resource_type)?;
    let (provider_name, provider_config) = manager.resolve_provider(provider)?;
    let resource = create_resource(kind, &provider_config, region)?;

    // Convert string options to JSON values
    let spec: HashMap<String, serde_json::Value> = options
        .into_iter()
        .map(|(k, v)| {
            // Try to parse as number first
            let value = if let Ok(n) = v.parse::<i64>() {
                serde_json::Value::Number(n.into())
            } else if let Ok(b) = v.parse::<bool>() {
                serde_json::Value::Bool(b)
            } else {
                serde_json::Value::String(v)
            };
            (k, value)
        })
        .collect();

    let resource_spec = ResourceSpec {
        kind: kind.as_str().to_string(),
        spec,
    };

    let item = resource.create(&resource_spec).await?;

    println!("{} '{}' created (provider: {})", resource.kind(), item.name, provider_name);

    Ok(())
}

/// Execute create command from file.
///
/// # Errors
/// Returns error if file reading or resource creation fails.
pub async fn execute_from_file(
    manager: &ConfigManager,
    file_path: &Path,
    provider: Option<&str>,
    region: Option<&str>,
) -> Result<()> {
    let content = fs::read_to_string(file_path)
        .with_context(|| format!("cannot read file: {}", file_path.display()))?;

    // Try to parse as YAML first, then JSON
    let spec: ResourceSpec = if file_path.extension().map_or(false, |e| e == "json") {
        serde_json::from_str(&content)
            .with_context(|| format!("cannot parse JSON file: {}", file_path.display()))?
    } else {
        // For YAML, we'll use JSON parser on YAML-like content
        // In a real implementation, you'd use serde_yaml
        serde_json::from_str(&content)
            .with_context(|| format!("cannot parse file: {}", file_path.display()))?
    };

    let kind = parse_resource_kind(&spec.kind)?;
    let (provider_name, provider_config) = manager.resolve_provider(provider)?;
    let resource = create_resource(kind, &provider_config, region)?;

    let item = resource.create(&spec).await?;

    println!("{} '{}' created (provider: {})", resource.kind(), item.name, provider_name);

    Ok(())
}
