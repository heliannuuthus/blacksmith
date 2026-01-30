//! Get command - list or get resources (kubectl-style).

use anyhow::Result;
use console::style;
use tabled::builder::Builder;
use tabled::settings::object::{Columns, Rows};
use tabled::settings::{Color, Modify, Style};

use crate::config::ConfigManager;
use crate::resources::{create_resource, parse_resource_kind, ListOptions, OutputFormat, Resource, ResourceItem};

/// Execute get command.
///
/// # Errors
/// Returns error if resource operations fail.
pub async fn execute(
    manager: &ConfigManager,
    resource_type: &str,
    name: Option<&str>,
    provider: Option<&str>,
    region: Option<&str>,
    output: OutputFormat,
    page: Option<i32>,
    page_size: Option<i32>,
) -> Result<()> {
    let kind = parse_resource_kind(resource_type)?;
    let (provider_name, provider_config) = manager.resolve_provider(provider)?;
    let resource = create_resource(kind, &provider_config, region)?;

    if let Some(name) = name {
        // Get single resource
        let item = resource.get(name).await?;
        print_single(&*resource, &item, output, &provider_name);
    } else {
        // List resources
        let opts = ListOptions {
            page,
            page_size,
            ..Default::default()
        };
        let result = resource.list(&opts).await?;
        print_list(&*resource, &result.items, output, &provider_name, result.total, result.page, result.page_size);
    }

    Ok(())
}

fn print_single(resource: &dyn Resource, item: &ResourceItem, output: OutputFormat, provider: &str) {
    match output {
        OutputFormat::Json => {
            if let Ok(json) = serde_json::to_string_pretty(item) {
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
            println!(
                "\n {} {} {}\n",
                style("📋").bold(),
                style(resource.kind().to_uppercase()).bold().cyan(),
                style(format!("(provider: {})", provider)).dim()
            );

            let columns = if matches!(output, OutputFormat::Wide) {
                resource.wide_columns()
            } else {
                resource.columns()
            };

            let row = if matches!(output, OutputFormat::Wide) {
                resource.format_wide_row(item)
            } else {
                resource.format_row(item)
            };

            let mut builder = Builder::default();
            for (col, val) in columns.iter().zip(row.iter()) {
                builder.push_record([col.to_string(), val.clone()]);
            }

            // Add extra fields
            for (k, v) in &item.extra {
                let key = k.to_uppercase().replace('_', " ");
                if !columns.iter().any(|c| c.to_uppercase() == key) {
                    if let Some(s) = v.as_str() {
                        builder.push_record([key, s.to_string()]);
                    }
                }
            }

            let mut table = builder.build();
            table
                .with(Style::rounded())
                .with(Modify::new(Columns::first()).with(Color::FG_CYAN));

            println!("{table}");
        }
    }
}

fn print_list(
    resource: &dyn Resource,
    items: &[ResourceItem],
    output: OutputFormat,
    provider: &str,
    total: i32,
    page: i32,
    page_size: i32,
) {
    match output {
        OutputFormat::Json => {
            if let Ok(json) = serde_json::to_string_pretty(items) {
                println!("{}", json);
            }
        }
        OutputFormat::Yaml => {
            println!("items:");
            for item in items {
                println!("- name: {}", item.name);
                println!("  kind: {}", item.kind);
                println!("  status: {}", item.status);
                println!("  region: {}", item.region);
            }
        }
        OutputFormat::Table | OutputFormat::Wide => {
            let wide = matches!(output, OutputFormat::Wide);
            let columns = if wide { resource.wide_columns() } else { resource.columns() };

            // Print header info
            if total > 0 {
                let total_pages = (total + page_size - 1) / page_size;
                println!(
                    "\n {} {} {} {}\n",
                    style("📦").bold(),
                    style(resource.kind().to_uppercase()).bold().cyan(),
                    style(format!("(provider: {})", provider)).dim(),
                    style(format!("[{}/{} of {}]", page, total_pages, total)).yellow()
                );
            } else {
                println!(
                    "\n {} {} {}\n",
                    style("📦").bold(),
                    style(resource.kind().to_uppercase()).bold().cyan(),
                    style(format!("(provider: {})", provider)).dim()
                );
            }

            if items.is_empty() {
                println!("{}", style("No resources found.").dim());
                return;
            }

            let mut builder = Builder::default();
            builder.push_record(columns.iter().map(|s| s.to_string()));

            for item in items {
                let row = if wide {
                    resource.format_wide_row(item)
                } else {
                    resource.format_row(item)
                };
                builder.push_record(row);
            }

            let mut table = builder.build();
            table
                .with(Style::rounded())
                .with(Modify::new(Rows::first()).with(Color::FG_CYAN))
                .with(Modify::new(Columns::first()).with(Color::FG_WHITE));

            // Color status column based on value
            for (i, item) in items.iter().enumerate() {
                let row_idx = i + 1;
                let color = match item.status.to_lowercase().as_str() {
                    "running" | "active" | "available" => Color::FG_GREEN,
                    "stopped" | "inactive" => Color::FG_YELLOW,
                    "error" | "failed" | "deleted" => Color::FG_RED,
                    "creating" | "pending" | "updating" => Color::FG_BLUE,
                    _ => Color::FG_WHITE,
                };
                table.with(Modify::new((row_idx, 2)).with(color));
            }

            println!("{table}");
        }
    }
}
