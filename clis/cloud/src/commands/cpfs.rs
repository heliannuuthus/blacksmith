//! CPFS management commands.

use anyhow::Result;
use clap::Subcommand;

use crate::backends::create_cpfs_backend;
use crate::config::ConfigManager;
use crate::services::cpfs::{CreateRequest, FileSystem};

/// CPFS subcommands.
#[derive(Subcommand)]
pub enum Commands {
    /// List file systems
    List {
        /// Provider name (defaults to current provider)
        #[arg(long)]
        provider: Option<String>,
        /// Region override
        #[arg(long)]
        region: Option<String>,
        /// Page number
        #[arg(long, default_value = "1")]
        page: i32,
        /// Page size
        #[arg(long, default_value = "10")]
        page_size: i32,
    },
    /// Get file system details
    Get {
        /// File system ID
        id: String,
        /// Provider name (defaults to current provider)
        #[arg(long)]
        provider: Option<String>,
        /// Region override
        #[arg(long)]
        region: Option<String>,
    },
    /// Create a file system
    Create {
        /// Protocol type (e.g., NFS)
        protocol_type: String,
        /// Storage type (e.g., standard)
        storage_type: String,
        /// Capacity in GB
        capacity: i64,
        /// Provider name (defaults to current provider)
        #[arg(long)]
        provider: Option<String>,
        /// Region override
        #[arg(long)]
        region: Option<String>,
        /// Zone ID
        #[arg(long)]
        zone: Option<String>,
        /// Description
        #[arg(long)]
        description: Option<String>,
    },
    /// Delete a file system
    Delete {
        /// File system ID
        id: String,
        /// Provider name (defaults to current provider)
        #[arg(long)]
        provider: Option<String>,
        /// Region override
        #[arg(long)]
        region: Option<String>,
    },
}

/// Execute CPFS command.
///
/// # Errors
/// Returns error if command execution fails.
pub async fn execute(manager: &ConfigManager, cmd: Commands) -> Result<()> {
    match cmd {
        Commands::List {
            provider,
            region,
            page,
            page_size,
        } => {
            let (name, provider_config) = manager.resolve_provider(provider.as_deref())?;
            let backend = create_cpfs_backend(&provider_config, region.as_deref())?;

            let response = backend.list(Some(page), Some(page_size)).await?;

            println!("File Systems (provider: {name}, total: {}):", response.total);
            if response.total > 0 {
                println!(
                    "Page: {}/{}",
                    response.page,
                    (response.total + response.page_size - 1) / response.page_size
                );
            }
            println!();

            if response.items.is_empty() {
                println!("No file systems found");
            } else {
                for fs in &response.items {
                    print_file_system(fs);
                }
            }
        }
        Commands::Get {
            id,
            provider,
            region,
        } => {
            let (name, provider_config) = manager.resolve_provider(provider.as_deref())?;
            let backend = create_cpfs_backend(&provider_config, region.as_deref())?;

            let fs = backend.get(&id).await?;
            println!("File System (provider: {name}):\n");
            print_file_system_detail(&fs);
        }
        Commands::Create {
            protocol_type,
            storage_type,
            capacity,
            provider,
            region,
            zone,
            description,
        } => {
            let (name, provider_config) = manager.resolve_provider(provider.as_deref())?;
            let backend = create_cpfs_backend(&provider_config, region.as_deref())?;

            let req = CreateRequest {
                protocol_type,
                storage_type,
                capacity,
                zone,
                description,
            };

            let fs = backend.create(req).await?;
            println!("File system created (provider: {name}):\n");
            print_file_system(&fs);
        }
        Commands::Delete {
            id,
            provider,
            region,
        } => {
            let (name, provider_config) = manager.resolve_provider(provider.as_deref())?;
            let backend = create_cpfs_backend(&provider_config, region.as_deref())?;

            backend.delete(&id).await?;
            println!("File system deleted: {id} (provider: {name})");
        }
    }
    Ok(())
}

fn print_file_system(fs: &FileSystem) {
    println!("ID: {}", fs.id);
    if let Some(ref name) = fs.name {
        println!("  Name: {name}");
    }
    println!("  Type: {}", fs.fs_type);
    println!("  Status: {}", fs.status);
    println!("  Region: {}", fs.region);
    if let Some(ref desc) = fs.description {
        println!("  Description: {desc}");
    }
    if let Some(capacity_gb) = fs.capacity_gb {
        println!("  Capacity: {}", crate::services::cpfs::format_capacity(capacity_gb));
    }
    if let Some(bandwidth) = fs.bandwidth {
        println!("  Bandwidth: {bandwidth} MB/s");
    }
    println!();
}

fn print_file_system_detail(fs: &FileSystem) {
    println!("ID: {}", fs.id);
    if let Some(ref name) = fs.name {
        println!("Name: {name}");
    }
    println!("Type: {}", fs.fs_type);
    println!("Status: {}", fs.status);
    println!("Region: {}", fs.region);
    if let Some(ref zone) = fs.zone {
        println!("Zone: {zone}");
    }
    if let Some(ref desc) = fs.description {
        println!("Description: {desc}");
    }
    if let Some(capacity_gb) = fs.capacity_gb {
        println!("Capacity: {}", crate::services::cpfs::format_capacity(capacity_gb));
    }
    if let Some(bandwidth) = fs.bandwidth {
        println!("Bandwidth: {bandwidth} MB/s");
    }
    if let Some(ref created_at) = fs.created_at {
        println!("Created: {created_at}");
    }
    if let Some(ref expires_at) = fs.expires_at {
        println!("Expires: {expires_at}");
    }
}
