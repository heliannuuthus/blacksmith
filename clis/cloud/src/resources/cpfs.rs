//! CPFS resource implementation.

use anyhow::{Context, Result};
use async_trait::async_trait;

use super::{ListOptions, ListResult, Resource, ResourceItem, ResourceSpec};
use crate::backends::create_cpfs_backend;
use crate::config::Provider;
use crate::services::cpfs::{format_capacity, CpfsBackend, CreateRequest};

/// CPFS resource handler.
pub struct CpfsResource {
    backend: Box<dyn CpfsBackend>,
    #[allow(dead_code)]
    region: String,
}

impl CpfsResource {
    /// Create a new CPFS resource handler.
    ///
    /// # Errors
    /// Returns error if backend creation fails.
    pub fn new(provider: &Provider, region: Option<&str>) -> Result<Self> {
        let effective_region = region
            .map(String::from)
            .or_else(|| provider.region.clone())
            .unwrap_or_else(|| "unknown".to_string());

        let backend = create_cpfs_backend(provider, region)?;

        Ok(Self {
            backend,
            region: effective_region,
        })
    }
}

#[async_trait]
impl Resource for CpfsResource {
    fn kind(&self) -> &'static str {
        "cpfs"
    }

    fn aliases(&self) -> &'static [&'static str] {
        &["fs"]
    }

    fn columns(&self) -> Vec<&'static str> {
        vec!["ID", "NAME", "TYPE", "STATUS", "CAPACITY", "CREATED"]
    }

    fn wide_columns(&self) -> Vec<&'static str> {
        vec!["ID", "NAME", "TYPE", "STATUS", "REGION", "ZONE", "CAPACITY", "BANDWIDTH", "CREATED"]
    }

    async fn list(&self, opts: &ListOptions) -> Result<ListResult> {
        let response = self
            .backend
            .list(opts.page, opts.page_size)
            .await
            .context("failed to list CPFS file systems")?;

        let items = response
            .items
            .into_iter()
            .map(|fs| {
                let mut item = ResourceItem::new("cpfs", &fs.id, &fs.status, &fs.region)
                    .with_extra("type", &fs.fs_type);

                if let Some(name) = &fs.name {
                    item = item.with_extra("display_name", name);
                }
                if let Some(capacity_gb) = fs.capacity_gb {
                    item = item
                        .with_extra("capacity", format_capacity(capacity_gb))
                        .with_extra("capacity_gb", capacity_gb);
                }
                if let Some(bandwidth) = fs.bandwidth {
                    item = item.with_extra("bandwidth", format!("{} MB/s", bandwidth));
                }
                if let Some(zone) = &fs.zone {
                    item = item.with_extra("zone", zone);
                }
                if let Some(created_at) = &fs.created_at {
                    item = item.with_created_at(created_at);
                }
                item
            })
            .collect();

        Ok(ListResult {
            items,
            total: response.total,
            page: response.page,
            page_size: response.page_size,
        })
    }

    async fn get(&self, name: &str) -> Result<ResourceItem> {
        let fs = self
            .backend
            .get(name)
            .await
            .with_context(|| format!("failed to get CPFS file system '{}'", name))?;

        let mut item = ResourceItem::new("cpfs", &fs.id, &fs.status, &fs.region)
            .with_extra("type", &fs.fs_type);

        if let Some(name) = &fs.name {
            item = item.with_extra("display_name", name);
        }
        if let Some(desc) = &fs.description {
            item = item.with_extra("description", desc);
        }
        if let Some(capacity_gb) = fs.capacity_gb {
            item = item
                .with_extra("capacity", format_capacity(capacity_gb))
                .with_extra("capacity_gb", capacity_gb);
        }
        if let Some(bandwidth) = fs.bandwidth {
            item = item.with_extra("bandwidth", format!("{} MB/s", bandwidth));
        }
        if let Some(zone) = &fs.zone {
            item = item.with_extra("zone", zone);
        }
        if let Some(created_at) = &fs.created_at {
            item = item.with_created_at(created_at);
        }
        if let Some(expires_at) = &fs.expires_at {
            item = item.with_extra("expires_at", expires_at);
        }

        Ok(item)
    }

    async fn create(&self, spec: &ResourceSpec) -> Result<ResourceItem> {
        let protocol_type = spec
            .spec
            .get("protocol_type")
            .and_then(|v| v.as_str())
            .unwrap_or("NFS")
            .to_string();

        let storage_type = spec
            .spec
            .get("storage_type")
            .and_then(|v| v.as_str())
            .unwrap_or("standard")
            .to_string();

        let capacity = spec
            .spec
            .get("capacity")
            .and_then(|v| v.as_i64())
            .unwrap_or(100);

        let zone = spec
            .spec
            .get("zone")
            .and_then(|v| v.as_str())
            .map(String::from);

        let description = spec
            .spec
            .get("description")
            .and_then(|v| v.as_str())
            .map(String::from);

        let req = CreateRequest {
            protocol_type,
            storage_type,
            capacity,
            zone,
            description,
        };

        let fs = self
            .backend
            .create(req)
            .await
            .context("failed to create CPFS file system")?;

        let mut item = ResourceItem::new("cpfs", &fs.id, &fs.status, &fs.region)
            .with_extra("type", &fs.fs_type);

        if let Some(capacity_gb) = fs.capacity_gb {
            item = item.with_extra("capacity", format_capacity(capacity_gb));
        }

        Ok(item)
    }

    async fn delete(&self, name: &str) -> Result<()> {
        self.backend
            .delete(name)
            .await
            .with_context(|| format!("failed to delete CPFS file system '{}'", name))
    }

    fn format_row(&self, item: &ResourceItem) -> Vec<String> {
        vec![
            item.name.clone(), // ID
            item.extra.get("display_name").and_then(|v| v.as_str()).unwrap_or("-").to_string(), // Name
            item.extra.get("type").and_then(|v| v.as_str()).unwrap_or("-").to_string(),
            item.status.clone(),
            item.extra.get("capacity").and_then(|v| v.as_str()).unwrap_or("-").to_string(),
            item.created_at.clone().unwrap_or_else(|| "-".to_string()),
        ]
    }

    fn format_wide_row(&self, item: &ResourceItem) -> Vec<String> {
        vec![
            item.name.clone(), // ID
            item.extra.get("display_name").and_then(|v| v.as_str()).unwrap_or("-").to_string(), // Name
            item.extra.get("type").and_then(|v| v.as_str()).unwrap_or("-").to_string(),
            item.status.clone(),
            item.region.clone(),
            item.extra.get("zone").and_then(|v| v.as_str()).unwrap_or("-").to_string(),
            item.extra.get("capacity").and_then(|v| v.as_str()).unwrap_or("-").to_string(),
            item.extra.get("bandwidth").and_then(|v| v.as_str()).unwrap_or("-").to_string(),
            item.created_at.clone().unwrap_or_else(|| "-".to_string()),
        ]
    }
}
