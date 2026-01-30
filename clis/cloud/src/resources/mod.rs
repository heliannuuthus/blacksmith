//! Resource abstraction for kubectl-style commands.
//!
//! This module provides a unified interface for managing cloud resources
//! similar to kubectl's `get`, `create`, `delete` commands.

pub mod cpfs;

use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::Provider;

/// Common options for listing resources.
#[derive(Debug, Clone, Default)]
pub struct ListOptions {
    /// Page number (1-based).
    pub page: Option<i32>,
    /// Page size.
    pub page_size: Option<i32>,
    /// Label selector (key=value pairs).
    #[allow(dead_code)]
    pub labels: HashMap<String, String>,
}

/// Common options for output formatting.
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum OutputFormat {
    /// Human-readable table format.
    #[default]
    Table,
    /// JSON format.
    Json,
    /// YAML format.
    Yaml,
    /// Wide table format with more columns.
    Wide,
}

/// A single resource item returned from list/get operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceItem {
    /// Resource kind (e.g., "cpfs", "ecs").
    pub kind: String,
    /// Resource ID/name.
    pub name: String,
    /// Resource status.
    pub status: String,
    /// Region.
    pub region: String,
    /// Creation time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    /// Additional fields for display.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl ResourceItem {
    /// Create a new resource item.
    #[must_use]
    pub fn new(kind: impl Into<String>, name: impl Into<String>, status: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            name: name.into(),
            status: status.into(),
            region: region.into(),
            created_at: None,
            extra: HashMap::new(),
        }
    }

    /// Add an extra field.
    #[must_use]
    pub fn with_extra(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        if let Ok(v) = serde_json::to_value(value) {
            self.extra.insert(key.into(), v);
        }
        self
    }

    /// Set creation time.
    #[must_use]
    pub fn with_created_at(mut self, created_at: impl Into<String>) -> Self {
        self.created_at = Some(created_at.into());
        self
    }
}

/// List response with pagination info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListResult {
    /// Resource items.
    pub items: Vec<ResourceItem>,
    /// Total count.
    pub total: i32,
    /// Current page.
    pub page: i32,
    /// Page size.
    pub page_size: i32,
}

/// Resource specification for create operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSpec {
    /// Resource kind.
    pub kind: String,
    /// Resource configuration.
    #[serde(flatten)]
    pub spec: HashMap<String, serde_json::Value>,
}

/// The core resource trait that all cloud resources must implement.
#[async_trait]
pub trait Resource: Send + Sync {
    /// Get the resource kind name (e.g., "cpfs", "ecs").
    fn kind(&self) -> &'static str;

    /// Get short aliases for this resource (e.g., ["fs"] for cpfs).
    #[allow(dead_code)]
    fn aliases(&self) -> &'static [&'static str] {
        &[]
    }

    /// Get column headers for table output.
    fn columns(&self) -> Vec<&'static str> {
        vec!["NAME", "STATUS", "REGION", "AGE"]
    }

    /// Get wide column headers for wide table output.
    fn wide_columns(&self) -> Vec<&'static str> {
        self.columns()
    }

    /// List resources.
    async fn list(&self, opts: &ListOptions) -> Result<ListResult>;

    /// Get a single resource by ID/name.
    async fn get(&self, name: &str) -> Result<ResourceItem>;

    /// Create a resource from spec.
    async fn create(&self, spec: &ResourceSpec) -> Result<ResourceItem>;

    /// Delete a resource by ID/name.
    async fn delete(&self, name: &str) -> Result<()>;

    /// Format a resource item as table row.
    fn format_row(&self, item: &ResourceItem) -> Vec<String> {
        vec![
            item.name.clone(),
            item.status.clone(),
            item.region.clone(),
            item.created_at.clone().unwrap_or_else(|| "-".to_string()),
        ]
    }

    /// Format a resource item as wide table row.
    fn format_wide_row(&self, item: &ResourceItem) -> Vec<String> {
        self.format_row(item)
    }
}

/// All supported resource kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceKind {
    /// CPFS file system.
    Cpfs,
}

impl ResourceKind {
    /// Get all available resource kinds.
    #[must_use]
    #[allow(dead_code)]
    pub const fn all() -> &'static [Self] {
        &[Self::Cpfs]
    }

    /// Get resource kind from string.
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cpfs" | "fs" => Some(Self::Cpfs),
            _ => None,
        }
    }

    /// Get display name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Cpfs => "cpfs",
        }
    }

    /// Get all names including aliases for shell completion.
    #[must_use]
    pub fn all_names() -> Vec<&'static str> {
        vec!["cpfs", "fs"]
    }
}

impl std::fmt::Display for ResourceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Create a resource handler for the given kind and provider.
///
/// # Errors
/// Returns error if the resource kind is not supported for the provider.
pub fn create_resource(
    kind: ResourceKind,
    provider: &Provider,
    region: Option<&str>,
) -> Result<Box<dyn Resource>> {
    match kind {
        ResourceKind::Cpfs => {
            let resource = cpfs::CpfsResource::new(provider, region)
                .context("failed to create CPFS resource handler")?;
            Ok(Box::new(resource))
        }
    }
}

/// Get resource kind from string, with helpful error message.
///
/// # Errors
/// Returns error if the resource kind is not recognized.
pub fn parse_resource_kind(s: &str) -> Result<ResourceKind> {
    ResourceKind::from_str(s).ok_or_else(|| {
        let available = ResourceKind::all_names().join(", ");
        anyhow::anyhow!("unknown resource type '{}'. Available: {}", s, available)
    })
}
