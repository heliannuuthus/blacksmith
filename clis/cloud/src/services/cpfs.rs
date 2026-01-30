//! CPFS (Cloud Parallel File Storage) service trait.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// CPFS file system information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSystem {
    /// File system ID
    pub id: String,
    /// File system name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// File system type
    pub fs_type: String,
    /// Description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Region ID
    pub region: String,
    /// Zone ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zone: Option<String>,
    /// Creation time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    /// Expiration time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    /// Bandwidth (MB/s)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bandwidth: Option<i64>,
    /// Capacity in GB (raw value for formatting)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capacity_gb: Option<i64>,
    /// Status
    pub status: String,
}

/// Fileset information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fileset {
    /// Fileset ID
    pub id: String,
    /// Fileset path
    pub path: String,
    /// Description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Quota in bytes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quota_bytes: Option<i64>,
    /// Used bytes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub used_bytes: Option<i64>,
    /// File count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_count: Option<i64>,
    /// Directory count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dir_count: Option<i64>,
    /// Creation time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    /// Update time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    /// Status
    pub status: String,
}

/// Fileset list response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesetListResponse {
    /// Filesets
    pub items: Vec<Fileset>,
    /// Total count
    pub total: i32,
    /// Page number
    pub page: i32,
    /// Page size
    pub page_size: i32,
}

/// CPFS list response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListResponse {
    /// File systems
    pub items: Vec<FileSystem>,
    /// Total count
    pub total: i32,
    /// Page number
    pub page: i32,
    /// Page size
    pub page_size: i32,
}

/// CPFS create request.
#[derive(Debug, Clone)]
pub struct CreateRequest {
    /// Protocol type (e.g., NFS)
    pub protocol_type: String,
    /// Storage type (e.g., standard)
    pub storage_type: String,
    /// Capacity in GB
    pub capacity: i64,
    /// Zone ID
    #[allow(dead_code)] // Will be used when implementing actual API calls
    pub zone: Option<String>,
    /// Description
    #[allow(dead_code)] // Will be used when implementing actual API calls
    pub description: Option<String>,
}

/// CPFS backend trait - unified interface for different providers.
#[async_trait]
pub trait CpfsBackend: Send + Sync {
    /// List file systems.
    async fn list(&self, page: Option<i32>, page_size: Option<i32>) -> Result<ListResponse>;

    /// Get file system by ID.
    async fn get(&self, id: &str) -> Result<FileSystem>;

    /// Create a file system.
    async fn create(&self, req: CreateRequest) -> Result<FileSystem>;

    /// Delete a file system.
    async fn delete(&self, id: &str) -> Result<()>;

    /// List filesets for a file system.
    async fn list_filesets(
        &self,
        fs_id: &str,
        page: Option<i32>,
        page_size: Option<i32>,
    ) -> Result<FilesetListResponse>;

    /// Delete a fileset.
    async fn delete_fileset(&self, fs_id: &str, fileset_id: &str) -> Result<()>;
}

/// Format capacity with appropriate unit (4 significant digits).
/// Input is in GB.
pub fn format_capacity(gb: i64) -> String {
    const KB: f64 = 1024.0;

    let gb_f = gb as f64;

    if gb_f >= KB * KB {
        // PB range
        let pb = gb_f / (KB * KB);
        format!("{:.4} PB", pb)
    } else if gb_f >= KB {
        // TB range
        let tb = gb_f / KB;
        format!("{:.4} TB", tb)
    } else {
        // GB range
        format!("{:.4} GB", gb_f)
    }
}

/// Format bytes with appropriate unit (4 significant digits).
pub fn format_bytes(bytes: i64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * KB;
    const GB: f64 = MB * KB;
    const TB: f64 = GB * KB;
    const PB: f64 = TB * KB;

    let b = bytes as f64;

    if b >= PB {
        format!("{:.4} PB", b / PB)
    } else if b >= TB {
        format!("{:.4} TB", b / TB)
    } else if b >= GB {
        format!("{:.4} GB", b / GB)
    } else if b >= MB {
        format!("{:.4} MB", b / MB)
    } else if b >= KB {
        format!("{:.4} KB", b / KB)
    } else {
        format!("{} B", bytes)
    }
}
