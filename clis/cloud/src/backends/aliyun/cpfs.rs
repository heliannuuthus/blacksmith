//! Aliyun CPFS/NAS backend.
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use super::client::AliyunClient;
use super::AliyunConfig;
use crate::services::cpfs::{CpfsBackend, CreateRequest, FileSystem, Fileset, FilesetListResponse, ListResponse};

const API_VERSION: &str = "2017-06-26";
const LOG_FILE: &str = "/tmp/cloud-cli-debug.log";

/// Write debug log to /tmp for troubleshooting API issues.
fn write_debug_log(action: &str, request_params: &BTreeMap<String, String>, response: &str) {
    let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    let log_entry = format!(
        "\n[{}] Action: {}\nParams: {:?}\nResponse:\n{}\n{}\n",
        timestamp, action, request_params, response, "=".repeat(80)
    );
    
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(LOG_FILE)
    {
        let _ = file.write_all(log_entry.as_bytes());
    }
}

/// Parse API response with logging on failure.
fn parse_response<T: serde::de::DeserializeOwned>(
    action: &str,
    params: &BTreeMap<String, String>,
    response: &str,
) -> Result<T> {
    match serde_json::from_str::<T>(response) {
        Ok(v) => Ok(v),
        Err(e) => {
            write_debug_log(action, params, response);
            Err(anyhow::anyhow!(
                "API parse error: {}. See {} for details.",
                e, LOG_FILE
            ))
        }
    }
}

pub struct AliyunCpfsBackend { client: AliyunClient }

impl AliyunCpfsBackend {
    #[must_use]
    pub fn new(config: AliyunConfig) -> Self {
        let endpoint = format!("nas.{}.aliyuncs.com", config.region);
        Self { client: AliyunClient::new(config.access_key_id, config.access_key_secret, endpoint) }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct ListResp { total_count: i32, page_size: i32, page_number: i32, file_systems: FsWrap }

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct FsWrap { file_system: Vec<AliyunFs> }

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct AliyunFs {
    file_system_id: String, file_system_type: String, #[serde(default)] description: Option<String>,
    region_id: String, #[serde(default)] zone_id: Option<String>, protocol_type: String,
    #[allow(dead_code)] #[serde(default)] storage_type: Option<String>, status: String, create_time: String,
    #[serde(default)] capacity: Option<i64>, #[serde(default)] bandwidth: Option<i64>,
    #[serde(default)] expire_time: Option<String>,
}

impl From<AliyunFs> for FileSystem {
    fn from(fs: AliyunFs) -> Self {
        Self { id: fs.file_system_id, name: fs.description.clone(),
            fs_type: format!("{}/{}", fs.file_system_type, fs.protocol_type),
            description: fs.description, region: fs.region_id, zone: fs.zone_id,
            created_at: Some(fs.create_time), expires_at: fs.expire_time,
            bandwidth: fs.bandwidth, capacity_gb: fs.capacity, status: fs.status }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct CreateResp { file_system_id: String }

// Fileset API structures
// Aliyun returns: {"TotalCount":26,"Entries":{"Entrie":[{...},{...}]}}
// Note: The field name in Entries is "Entrie" (without 's'), which is a typo in Aliyun's API
#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct FilesetListResp {
    total_count: i32,
    #[allow(dead_code)]
    next_token: Option<String>,
    #[serde(default)]
    entries: Option<FilesetEntries>,
}

#[derive(Deserialize, Debug, Default)]
#[serde(rename_all = "PascalCase")]
struct FilesetEntries {
    /// Note: Aliyun API uses "Entrie" (typo) instead of "Entry" or "Entries"
    #[serde(default, alias = "Entry")]
    entrie: Vec<AliyunFileset>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct AliyunFileset {
    fset_id: String,
    #[serde(default)]
    file_system_path: Option<String>,
    #[serde(default)]
    description: Option<String>,
    /// Quota is a nested object with SizeLimit and FileCountLimit
    #[serde(default)]
    quota: Option<FilesetQuota>,
    /// SpaceUsage in bytes (actual used space)
    #[serde(default)]
    space_usage: Option<i64>,
    #[allow(dead_code)]
    #[serde(default)]
    file_count_usage: Option<i64>,
    #[serde(default)]
    create_time: Option<String>,
    #[serde(default)]
    update_time: Option<String>,
    #[serde(default)]
    status: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
#[serde(rename_all = "PascalCase")]
struct FilesetQuota {
    #[serde(default)]
    size_limit: Option<i64>,
    #[serde(default)]
    file_count_limit: Option<i64>,
}

impl From<AliyunFileset> for Fileset {
    fn from(f: AliyunFileset) -> Self {
        Self {
            id: f.fset_id,
            path: f.file_system_path.unwrap_or_else(|| "-".to_string()),
            description: f.description,
            quota_bytes: f.quota.as_ref().and_then(|q| q.size_limit),
            used_bytes: f.space_usage,
            file_count: f.quota.and_then(|q| q.file_count_limit),
            dir_count: None,
            created_at: f.create_time,
            updated_at: f.update_time,
            status: f.status.unwrap_or_else(|| "Unknown".to_string()),
        }
    }
}

#[async_trait]
impl CpfsBackend for AliyunCpfsBackend {
    async fn list(&self, page: Option<i32>, page_size: Option<i32>) -> Result<ListResponse> {
        let mut p = BTreeMap::new();
        p.insert("PageNumber".into(), page.unwrap_or(1).to_string());
        p.insert("PageSize".into(), page_size.unwrap_or(10).to_string());
        let r = self.client.call("DescribeFileSystems", API_VERSION, p.clone()).await?;
        let d: ListResp = parse_response("DescribeFileSystems", &p, &r)?;
        Ok(ListResponse {
            items: d.file_systems.file_system.into_iter().map(Into::into).collect(),
            total: d.total_count,
            page: d.page_number,
            page_size: d.page_size,
        })
    }

    async fn get(&self, id: &str) -> Result<FileSystem> {
        let mut p = BTreeMap::new();
        p.insert("FileSystemId".into(), id.into());
        let r = self.client.call("DescribeFileSystems", API_VERSION, p.clone()).await?;
        let d: ListResp = parse_response("DescribeFileSystems", &p, &r)?;
        d.file_systems
            .file_system
            .into_iter()
            .next()
            .map(Into::into)
            .ok_or_else(|| anyhow::anyhow!("FileSystem not found: {}", id))
    }

    async fn create(&self, req: CreateRequest) -> Result<FileSystem> {
        let mut p = BTreeMap::new();
        p.insert("ProtocolType".into(), req.protocol_type);
        p.insert("StorageType".into(), req.storage_type);
        p.insert("Capacity".into(), req.capacity.to_string());
        if let Some(z) = req.zone {
            p.insert("ZoneId".into(), z);
        }
        if let Some(d) = req.description {
            p.insert("Description".into(), d);
        }
        let r = self.client.call("CreateFileSystem", API_VERSION, p.clone()).await?;
        let d: CreateResp = parse_response("CreateFileSystem", &p, &r)?;
        self.get(&d.file_system_id).await
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let mut p = BTreeMap::new();
        p.insert("FileSystemId".into(), id.into());
        self.client.call("DeleteFileSystem", API_VERSION, p).await?;
        Ok(())
    }

    async fn list_filesets(
        &self,
        fs_id: &str,
        page: Option<i32>,
        page_size: Option<i32>,
    ) -> Result<FilesetListResponse> {
        let mut p = BTreeMap::new();
        p.insert("FileSystemId".into(), fs_id.into());
        p.insert("MaxResults".into(), page_size.unwrap_or(20).to_string());
        let r = self.client.call("DescribeFilesets", API_VERSION, p.clone()).await?;
        let d: FilesetListResp = parse_response("DescribeFilesets", &p, &r)?;
        
        // Extract filesets from nested structure: Entries.Entrie[]
        let items = d
            .entries
            .unwrap_or_default()
            .entrie
            .into_iter()
            .map(Into::into)
            .collect();
        
        Ok(FilesetListResponse {
            items,
            total: d.total_count,
            page: page.unwrap_or(1),
            page_size: page_size.unwrap_or(20),
        })
    }

    async fn delete_fileset(&self, fs_id: &str, fileset_id: &str) -> Result<()> {
        let mut p = BTreeMap::new();
        p.insert("FileSystemId".into(), fs_id.into());
        p.insert("FsetId".into(), fileset_id.into());
        // Note: DryRun=false means actually delete
        p.insert("DryRun".into(), "false".into());
        
        let r = self.client.call("DeleteFileset", API_VERSION, p.clone()).await?;
        
        // Check for error response
        if r.contains("\"Code\"") && r.contains("\"Message\"") {
            write_debug_log("DeleteFileset", &p, &r);
            // Try to extract error message
            if let Ok(err) = serde_json::from_str::<serde_json::Value>(&r) {
                let code = err.get("Code").and_then(|v| v.as_str()).unwrap_or("Unknown");
                let msg = err.get("Message").and_then(|v| v.as_str()).unwrap_or(&r);
                return Err(anyhow::anyhow!("{}: {}", code, msg));
            }
        }
        
        Ok(())
    }
}
