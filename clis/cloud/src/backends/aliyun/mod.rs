//! Aliyun (Alibaba Cloud) backend implementation.

pub mod client;
pub mod cpfs;

pub use cpfs::AliyunCpfsBackend;

use crate::config::{Auth, Provider};
use anyhow::Result;

/// Aliyun configuration extracted from Provider.
#[derive(Debug, Clone)]
pub struct AliyunConfig {
    pub access_key_id: String,
    pub access_key_secret: String,
    pub region: String,
}

impl AliyunConfig {
    pub fn from_provider(provider: &Provider, region_override: Option<&str>) -> Result<Self> {
        let (access_key_id, access_key_secret) = match &provider.auth {
            Auth::AccessKey { access_key_id, access_key_secret } => 
                (access_key_id.clone(), access_key_secret.clone()),
            _ => anyhow::bail!("Aliyun requires AccessKey authentication"),
        };
        let region = region_override
            .map(ToOwned::to_owned)
            .or_else(|| provider.region.clone())
            .unwrap_or_else(|| "cn-hangzhou".to_owned());
        Ok(Self { access_key_id, access_key_secret, region })
    }
}
