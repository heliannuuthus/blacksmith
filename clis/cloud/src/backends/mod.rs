//! Backend implementations for cloud providers.

pub mod aliyun;

use anyhow::Result;

use crate::config::{Provider, ProviderType};
use crate::services::CpfsBackend;

/// Create a CPFS backend for the given provider.
///
/// # Errors
/// Returns error if the provider type is not supported or configuration is invalid.
pub fn create_cpfs_backend(
    provider: &Provider,
    region_override: Option<&str>,
) -> Result<Box<dyn CpfsBackend>> {
    match provider.provider_type {
        ProviderType::Aliyun => {
            let config = aliyun::AliyunConfig::from_provider(provider, region_override)?;
            Ok(Box::new(aliyun::AliyunCpfsBackend::new(config)))
        }
        ProviderType::Aws => {
            anyhow::bail!("AWS CPFS is not supported yet")
        }
        ProviderType::Tencent => {
            anyhow::bail!("Tencent Cloud CPFS is not supported yet")
        }
        ProviderType::Gcp => {
            anyhow::bail!("GCP does not support CPFS")
        }
        ProviderType::Azure => {
            anyhow::bail!("Azure does not support CPFS")
        }
    }
}
