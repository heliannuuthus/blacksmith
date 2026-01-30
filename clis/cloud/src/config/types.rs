//! Configuration types for cloud CLI.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Supported cloud provider types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    /// Alibaba Cloud
    Aliyun,
    /// Amazon Web Services
    Aws,
    /// Tencent Cloud
    Tencent,
    /// Google Cloud Platform
    Gcp,
    /// Microsoft Azure
    Azure,
}

impl fmt::Display for ProviderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Aliyun => write!(f, "aliyun"),
            Self::Aws => write!(f, "aws"),
            Self::Tencent => write!(f, "tencent"),
            Self::Gcp => write!(f, "gcp"),
            Self::Azure => write!(f, "azure"),
        }
    }
}

impl ProviderType {
    /// Get all available provider types.
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::Aliyun,
            Self::Aws,
            Self::Tencent,
            Self::Gcp,
            Self::Azure,
        ]
    }

    /// Get display names for all provider types.
    #[must_use]
    pub fn display_names() -> Vec<&'static str> {
        vec!["Aliyun", "AWS", "Tencent", "GCP", "Azure"]
    }
}

/// Authentication methods for different providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Auth {
    /// Access Key / Secret Key authentication (Aliyun, AWS, etc.)
    AccessKey {
        /// Access Key ID
        access_key_id: String,
        /// Access Key Secret
        access_key_secret: String,
    },
    /// Token-based authentication
    Token {
        /// Token string
        token: String,
    },
    /// OAuth2 authentication (Google Cloud, Azure, etc.)
    OAuth2 {
        /// Client ID
        client_id: String,
        /// Client Secret
        client_secret: String,
        /// Refresh Token
        refresh_token: String,
    },
}

impl Auth {
    /// Get the auth type name.
    #[must_use]
    #[allow(dead_code)] // May be used in future commands
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::AccessKey { .. } => "access_key",
            Self::Token { .. } => "token",
            Self::OAuth2 { .. } => "oauth2",
        }
    }

    /// Get all available auth type names.
    #[must_use]
    pub fn type_names() -> &'static [&'static str] {
        &["AccessKey", "Token", "OAuth2"]
    }
}

/// Provider configuration (a named configuration instance).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provider {
    /// Cloud provider type
    #[serde(rename = "type")]
    pub provider_type: ProviderType,
    /// Default region
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// Authentication information
    pub auth: Auth,
}

impl Provider {
    /// Create a new provider configuration.
    #[must_use]
    pub fn new(provider_type: ProviderType, auth: Auth, region: Option<String>) -> Self {
        Self {
            provider_type,
            region,
            auth,
        }
    }
}
