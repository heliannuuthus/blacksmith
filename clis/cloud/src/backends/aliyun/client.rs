//! Aliyun OpenAPI client with signature implementation.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use base64::Engine;
use chrono::Utc;
use hmac::{Hmac, Mac};
use sha2::Sha256;

/// Aliyun OpenAPI client.
#[derive(Clone)]
pub struct AliyunClient {
    access_key_id: String,
    access_key_secret: String,
    endpoint: String,
    http_client: reqwest::Client,
}

impl AliyunClient {
    /// Create a new Aliyun client.
    #[must_use]
    pub fn new(access_key_id: String, access_key_secret: String, endpoint: String) -> Self {
        Self {
            access_key_id,
            access_key_secret,
            endpoint,
            http_client: reqwest::Client::new(),
        }
    }

    /// Call an API action.
    pub async fn call(
        &self,
        action: &str,
        version: &str,
        params: BTreeMap<String, String>,
    ) -> Result<String> {
        let mut all_params = self.build_common_params(action, version);
        for (k, v) in params {
            all_params.insert(k, v);
        }
        let signature = self.sign(&all_params)?;
        all_params.insert("Signature".to_string(), signature);

        let query_string: String = all_params
            .iter()
            .map(|(k, v)| format!("{}={}", url_encode(k), url_encode(v)))
            .collect::<Vec<_>>()
            .join("&");

        let url = format!("https://{}/?{}", self.endpoint, query_string);

        let response = self.http_client.get(&url).send().await
            .context("failed to send request")?;

        let status = response.status();
        let body = response.text().await.context("failed to read body")?;

        if !status.is_success() {
            if let Ok(e) = serde_json::from_str::<AliyunErrorResponse>(&body) {
                anyhow::bail!("Aliyun API: {} - {} ({})", e.code, e.message, e.request_id);
            }
            anyhow::bail!("Aliyun API: {} - {}", status, body);
        }
        Ok(body)
    }

    fn build_common_params(&self, action: &str, version: &str) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("Action".into(), action.into());
        p.insert("Version".into(), version.into());
        p.insert("Format".into(), "JSON".into());
        p.insert("AccessKeyId".into(), self.access_key_id.clone());
        p.insert("SignatureMethod".into(), "HMAC-SHA256".into());
        p.insert("SignatureVersion".into(), "1.0".into());
        p.insert("SignatureNonce".into(), uuid::Uuid::new_v4().to_string());
        p.insert("Timestamp".into(), Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string());
        p
    }

    fn sign(&self, params: &BTreeMap<String, String>) -> Result<String> {
        let cq: String = params.iter()
            .map(|(k, v)| format!("{}={}", url_encode(k), url_encode(v)))
            .collect::<Vec<_>>().join("&");
        let sts = format!("GET&{}&{}", url_encode("/"), url_encode(&cq));
        let key = format!("{}&", self.access_key_secret);
        let mut mac = Hmac::<Sha256>::new_from_slice(key.as_bytes()).context("HMAC")?;
        mac.update(sts.as_bytes());
        Ok(base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes()))
    }
}

fn url_encode(s: &str) -> String {
    urlencoding::encode(s).replace('+', "%20").replace('*', "%2A").replace("%7E", "~")
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
struct AliyunErrorResponse {
    request_id: String,
    code: String,
    message: String,
}

impl std::fmt::Debug for AliyunClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AliyunClient").field("endpoint", &self.endpoint).finish()
    }
}
