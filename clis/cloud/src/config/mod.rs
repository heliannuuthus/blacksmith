//! Configuration module for cloud CLI.

pub mod manager;
pub mod types;

pub use manager::{Config, ConfigManager};
pub use types::{Auth, Provider, ProviderType};
