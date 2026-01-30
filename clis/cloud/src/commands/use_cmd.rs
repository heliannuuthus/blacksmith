//! Use command for switching providers.

use anyhow::Result;

use crate::config::ConfigManager;

/// Execute use command (set current provider).
pub fn execute(manager: &ConfigManager, name: Option<String>) -> Result<()> {
    if let Some(name) = name {
        manager.set_current(&name)?;
        println!("Switched to provider: {}", name);
    } else {
        if let Some(current) = manager.current()? {
            println!("Current provider: {}", current);
        } else {
            println!("No provider selected");
            println!("\nUse: cloud use <provider>");
        }
    }
    Ok(())
}
