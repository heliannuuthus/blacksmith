use clap::Parser;
use hostctl::app::App;
use hostctl::cli::Cli;
use hostctl::config::Config;
use std::process;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // 初始化配置
    let config = match Config::load() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("错误: 无法加载配置: {}", e);
            process::exit(1);
        }
    };

    // 运行应用
    if let Err(e) = App::new(config).run(cli).await {
        eprintln!("错误: {}", e);
        process::exit(1);
    }
}
