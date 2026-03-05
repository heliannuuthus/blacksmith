use crate::config::Config;
use crate::hosts::HostsFile;
use crate::profile::Profile;
use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;
use ratatui::Terminal;
use std::io;

pub struct Tui {
    config: Config,
    state: TuiState,
}

#[derive(Default)]
struct TuiState {
    profiles: Vec<String>,
    selected_profile: usize,
    mode: TuiMode,
    input_buffer: String,
    message: Option<String>,
}

enum TuiMode {
    Main,
    CreateProfile,
    DeleteProfile,
    EnableProfile,
}

impl Default for TuiMode {
    fn default() -> Self {
        TuiMode::Main
    }
}

impl Tui {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            state: TuiState::default(),
        }
    }

    pub async fn run(&mut self) -> Result<()> {
        // 初始化终端
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;

        let mut terminal = Terminal::new(ratatui::backend::CrosstermBackend::new(stdout))?;

        // 加载配置文件列表
        self.load_profiles()?;

        // 主循环
        loop {
            terminal.draw(|f| self.ui(f))?;

            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match self.state.mode {
                        TuiMode::Main => {
                            if self.handle_main_input(key.code)? {
                                break;
                            }
                        }
                        TuiMode::CreateProfile => {
                            if self.handle_create_input(key.code)? {
                                break;
                            }
                        }
                        TuiMode::DeleteProfile => {
                            if self.handle_delete_input(key.code)? {
                                break;
                            }
                        }
                        TuiMode::EnableProfile => {
                            if self.handle_enable_input(key.code)? {
                                break;
                            }
                        }
                    }
                }
            }
        }

        // 清理终端
        disable_raw_mode()?;
        execute!(io::stdout(), LeaveAlternateScreen)?;

        Ok(())
    }

    fn ui(&self, f: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(0),
                Constraint::Length(3),
            ])
            .split(f.area());

        // 标题
        let title = Paragraph::new("hostctl - Hosts 文件管理工具")
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL));
        f.render_widget(title, chunks[0]);

        // 主内容区域
        match self.state.mode {
            TuiMode::Main => self.render_main(f, chunks[1]),
            TuiMode::CreateProfile => self.render_create(f, chunks[1]),
            TuiMode::DeleteProfile => self.render_delete(f, chunks[1]),
            TuiMode::EnableProfile => self.render_enable(f, chunks[1]),
        }

        // 底部帮助栏
        let help = self.get_help_text();
        let help_paragraph = Paragraph::new(help)
            .style(Style::default().fg(Color::Gray))
            .block(Block::default().borders(Borders::ALL));
        f.render_widget(help_paragraph, chunks[2]);
    }

    fn render_main(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // 配置文件列表
        let items: Vec<ListItem> = self
            .state
            .profiles
            .iter()
            .enumerate()
            .map(|(i, profile)| {
                let style = if i == self.state.selected_profile {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD | Modifier::REVERSED)
                } else {
                    Style::default().fg(Color::White)
                };
                ListItem::new(format!("  • {}", profile)).style(style)
            })
            .collect();

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("配置文件列表")
                    .title_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            )
            .highlight_style(Style::default().add_modifier(Modifier::BOLD));

        let mut list_state = ListState::default();
        list_state.select(Some(self.state.selected_profile));
        f.render_stateful_widget(list, chunks[0], &mut list_state);

        // 右侧信息面板
        let info = if !self.state.profiles.is_empty() {
            let selected = &self.state.profiles[self.state.selected_profile];
            let profile_path = self.config.profiles_dir.join(format!("{}.toml", selected));
            if let Ok(profile) = Profile::load(&profile_path) {
                format!(
                    "名称: {}\n描述: {}\n条目数: {}",
                    profile.name,
                    profile.description.as_deref().unwrap_or("无"),
                    profile.entries.len()
                )
            } else {
                format!("无法加载配置: {}", selected)
            }
        } else {
            "没有配置文件\n按 'n' 创建新配置".to_string()
        };

        let info_paragraph = Paragraph::new(info)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("配置信息")
                    .title_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            )
            .wrap(Wrap { trim: true });
        f.render_widget(info_paragraph, chunks[1]);

        // 显示消息
        if let Some(ref msg) = self.state.message {
            let msg_paragraph = Paragraph::new(msg.clone())
                .style(Style::default().fg(Color::Green))
                .block(Block::default().borders(Borders::ALL));
            let msg_area = Rect {
                x: area.x + area.width / 4,
                y: area.y + area.height / 2,
                width: area.width / 2,
                height: 3,
            };
            f.render_widget(msg_paragraph, msg_area);
        }
    }

    fn render_create(&self, f: &mut Frame, area: Rect) {
        let text = format!("输入配置文件名: {}", self.state.input_buffer);
        let input = Paragraph::new(text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("创建配置")
                    .title_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            );
        f.render_widget(input, area);
    }

    fn render_delete(&self, f: &mut Frame, area: Rect) {
        let selected = if !self.state.profiles.is_empty() {
            &self.state.profiles[self.state.selected_profile]
        } else {
            ""
        };
        let text = format!("确认删除配置 '{}'? (y/n)", selected);
        let confirm = Paragraph::new(text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("删除配置")
                    .title_style(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            );
        f.render_widget(confirm, area);
    }

    fn render_enable(&self, f: &mut Frame, area: Rect) {
        let selected = if !self.state.profiles.is_empty() {
            &self.state.profiles[self.state.selected_profile]
        } else {
            ""
        };
        let text = format!("确认启用配置 '{}'? (y/n)", selected);
        let confirm = Paragraph::new(text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("启用配置")
                    .title_style(Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            );
        f.render_widget(confirm, area);
    }

    fn handle_main_input(&mut self, key: KeyCode) -> Result<bool> {
        match key {
            KeyCode::Char('q') => return Ok(true),
            KeyCode::Up => {
                if !self.state.profiles.is_empty() {
                    self.state.selected_profile = self
                        .state
                        .selected_profile
                        .saturating_sub(1)
                        .min(self.state.profiles.len().saturating_sub(1));
                }
            }
            KeyCode::Down => {
                if !self.state.profiles.is_empty() {
                    self.state.selected_profile = (self.state.selected_profile + 1)
                        .min(self.state.profiles.len().saturating_sub(1));
                }
            }
            KeyCode::Char('n') => {
                self.state.mode = TuiMode::CreateProfile;
                self.state.input_buffer.clear();
            }
            KeyCode::Char('d') => {
                if !self.state.profiles.is_empty() {
                    self.state.mode = TuiMode::DeleteProfile;
                }
            }
            KeyCode::Char('e') => {
                if !self.state.profiles.is_empty() {
                    self.state.mode = TuiMode::EnableProfile;
                }
            }
            KeyCode::Enter => {
                if !self.state.profiles.is_empty() {
                    self.state.mode = TuiMode::EnableProfile;
                }
            }
            _ => {}
        }
        Ok(false)
    }

    fn handle_create_input(&mut self, key: KeyCode) -> Result<bool> {
        match key {
            KeyCode::Esc => {
                self.state.mode = TuiMode::Main;
                self.state.input_buffer.clear();
            }
            KeyCode::Enter => {
                if !self.state.input_buffer.is_empty() {
                    let name = self.state.input_buffer.clone();
                    let profile = Profile::new(name.clone());
                    let profile_path = self.config.profiles_dir.join(format!("{}.toml", name));
                    profile.save(&profile_path)?;
                    self.load_profiles()?;
                    self.state.mode = TuiMode::Main;
                    self.state.input_buffer.clear();
                    self.state.message = Some(format!("配置 '{}' 已创建", name));
                }
            }
            KeyCode::Backspace => {
                self.state.input_buffer.pop();
            }
            KeyCode::Char(c) => {
                self.state.input_buffer.push(c);
            }
            _ => {}
        }
        Ok(false)
    }

    fn handle_delete_input(&mut self, key: KeyCode) -> Result<bool> {
        match key {
            KeyCode::Esc => {
                self.state.mode = TuiMode::Main;
            }
            KeyCode::Char('y') => {
                if !self.state.profiles.is_empty() {
                    let name = self.state.profiles[self.state.selected_profile].clone();
                    let profile_path = self.config.profiles_dir.join(format!("{}.toml", name));
                    std::fs::remove_file(&profile_path)?;
                    self.load_profiles()?;
                    if self.state.selected_profile >= self.state.profiles.len() {
                        self.state.selected_profile = self.state.profiles.len().saturating_sub(1);
                    }
                    self.state.mode = TuiMode::Main;
                    self.state.message = Some(format!("配置 '{}' 已删除", name));
                }
            }
            KeyCode::Char('n') => {
                self.state.mode = TuiMode::Main;
            }
            _ => {}
        }
        Ok(false)
    }

    fn handle_enable_input(&mut self, key: KeyCode) -> Result<bool> {
        match key {
            KeyCode::Esc => {
                self.state.mode = TuiMode::Main;
            }
            KeyCode::Char('y') => {
                if !self.state.profiles.is_empty() {
                    let name = self.state.profiles[self.state.selected_profile].clone();
                    let profile_path = self.config.profiles_dir.join(format!("{}.toml", name));
                    let profile = Profile::load(&profile_path)?;
                    let hosts_file = HostsFile::new(self.config.hosts_file.clone());
                    
                    // 备份
                    let backup_name = format!(
                        "backup_{}.txt",
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    );
                    let backup_path = self.config.backup_dir.join(&backup_name);
                    hosts_file.backup(&backup_path)?;
                    
                    // 启用（使用 tokio runtime 执行异步操作）
                    let content = profile.to_hosts_content();
                    let hosts_file_clone = hosts_file.clone();
                    let content_clone = content.clone();
                    
                    // 在同步上下文中执行异步操作
                    match tokio::runtime::Handle::try_current() {
                        Ok(handle) => {
                            handle.block_on(hosts_file_clone.write(&content_clone))?;
                        }
                        Err(_) => {
                            // 如果没有运行时，创建一个临时的
                            let rt = tokio::runtime::Runtime::new()
                                .context("无法创建 tokio runtime")?;
                            rt.block_on(hosts_file_clone.write(&content_clone))?;
                        }
                    }
                    
                    self.state.mode = TuiMode::Main;
                    self.state.message = Some(format!("配置 '{}' 已启用", name));
                }
            }
            KeyCode::Char('n') => {
                self.state.mode = TuiMode::Main;
            }
            _ => {}
        }
        Ok(false)
    }

    fn load_profiles(&mut self) -> Result<()> {
        self.state.profiles = self.get_profiles()?;
        if self.state.selected_profile >= self.state.profiles.len() {
            self.state.selected_profile = 0;
        }
        Ok(())
    }

    fn get_profiles(&self) -> Result<Vec<String>> {
        let mut profiles = Vec::new();
        if !self.config.profiles_dir.exists() {
            return Ok(profiles);
        }

        for entry in std::fs::read_dir(&self.config.profiles_dir)? {
            let entry = entry?;
            if entry.path().is_file()
                && entry.path().extension().and_then(|s| s.to_str()) == Some("toml")
            {
                if let Some(name) = entry.path().file_stem().and_then(|s| s.to_str()) {
                    profiles.push(name.to_string());
                }
            }
        }

        profiles.sort();
        Ok(profiles)
    }

    fn get_help_text(&self) -> String {
        match self.state.mode {
            TuiMode::Main => "↑↓: 选择 | n: 新建 | d: 删除 | e/Enter: 启用 | q: 退出".to_string(),
            TuiMode::CreateProfile => "输入名称后按 Enter 确认，Esc 取消".to_string(),
            TuiMode::DeleteProfile => "y: 确认删除 | n/Esc: 取消".to_string(),
            TuiMode::EnableProfile => "y: 确认启用 | n/Esc: 取消".to_string(),
        }
    }
}
