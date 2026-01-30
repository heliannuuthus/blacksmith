//! Interactive list command with TUI interface (k9s-style).

use std::io::{self, stdout};
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState},
    Frame, Terminal,
};

use crate::backends::create_cpfs_backend;
use crate::config::{ConfigManager, Provider};
use crate::resources::{create_resource, parse_resource_kind, ListOptions, Resource, ResourceItem};
use crate::services::cpfs::{format_bytes, Fileset};

// ============================================================================
// k9s-style Theme Colors
// ============================================================================

mod theme {
    use ratatui::style::Color;

    // Main colors (k9s inspired)
    pub const FG_PRIMARY: Color = Color::Rgb(198, 208, 245);      // Light gray text
    pub const FG_SECONDARY: Color = Color::Rgb(140, 155, 186);    // Dimmed text
    pub const FG_MUTED: Color = Color::Rgb(98, 114, 164);         // Very dim text

    // Accent colors
    pub const ACCENT_GREEN: Color = Color::Rgb(166, 218, 149);    // k9s green
    pub const ACCENT_BLUE: Color = Color::Rgb(138, 173, 244);     // k9s blue
    pub const ACCENT_YELLOW: Color = Color::Rgb(238, 212, 159);   // k9s yellow
    pub const ACCENT_RED: Color = Color::Rgb(237, 135, 150);      // k9s red
    pub const ACCENT_MAGENTA: Color = Color::Rgb(198, 160, 246);  // k9s magenta
    pub const ACCENT_CYAN: Color = Color::Rgb(139, 213, 202);     // k9s cyan
    pub const ACCENT_ORANGE: Color = Color::Rgb(245, 169, 127);   // k9s orange

    // Border colors
    pub const BORDER_NORMAL: Color = Color::Rgb(69, 71, 90);      // Normal border
    pub const BORDER_FOCUS: Color = Color::Rgb(137, 180, 250);    // Focused border (blue)

    // Status colors
    pub const STATUS_RUNNING: Color = ACCENT_GREEN;
    pub const STATUS_STOPPED: Color = ACCENT_YELLOW;
    pub const STATUS_ERROR: Color = ACCENT_RED;
    pub const STATUS_PENDING: Color = ACCENT_BLUE;
}

/// View mode for the TUI.
#[derive(Debug, Clone, PartialEq)]
enum ViewMode {
    /// Main resource list (e.g., CPFS list).
    ResourceList,
    /// Fileset list for a specific CPFS.
    FilesetList { cpfs_id: String, cpfs_name: String },
}

/// Dialog mode for confirmations.
#[derive(Debug, Clone, PartialEq)]
enum DialogMode {
    /// No dialog shown.
    None,
    /// Delete confirmation dialog for a fileset.
    DeleteFilesetConfirm {
        /// File system ID.
        fs_id: String,
        /// Fileset ID.
        fileset_id: String,
        /// Fileset path (for display).
        fileset_path: String,
        /// Selected button: false = cancel (default), true = confirm.
        confirm_selected: bool,
    },
    /// Deleting in progress.
    Deleting,
}

/// Application state for the TUI.
struct App {
    /// Current view mode.
    view_mode: ViewMode,
    /// Current dialog mode.
    dialog_mode: DialogMode,
    /// Current table state (selection).
    table_state: TableState,
    /// Current page number.
    current_page: i32,
    /// Total pages.
    total_pages: i32,
    /// Total items.
    total_items: i32,
    /// Items on current page.
    items: Vec<ResourceItem>,
    /// Fileset items (when in fileset view).
    filesets: Vec<Fileset>,
    /// Resource kind name.
    kind: String,
    /// Provider name.
    provider: String,
    /// Page size.
    page_size: i32,
    /// Column definitions.
    columns: Vec<&'static str>,
    /// Is loading.
    loading: bool,
    /// Error message if any.
    error: Option<String>,
    /// Success message if any.
    success: Option<String>,
    /// Should quit.
    should_quit: bool,
    /// Should refresh.
    should_refresh: bool,
}

impl App {
    fn new(kind: String, provider: String, columns: Vec<&'static str>, page_size: i32) -> Self {
        let mut table_state = TableState::default();
        table_state.select(Some(0));

        Self {
            view_mode: ViewMode::ResourceList,
            dialog_mode: DialogMode::None,
            table_state,
            current_page: 1,
            total_pages: 1,
            total_items: 0,
            items: Vec::new(),
            filesets: Vec::new(),
            kind,
            provider,
            page_size,
            columns,
            loading: true,
            error: None,
            success: None,
            should_quit: false,
            should_refresh: true,
        }
    }

    fn show_delete_fileset_confirm(&mut self, fs_id: String, fileset_id: String, fileset_path: String) {
        self.dialog_mode = DialogMode::DeleteFilesetConfirm {
            fs_id,
            fileset_id,
            fileset_path,
            confirm_selected: false, // 默认选中取消
        };
    }

    fn selected_fileset(&self) -> Option<&Fileset> {
        self.selected_index().and_then(|i| self.filesets.get(i))
    }

    fn cancel_dialog(&mut self) {
        self.dialog_mode = DialogMode::None;
    }

    fn is_dialog_open(&self) -> bool {
        !matches!(self.dialog_mode, DialogMode::None)
    }

    fn selected_index(&self) -> Option<usize> {
        self.table_state.selected()
    }

    fn selected_item(&self) -> Option<&ResourceItem> {
        self.selected_index().and_then(|i| self.items.get(i))
    }

    fn item_count(&self) -> usize {
        match &self.view_mode {
            ViewMode::ResourceList => self.items.len(),
            ViewMode::FilesetList { .. } => self.filesets.len(),
        }
    }

    fn next(&mut self) {
        let count = self.item_count();
        if count == 0 {
            return;
        }
        let i = match self.table_state.selected() {
            Some(i) => {
                if i >= count - 1 {
                    0
                } else {
                    i + 1
                }
            }
            None => 0,
        };
        self.table_state.select(Some(i));
    }

    fn previous(&mut self) {
        let count = self.item_count();
        if count == 0 {
            return;
        }
        let i = match self.table_state.selected() {
            Some(i) => {
                if i == 0 {
                    count - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        self.table_state.select(Some(i));
    }

    fn next_page(&mut self) {
        if self.current_page < self.total_pages {
            self.current_page += 1;
            self.should_refresh = true;
            self.table_state.select(Some(0));
        }
    }

    fn prev_page(&mut self) {
        if self.current_page > 1 {
            self.current_page -= 1;
            self.should_refresh = true;
            self.table_state.select(Some(0));
        }
    }

    fn refresh(&mut self) {
        self.should_refresh = true;
    }

    fn enter_fileset_view(&mut self, cpfs_id: String, cpfs_name: String) {
        self.view_mode = ViewMode::FilesetList { cpfs_id, cpfs_name };
        self.current_page = 1;
        self.table_state.select(Some(0));
        self.filesets.clear();
        self.should_refresh = true;
    }

    fn back_to_resource_list(&mut self) {
        if matches!(self.view_mode, ViewMode::FilesetList { .. }) {
            self.view_mode = ViewMode::ResourceList;
            self.table_state.select(Some(0));
            self.filesets.clear();
            // Don't refresh, we still have the data
        }
    }
}

/// Execute interactive list command with TUI.
pub async fn execute(
    manager: &ConfigManager,
    resource_type: &str,
    provider: Option<&str>,
    region: Option<&str>,
    page_size: i32,
) -> Result<()> {
    let kind = parse_resource_kind(resource_type)?;
    let (provider_name, provider_config) = manager.resolve_provider(provider)?;
    let resource = create_resource(kind, &provider_config, region)?;

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let columns = resource.columns();
    let mut app = App::new(
        resource.kind().to_string(),
        provider_name.clone(),
        columns,
        page_size,
    );

    // Main loop
    let result = run_app(&mut terminal, &mut app, &*resource, &provider_config, region).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    resource: &dyn Resource,
    provider: &Provider,
    region: Option<&str>,
) -> Result<()> {
    // Create backend for fileset queries
    let cpfs_backend = create_cpfs_backend(provider, region).ok();

    loop {
        // Fetch data if needed
        if app.should_refresh {
            app.loading = true;
            app.error = None;

            match &app.view_mode {
                ViewMode::ResourceList => {
                    let opts = ListOptions {
                        page: Some(app.current_page),
                        page_size: Some(app.page_size),
                        ..Default::default()
                    };

                    match resource.list(&opts).await {
                        Ok(result) => {
                            app.items = result.items;
                            app.total_items = result.total;
                            app.total_pages = if result.total > 0 {
                                (result.total + app.page_size - 1) / app.page_size
                            } else {
                                1
                            };
                            if app.table_state.selected().is_none() && !app.items.is_empty() {
                                app.table_state.select(Some(0));
                            }
                        }
                        Err(e) => {
                            app.error = Some(e.to_string());
                        }
                    }
                }
                ViewMode::FilesetList { cpfs_id, .. } => {
                    if let Some(ref backend) = cpfs_backend {
                        match backend
                            .list_filesets(cpfs_id, Some(app.current_page), Some(app.page_size))
                            .await
                        {
                            Ok(result) => {
                                app.filesets = result.items;
                                app.total_items = result.total;
                                app.total_pages = if result.total > 0 {
                                    (result.total + app.page_size - 1) / app.page_size
                                } else {
                                    1
                                };
                                if app.table_state.selected().is_none() && !app.filesets.is_empty()
                                {
                                    app.table_state.select(Some(0));
                                }
                            }
                            Err(e) => {
                                let err_msg = e.to_string();
                                // Provide more helpful error messages for common issues
                                if err_msg.contains("GrayCondition.NotMet") {
                                    app.error = Some("Fileset API requires whitelist access. Contact Aliyun support to enable.".to_string());
                                } else if err_msg.contains("InvalidFilesystemType.NotSupport") {
                                    app.error = Some("This filesystem type does not support Fileset.".to_string());
                                } else {
                                    app.error = Some(err_msg);
                                }
                            }
                        }
                    } else {
                        app.error = Some("Backend not available".to_string());
                    }
                }
            }

            app.loading = false;
            app.should_refresh = false;
        }

        // Draw UI
        terminal.draw(|f| ui(f, app, resource))?;

        // Handle input with timeout for async refresh
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    // Clear previous messages on any key press
                    app.error = None;
                    app.success = None;

                    // Handle dialog mode first
                    if app.is_dialog_open() {
                        match &mut app.dialog_mode {
                            DialogMode::DeleteFilesetConfirm { fs_id, fileset_id, fileset_path, confirm_selected } => {
                                match key.code {
                                    KeyCode::Left | KeyCode::Right | KeyCode::Tab => {
                                        // 左右切换选择
                                        *confirm_selected = !*confirm_selected;
                                    }
                                    KeyCode::Enter => {
                                        if *confirm_selected {
                                            // 执行删除
                                            let fs = fs_id.clone();
                                            let fset = fileset_id.clone();
                                            let path = fileset_path.clone();
                                            app.dialog_mode = DialogMode::Deleting;
                                            
                                            // 绘制删除中状态
                                            terminal.draw(|f| ui(f, app, resource))?;
                                            
                                            // 执行删除 fileset
                                            if let Some(ref backend) = cpfs_backend {
                                                match backend.delete_fileset(&fs, &fset).await {
                                                    Ok(()) => {
                                                        app.success = Some(format!("已成功删除 {}", path));
                                                        app.dialog_mode = DialogMode::None;
                                                        app.should_refresh = true;
                                                    }
                                                    Err(e) => {
                                                        let err_msg = e.to_string();
                                                        if err_msg.contains("DeletionProtection") {
                                                            app.error = Some("删除失败: Fileset 已开启删除保护".to_string());
                                                        } else if err_msg.contains("FilesetNotEmpty") || err_msg.contains("NotEmpty") {
                                                            app.error = Some("删除失败: Fileset 不为空".to_string());
                                                        } else {
                                                            app.error = Some(format!("删除失败: {}", err_msg));
                                                        }
                                                        app.dialog_mode = DialogMode::None;
                                                    }
                                                }
                                            } else {
                                                app.error = Some("后端不可用".to_string());
                                                app.dialog_mode = DialogMode::None;
                                            }
                                        } else {
                                            // 取消
                                            app.cancel_dialog();
                                        }
                                    }
                                    KeyCode::Esc => {
                                        // ESC 直接取消
                                        app.cancel_dialog();
                                    }
                                    _ => {}
                                }
                            }
                            DialogMode::Deleting => {
                                // 删除中，忽略输入
                            }
                            DialogMode::None => {}
                        }
                        continue;
                    }

                    // Normal mode key handling
                    match key.code {
                        KeyCode::Char('q') => {
                            app.should_quit = true;
                        }
                        KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            // Ctrl+D to delete fileset (only in fileset view)
                            if let ViewMode::FilesetList { cpfs_id, .. } = &app.view_mode {
                                if let Some(fileset) = app.selected_fileset().cloned() {
                                    app.show_delete_fileset_confirm(
                                        cpfs_id.clone(),
                                        fileset.id.clone(),
                                        fileset.path.clone(),
                                    );
                                }
                            }
                        }
                        KeyCode::Esc => {
                            // ESC goes back or quits
                            if matches!(app.view_mode, ViewMode::FilesetList { .. }) {
                                app.back_to_resource_list();
                            } else {
                                app.should_quit = true;
                            }
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            app.next();
                        }
                        KeyCode::Up | KeyCode::Char('k') => {
                            app.previous();
                        }
                        KeyCode::Right | KeyCode::Char('l') | KeyCode::Char('n') => {
                            app.next_page();
                        }
                        KeyCode::Left | KeyCode::Char('h') | KeyCode::Char('p') => {
                            // In fileset view, left goes back
                            if matches!(app.view_mode, ViewMode::FilesetList { .. }) {
                                app.back_to_resource_list();
                            } else {
                                app.prev_page();
                            }
                        }
                        KeyCode::Enter => {
                            // Enter on CPFS goes to fileset view (only for cpfs/bmcpfs types)
                            if matches!(app.view_mode, ViewMode::ResourceList) {
                                if let Some(item) = app.selected_item().cloned() {
                                    if app.kind == "cpfs" {
                                        // Check if this filesystem type supports filesets
                                        // Only cpfs and bmcpfs types support filesets, not standard or extreme NAS
                                        let fs_type = item
                                            .extra
                                            .get("type")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");
                                        
                                        if fs_type.starts_with("cpfs") || fs_type.starts_with("bmcpfs") {
                                            let display_name = item
                                                .extra
                                                .get("display_name")
                                                .and_then(|v| v.as_str())
                                                .map(String::from)
                                                .unwrap_or_else(|| item.name.clone());
                                            app.enter_fileset_view(item.name.clone(), display_name);
                                        } else {
                                            // Show error for unsupported types
                                            app.error = Some(format!(
                                                "Fileset not supported for {} (only cpfs/bmcpfs types)", 
                                                fs_type
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                        KeyCode::Char('r') => {
                            app.refresh();
                        }
                        KeyCode::Home => {
                            app.table_state.select(Some(0));
                        }
                        KeyCode::End => {
                            let count = app.item_count();
                            if count > 0 {
                                app.table_state.select(Some(count - 1));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}

fn ui(f: &mut Frame, app: &mut App, resource: &dyn Resource) {
    let size = f.area();

    // Create layout (no custom background - use terminal default)
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(5),    // Table
            Constraint::Length(3), // Footer/Help
        ])
        .split(size);

    // Header
    render_header(f, app, chunks[0]);

    // Main table
    match &app.view_mode {
        ViewMode::ResourceList => render_resource_table(f, app, resource, chunks[1]),
        ViewMode::FilesetList { .. } => render_fileset_table(f, app, chunks[1]),
    }

    // Footer with help
    render_footer(f, app, chunks[2]);

    // Render dialog overlay if active
    if app.is_dialog_open() {
        render_dialog(f, app, size);
    }
}

fn render_dialog(f: &mut Frame, app: &App, area: Rect) {
    // Calculate dialog size and position (centered)
    let dialog_width = 60.min(area.width.saturating_sub(4));
    let dialog_height = 7;
    let dialog_x = (area.width.saturating_sub(dialog_width)) / 2;
    let dialog_y = (area.height.saturating_sub(dialog_height)) / 2;
    
    let dialog_area = Rect::new(dialog_x, dialog_y, dialog_width, dialog_height);

    // Clear the dialog area with a block
    let clear_block = Block::default()
        .style(Style::default().bg(Color::Rgb(30, 30, 46)));
    f.render_widget(clear_block, dialog_area);

    match &app.dialog_mode {
        DialogMode::DeleteFilesetConfirm { fileset_id, fileset_path, confirm_selected, .. } => {
            let display = format!("{} ({})", fileset_path, fileset_id);

            // 按钮样式：选中时高亮
            let (confirm_style, cancel_style) = if *confirm_selected {
                (
                    Style::default().fg(Color::Black).bg(theme::ACCENT_RED).add_modifier(Modifier::BOLD),
                    Style::default().fg(theme::FG_MUTED),
                )
            } else {
                (
                    Style::default().fg(theme::FG_MUTED),
                    Style::default().fg(Color::Black).bg(theme::ACCENT_GREEN).add_modifier(Modifier::BOLD),
                )
            };

            let text = vec![
                Line::from(vec![
                    Span::styled("⚠ ", Style::default().fg(theme::ACCENT_YELLOW)),
                    Span::styled("删除确认", Style::default().fg(theme::ACCENT_RED).add_modifier(Modifier::BOLD)),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("删除: ", Style::default().fg(theme::FG_SECONDARY)),
                    Span::styled(
                        display.chars().take(dialog_width as usize - 10).collect::<String>(),
                        Style::default().fg(theme::FG_PRIMARY).add_modifier(Modifier::BOLD)
                    ),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("       ", Style::default()),
                    Span::styled(" 确认 ", confirm_style),
                    Span::styled("   ", Style::default()),
                    Span::styled(" 取消 ", cancel_style),
                    Span::styled("       ", Style::default()),
                ]),
            ];

            let dialog = Paragraph::new(text)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(theme::ACCENT_RED))
                        .style(Style::default().bg(Color::Rgb(30, 30, 46))),
                )
                .alignment(ratatui::layout::Alignment::Center);

            f.render_widget(dialog, dialog_area);
        }
        DialogMode::Deleting => {
            let text = vec![
                Line::from(""),
                Line::from(vec![
                    Span::styled("◐ ", Style::default().fg(theme::ACCENT_YELLOW)),
                    Span::styled("删除中...", Style::default().fg(theme::FG_PRIMARY)),
                ]),
                Line::from(""),
            ];

            let dialog = Paragraph::new(text)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(theme::ACCENT_YELLOW))
                        .style(Style::default().bg(Color::Rgb(30, 30, 46))),
                )
                .alignment(ratatui::layout::Alignment::Center);

            f.render_widget(dialog, dialog_area);
        }
        DialogMode::None => {}
    }
}

fn render_header(f: &mut Frame, app: &App, area: Rect) {
    let status = if app.loading {
        Span::styled("◐ 加载中", Style::default().fg(theme::ACCENT_YELLOW))
    } else if app.error.is_some() {
        Span::styled("✗ 错误", Style::default().fg(theme::ACCENT_RED))
    } else if app.success.is_some() {
        Span::styled("✓ 成功", Style::default().fg(theme::ACCENT_GREEN))
    } else {
        Span::styled("● 就绪", Style::default().fg(theme::ACCENT_GREEN))
    };

    let (title_kind, breadcrumb) = match &app.view_mode {
        ViewMode::ResourceList => (app.kind.to_uppercase(), String::new()),
        ViewMode::FilesetList { cpfs_name, .. } => {
            ("FILESET".to_string(), format!(" ❯ {}", cpfs_name))
        }
    };

    let title = Line::from(vec![
        Span::styled(
            format!(" {} ", title_kind),
            Style::default()
                .fg(Color::Black)
                .bg(theme::ACCENT_GREEN)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(breadcrumb, Style::default().fg(theme::FG_PRIMARY)),
        Span::raw("  "),
        Span::styled(
            format!("上下文:{}", app.provider),
            Style::default().fg(theme::ACCENT_MAGENTA),
        ),
        Span::styled(" │ ", Style::default().fg(theme::BORDER_NORMAL)),
        Span::styled(
            format!("页:{}/{}", app.current_page, app.total_pages),
            Style::default().fg(theme::ACCENT_CYAN),
        ),
        Span::styled(" │ ", Style::default().fg(theme::BORDER_NORMAL)),
        Span::styled(
            format!("共:{}", app.total_items),
            Style::default().fg(theme::ACCENT_ORANGE),
        ),
        Span::styled(" │ ", Style::default().fg(theme::BORDER_NORMAL)),
        status,
    ]);

    let header = Paragraph::new(title).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::BORDER_FOCUS)),
    );

    f.render_widget(header, area);
}

fn render_resource_table(f: &mut Frame, app: &mut App, resource: &dyn Resource, area: Rect) {
    // Build header row
    let header_cells = app.columns.iter().map(|h| {
        Cell::from(*h).style(
            Style::default()
                .fg(theme::ACCENT_BLUE)
                .add_modifier(Modifier::BOLD),
        )
    });
    let header = Row::new(header_cells).height(1).bottom_margin(1);

    // Build data rows
    let rows = app.items.iter().map(|item| {
        let row_data = resource.format_row(item);
        let cells = row_data.into_iter().enumerate().map(|(i, content)| {
            let style = if i == 3 {
                // Status column (now index 3 after adding NAME)
                match item.status.to_lowercase().as_str() {
                    "running" | "active" | "available" => Style::default().fg(theme::STATUS_RUNNING),
                    "stopped" | "inactive" => Style::default().fg(theme::STATUS_STOPPED),
                    "error" | "failed" | "deleted" => Style::default().fg(theme::STATUS_ERROR),
                    "creating" | "pending" | "updating" => Style::default().fg(theme::STATUS_PENDING),
                    _ => Style::default().fg(theme::FG_PRIMARY),
                }
            } else if i == 0 {
                // ID column
                Style::default()
                    .fg(theme::ACCENT_CYAN)
                    .add_modifier(Modifier::BOLD)
            } else if i == 1 {
                // Name column
                Style::default().fg(theme::FG_PRIMARY)
            } else {
                Style::default().fg(theme::FG_SECONDARY)
            };
            Cell::from(content).style(style)
        });
        Row::new(cells).height(1)
    });

    // Calculate column widths based on column names
    let widths: Vec<Constraint> = app
        .columns
        .iter()
        .enumerate()
        .map(|(i, col)| match *col {
            "ID" => Constraint::Min(28),
            "NAME" => Constraint::Min(20),
            "TYPE" => Constraint::Min(14),
            "STATUS" => Constraint::Min(10),
            "CAPACITY" => Constraint::Min(12),
            "CREATED" => Constraint::Min(22),
            _ => {
                if i == app.columns.len() - 1 {
                    Constraint::Min(22)
                } else {
                    Constraint::Min(12)
                }
            }
        })
        .collect();

    let table = Table::new(rows, &widths)
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::BORDER_NORMAL)),
        )
        .row_highlight_style(
            Style::default()
                .add_modifier(Modifier::BOLD)
                .add_modifier(Modifier::UNDERLINED),
        )
        .highlight_symbol("▸ ");

    f.render_stateful_widget(table, area, &mut app.table_state);
}

fn render_fileset_table(f: &mut Frame, app: &mut App, area: Rect) {
    let columns = vec!["ID", "PATH", "STATUS", "QUOTA", "USED", "CREATED"];

    // Build header row
    let header_cells = columns.iter().map(|h| {
        Cell::from(*h).style(
            Style::default()
                .fg(theme::ACCENT_BLUE)
                .add_modifier(Modifier::BOLD),
        )
    });
    let header = Row::new(header_cells).height(1).bottom_margin(1);

    // Build data rows
    let rows = app.filesets.iter().map(|fs| {
        let quota = fs
            .quota_bytes
            .map(format_bytes)
            .unwrap_or_else(|| "-".to_string());
        let used = fs
            .used_bytes
            .map(format_bytes)
            .unwrap_or_else(|| "-".to_string());
        let created = fs.created_at.clone().unwrap_or_else(|| "-".to_string());

        let status_style = match fs.status.to_lowercase().as_str() {
            "running" | "active" | "available" => Style::default().fg(theme::STATUS_RUNNING),
            "stopped" | "inactive" => Style::default().fg(theme::STATUS_STOPPED),
            "error" | "failed" => Style::default().fg(theme::STATUS_ERROR),
            _ => Style::default().fg(theme::FG_PRIMARY),
        };

        let cells = vec![
            Cell::from(fs.id.clone()).style(
                Style::default()
                    .fg(theme::ACCENT_CYAN)
                    .add_modifier(Modifier::BOLD),
            ),
            Cell::from(fs.path.clone()).style(Style::default().fg(theme::FG_PRIMARY)),
            Cell::from(fs.status.clone()).style(status_style),
            Cell::from(quota).style(Style::default().fg(theme::FG_SECONDARY)),
            Cell::from(used).style(Style::default().fg(theme::FG_SECONDARY)),
            Cell::from(created).style(Style::default().fg(theme::FG_MUTED)),
        ];
        Row::new(cells).height(1)
    });

    let widths = vec![
        Constraint::Min(20),
        Constraint::Min(30),
        Constraint::Min(12),
        Constraint::Min(14),
        Constraint::Min(14),
        Constraint::Min(24),
    ];

    let table = Table::new(rows, &widths)
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::BORDER_NORMAL)),
        )
        .row_highlight_style(
            Style::default()
                .add_modifier(Modifier::BOLD)
                .add_modifier(Modifier::UNDERLINED),
        )
        .highlight_symbol("▸ ");

    f.render_stateful_widget(table, area, &mut app.table_state);
}

fn render_footer(f: &mut Frame, app: &App, area: Rect) {
    let key_style = Style::default()
        .fg(Color::Black)
        .bg(theme::ACCENT_GREEN)
        .add_modifier(Modifier::BOLD);
    let delete_key_style = Style::default()
        .fg(Color::Black)
        .bg(theme::ACCENT_RED)
        .add_modifier(Modifier::BOLD);
    let desc_style = Style::default().fg(theme::FG_SECONDARY);
    let sep_style = Style::default().fg(theme::BORDER_NORMAL);

    let help_text = match &app.view_mode {
        ViewMode::ResourceList => vec![
            Span::styled(" ↑↓ ", key_style),
            Span::styled(" 导航 ", desc_style),
            Span::styled("│", sep_style),
            Span::styled(" ←→ ", key_style),
            Span::styled(" 翻页 ", desc_style),
            Span::styled("│", sep_style),
            Span::styled(" ⏎ ", key_style),
            Span::styled(" 查看Fileset ", desc_style),
            Span::styled("│", sep_style),
            Span::styled(" r ", key_style),
            Span::styled(" 刷新 ", desc_style),
            Span::styled("│", sep_style),
            Span::styled(" q ", key_style),
            Span::styled(" 退出 ", desc_style),
        ],
        ViewMode::FilesetList { .. } => vec![
            Span::styled(" ↑↓ ", key_style),
            Span::styled(" 导航 ", desc_style),
            Span::styled("│", sep_style),
            Span::styled(" ←→ ", key_style),
            Span::styled(" 翻页 ", desc_style),
            Span::styled("│", sep_style),
            Span::styled(" ^D ", delete_key_style),
            Span::styled(" 删除 ", desc_style),
            Span::styled("│", sep_style),
            Span::styled(" esc ", key_style),
            Span::styled(" 返回 ", desc_style),
            Span::styled("│", sep_style),
            Span::styled(" r ", key_style),
            Span::styled(" 刷新 ", desc_style),
            Span::styled("│", sep_style),
            Span::styled(" q ", key_style),
            Span::styled(" 退出 ", desc_style),
        ],
    };

    // 显示错误或成功消息
    let content = if let Some(ref err) = app.error {
        // 计算可用宽度
        let available_width = area.width.saturating_sub(10) as usize;
        let display_err = if err.len() > available_width {
            // 截断并提示查看日志
            format!(
                "{}... (详见 /tmp/cloud-cli-debug.log)",
                err.chars().take(available_width.saturating_sub(40)).collect::<String>()
            )
        } else {
            err.clone()
        };
        
        Line::from(vec![
            Span::styled(" ✗ ", Style::default().fg(theme::ACCENT_RED)),
            Span::styled(display_err, Style::default().fg(theme::ACCENT_RED)),
        ])
    } else if let Some(ref msg) = app.success {
        Line::from(vec![
            Span::styled(" ✓ ", Style::default().fg(theme::ACCENT_GREEN)),
            Span::styled(msg.clone(), Style::default().fg(theme::ACCENT_GREEN)),
        ])
    } else {
        Line::from(help_text)
    };

    let footer = Paragraph::new(content).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::BORDER_NORMAL)),
    );

    f.render_widget(footer, area);
}
