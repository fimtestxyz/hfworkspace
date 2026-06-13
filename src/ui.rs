use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph},
};

use crate::app::{App, MessageRole, Mode, ModelStatus};

// ─── Catppuccin Mocha Color Palette ─────────────────────

const SURFACE: Color = Color::Rgb(30, 30, 46);
const TEXT: Color = Color::Rgb(205, 214, 244);
const SUBTEXT: Color = Color::Rgb(166, 173, 200);
const ACCENT: Color = Color::Rgb(137, 180, 250);
const GREEN: Color = Color::Rgb(166, 227, 161);
const RED: Color = Color::Rgb(243, 139, 168);
const YELLOW: Color = Color::Rgb(249, 226, 175);
const MAUVE: Color = Color::Rgb(203, 166, 247);
const PEACH: Color = Color::Rgb(250, 179, 135);
const TEAL: Color = Color::Rgb(148, 226, 213);
const LAVENDER: Color = Color::Rgb(180, 190, 254);

// ─── Main draw dispatcher ────────────────────────────────

pub fn draw(f: &mut Frame, app: &mut App) {
    match app.mode {
        Mode::Models => draw_models(f, app),
        Mode::Chat => draw_chat(f, app),
        Mode::Help => {
            match app.previous_mode {
                Mode::Chat => draw_chat(f, app),
                _ => draw_models(f, app),
            }
            draw_help_popup(f);
        }
        Mode::DownloadPopup => {
            draw_models(f, app);
            draw_download_popup(f, app);
        }
        Mode::SearchPopup => {
            draw_models(f, app);
            draw_search_popup(f, app);
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  MODELS SCREEN
// ═══════════════════════════════════════════════════════════

fn draw_models(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(3),
        ])
        .split(f.area());

    // ── Header ──
    let filter_indicator = if app.show_installed_only { " [Installed Only]" } else { "" };
    let header = Paragraph::new(Line::from(vec![
        Span::styled("  🤗 ", Style::default().fg(YELLOW)),
        Span::styled("HuggingFace ", Style::default().fg(TEXT).bold()),
        Span::styled("Model Manager", Style::default().fg(ACCENT).bold()),
        Span::styled(filter_indicator, Style::default().fg(PEACH).bold()),
        Span::raw("  "),
        Span::styled(
            "Tab=Chat  s=Search  i=Installed  d=Download  ?=Help  q=Quit",
            Style::default().fg(SUBTEXT),
        ),
    ]))
    .block(Block::default().style(Style::default().bg(SURFACE)));
    f.render_widget(header, chunks[0]);

    // ── Model List ──
    let display_indices: Vec<usize> = if app.show_installed_only {
        app.models
            .iter()
            .enumerate()
            .filter(|(_, m)| matches!(m.status, ModelStatus::Downloaded | ModelStatus::Loaded))
            .map(|(i, _)| i)
            .collect()
    } else {
        (0..app.models.len()).collect()
    };
    app.display_indices = display_indices;

    let items: Vec<ListItem> = app
        .display_indices
        .iter()
        .map(|&i| &app.models[i])
        .map(|m| {
            let status_style = match &m.status {
                ModelStatus::Downloaded => Style::default().fg(GREEN),
                ModelStatus::Loaded => Style::default().fg(ACCENT).bold(),
                ModelStatus::Downloading(_p) => Style::default().fg(YELLOW),
                ModelStatus::NotDownloaded => Style::default().fg(SUBTEXT),
                ModelStatus::Error(_) => Style::default().fg(RED),
            };

            let status_text = match &m.status {
                ModelStatus::Downloading(p) => format!("↓ {}%", (*p * 100.0) as u32),
                other => other.to_string(),
            };

            let size_text = m
                .size_mb
                .map(|s| format!("{s} MB"))
                .unwrap_or_else(|| "—".to_string());

            let line = Line::from(vec![
                Span::styled(format!(" {} ", status_text), status_style),
                Span::styled(&m.repo_id, Style::default().fg(TEXT).bold()),
                Span::raw("  "),
                Span::styled(size_text, Style::default().fg(LAVENDER)),
                Span::raw("  "),
                Span::styled(&m.description, Style::default().fg(SUBTEXT)),
            ]);
            ListItem::new(line)
        })
        .collect();

    let list_count = app.display_indices.len();
    let list_title = if app.show_installed_only {
        format!("Installed ({list_count})")
    } else {
        format!("Models ({list_count})")
    };

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(MAUVE))
                .title(Line::from(vec![
                    Span::styled(" 📦 ", Style::default().fg(PEACH)),
                    Span::styled(list_title, Style::default().fg(TEXT).bold()),
                ]))
                .style(Style::default().bg(Color::Reset)),
        )
        .highlight_style(
            Style::default()
                .bg(ACCENT)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");

    f.render_stateful_widget(list, chunks[1], &mut app.model_list_state);

    // ── Status Bar ──
    let status_text = if let Some((msg, time)) = &app.status_message {
        format!(" {} | {} | {} models loaded", time, msg, app.models.len())
    } else {
        format!(" {} models in catalog", app.models.len())
    };

    let status = Paragraph::new(Line::from(Span::styled(
        status_text,
        Style::default().fg(TEAL),
    )))
    .block(Block::default().style(Style::default().bg(SURFACE)));
    f.render_widget(status, chunks[2]);
}

// ═══════════════════════════════════════════════════════════
//  CHAT SCREEN
// ═══════════════════════════════════════════════════════════

fn draw_chat(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(3),
        ])
        .split(f.area());

    // ── Header ──
    let model_name = app.loaded_model.as_deref().unwrap_or("No model loaded");
    let header = Paragraph::new(Line::from(vec![
        Span::styled(" 💬 ", Style::default().fg(GREEN)),
        Span::styled("Chat", Style::default().fg(TEXT).bold()),
        Span::raw("  "),
        Span::styled(format!("Model: {model_name}"), Style::default().fg(ACCENT)),
        Span::raw("  "),
        Span::styled(
            "Tab=Models  Esc=Back  F1=Help  Enter=Send",
            Style::default().fg(SUBTEXT),
        ),
    ]))
    .block(Block::default().style(Style::default().bg(SURFACE)));
    f.render_widget(header, chunks[0]);

    // ── Messages ──
    let mut lines: Vec<Line> = Vec::new();

    if app.messages.is_empty() {
        lines.push(Line::from(Span::styled(
            "  Welcome! Select a model (Tab to Models screen), then type a message.",
            Style::default().fg(SUBTEXT),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  If no model is loaded, responses will be simulated.",
            Style::default().fg(SUBTEXT).italic(),
        )));
    }

    for msg in &app.messages {
        match msg.role {
            MessageRole::System => {
                lines.push(Line::from(Span::styled(
                    format!("  ── {} ──", msg.content),
                    Style::default().fg(YELLOW).italic(),
                )));
            }
            MessageRole::User => {
                lines.push(Line::from(""));
                lines.push(Line::from(vec![
                    Span::styled("  🧑 ", Style::default().fg(PEACH)),
                    Span::styled("You", Style::default().fg(PEACH).bold()),
                ]));
                for text_line in wrap_text(&msg.content, 2) {
                    lines.push(Line::from(Span::styled(text_line, Style::default().fg(TEXT))));
                }
            }
            MessageRole::Assistant => {
                lines.push(Line::from(""));
                lines.push(Line::from(vec![
                    Span::styled("  🤖 ", Style::default().fg(GREEN)),
                    Span::styled("Assistant", Style::default().fg(GREEN).bold()),
                ]));
                for text_line in wrap_text(&msg.content, 2) {
                    lines.push(Line::from(Span::styled(text_line, Style::default().fg(LAVENDER))));
                }
            }
        }
    }

    // Streaming text while generating
    if app.is_generating && !app.streaming_text.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("  🤖 ", Style::default().fg(GREEN)),
            Span::styled("Assistant", Style::default().fg(GREEN).bold()),
        ]));
        for text_line in wrap_text(&app.streaming_text, 2) {
            lines.push(Line::from(Span::styled(text_line, Style::default().fg(LAVENDER))));
        }
        lines.push(Line::from(Span::styled(
            "  ▌",
            Style::default().fg(ACCENT).slow_blink(),
        )));
    }

    let msg_count = lines.len() as u16;
    let area_height = chunks[1].height.saturating_sub(2);
    let scroll = if msg_count > area_height { msg_count - area_height } else { 0 };

    let messages = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(MAUVE))
                .style(Style::default().bg(Color::Reset)),
        )
        .scroll((scroll, 0));

    f.render_widget(messages, chunks[1]);

    // ── Input Box ──
    let input_style = if app.is_generating {
        Style::default().fg(SUBTEXT)
    } else {
        Style::default().fg(TEXT)
    };

    let prompt_symbol = if app.is_generating {
        Span::styled(" ⏳ ", Style::default().fg(YELLOW))
    } else {
        Span::styled(" > ", Style::default().fg(GREEN))
    };

    let input_text = Paragraph::new(Line::from(vec![
        prompt_symbol,
        Span::styled(&app.chat_input, input_style),
        Span::styled("▌", Style::default().fg(ACCENT)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(if app.is_generating { YELLOW } else { ACCENT }))
            .title(if app.is_generating {
                Line::from(Span::styled(" Generating… ", Style::default().fg(YELLOW)))
            } else {
                Line::from(Span::styled(" Input ", Style::default().fg(ACCENT)))
            })
            .style(Style::default().bg(Color::Reset)),
    );

    f.render_widget(input_text, chunks[2]);
}

// ═══════════════════════════════════════════════════════════
//  DOWNLOAD POPUP
// ═══════════════════════════════════════════════════════════

fn draw_download_popup(f: &mut Frame, app: &App) {
    let area = centered_rect(60, 20, f.area());
    f.render_widget(Clear, area);

    let popup = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            "  Enter HuggingFace model repo ID:",
            Style::default().fg(TEAL),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  > ", Style::default().fg(GREEN)),
            Span::styled(&app.download_input, Style::default().fg(TEXT)),
            Span::styled("▌", Style::default().fg(ACCENT)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  e.g. google/gemma-2-2b-it",
            Style::default().fg(SUBTEXT).italic(),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "  Enter=Download  Esc=Cancel",
            Style::default().fg(YELLOW),
        )),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(MAUVE))
            .title(Line::from(vec![
                Span::styled(" ⬇ ", Style::default().fg(YELLOW)),
                Span::styled("Download Model", Style::default().fg(TEXT).bold()),
            ]))
            .style(Style::default().bg(SURFACE)),
    );

    f.render_widget(popup, area);
}

// ═══════════════════════════════════════════════════════════
//  SEARCH POPUP
// ═══════════════════════════════════════════════════════════

fn draw_search_popup(f: &mut Frame, app: &mut App) {
    let area = centered_rect(75, 70, f.area());
    f.render_widget(Clear, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(1),
            Constraint::Min(3),
            Constraint::Length(1),
        ])
        .split(area);

    // ── Search input ──
    let input = Paragraph::new(Line::from(vec![
        Span::styled(" > ", Style::default().fg(GREEN)),
        Span::styled(&app.search_input, Style::default().fg(TEXT)),
        Span::styled("▌", Style::default().fg(ACCENT)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(ACCENT))
            .title(Line::from(vec![
                Span::styled(" 🔍 ", Style::default().fg(YELLOW)),
                Span::styled("Search HuggingFace Hub", Style::default().fg(TEXT).bold()),
            ]))
            .style(Style::default().bg(SURFACE)),
    );
    f.render_widget(input, chunks[0]);

    // ── Hint line ──
    let hint = if app.search_results.is_empty() && !app.search_loading {
        "  Type a query and press Enter to search  ·  Esc=Cancel"
    } else if app.search_loading {
        "  Searching..."
    } else {
        "  ↑/↓=Navigate  Enter=Install  Esc=Cancel"
    };
    let hint_line = Paragraph::new(Line::from(Span::styled(
        hint,
        Style::default().fg(SUBTEXT).italic(),
    )))
    .style(Style::default().bg(SURFACE));
    f.render_widget(hint_line, chunks[1]);

    // ── Results list ──
    if app.search_loading {
        let loading = Paragraph::new(Line::from(Span::styled(
            "  ⏳ Loading results...",
            Style::default().fg(YELLOW),
        )))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(MAUVE))
                .style(Style::default().bg(Color::Reset)),
        );
        f.render_widget(loading, chunks[2]);
    } else if app.search_results.is_empty() {
        let empty = Paragraph::new(Line::from(Span::styled(
            "  No results. Type a search query above.",
            Style::default().fg(SUBTEXT),
        )))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(MAUVE))
                .title(Line::from(vec![
                    Span::styled(" Results ", Style::default().fg(TEXT).bold()),
                ]))
                .style(Style::default().bg(Color::Reset)),
        );
        f.render_widget(empty, chunks[2]);
    } else {
        let items: Vec<ListItem> = app
            .search_results
            .iter()
            .map(|r| {
                let downloads = format_downloads(r.downloads);
                let task = r.pipeline_tag.as_deref().unwrap_or("—");
                let line = Line::from(vec![
                    Span::styled(&r.repo_id, Style::default().fg(TEXT).bold()),
                    Span::raw("  "),
                    Span::styled(format!("↓{downloads}"), Style::default().fg(GREEN)),
                    Span::raw("  "),
                    Span::styled(format!("♥{}", r.likes), Style::default().fg(RED)),
                    Span::raw("  "),
                    Span::styled(format!("[{task}]"), Style::default().fg(TEAL)),
                ]);
                ListItem::new(line)
            })
            .collect();

        let result_count = app.search_results.len();
        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(MAUVE))
                    .title(Line::from(vec![
                        Span::styled(
                            format!(" Results ({result_count}) "),
                            Style::default().fg(TEXT).bold(),
                        ),
                    ]))
                    .style(Style::default().bg(Color::Reset)),
            )
            .highlight_style(
                Style::default()
                    .bg(ACCENT)
                    .fg(Color::Black)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("▶ ");

        f.render_stateful_widget(list, chunks[2], &mut app.search_list_state);
    }

    // ── Footer ──
    let footer = Paragraph::new(Line::from(Span::styled(
        format!(
            " {} result{}  ·  Enter=Install selected  ·  Esc=Close",
            app.search_results.len(),
            if app.search_results.len() == 1 { "" } else { "s" },
        ),
        Style::default().fg(TEAL),
    )))
    .style(Style::default().bg(SURFACE));
    f.render_widget(footer, chunks[3]);
}

fn format_downloads(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

// ═══════════════════════════════════════════════════════════
//  HELP POPUP
// ═══════════════════════════════════════════════════════════

fn draw_help_popup(f: &mut Frame) {
    let area = centered_rect(55, 55, f.area());
    f.render_widget(Clear, area);

    let help = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            "  ⌨  Keyboard Shortcuts",
            Style::default().fg(ACCENT).bold(),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Tab       ", Style::default().fg(PEACH).bold()),
            Span::styled("Switch Models ↔ Chat", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  d         ", Style::default().fg(PEACH).bold()),
            Span::styled("Download model (popup)", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  s         ", Style::default().fg(PEACH).bold()),
            Span::styled("Search HuggingFace Hub", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  i         ", Style::default().fg(PEACH).bold()),
            Span::styled("Toggle installed-only filter", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  l / Enter  ", Style::default().fg(PEACH).bold()),
            Span::styled("Load selected model", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  x         ", Style::default().fg(PEACH).bold()),
            Span::styled("Delete model from cache", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  ↑ / ↓     ", Style::default().fg(PEACH).bold()),
            Span::styled("Navigate model list", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  j / k     ", Style::default().fg(PEACH).bold()),
            Span::styled("Navigate (vim-style)", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  Enter     ", Style::default().fg(PEACH).bold()),
            Span::styled("Send message / confirm", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  F1        ", Style::default().fg(PEACH).bold()),
            Span::styled("Help (Chat mode)", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  Esc       ", Style::default().fg(PEACH).bold()),
            Span::styled("Models screen / cancel generation", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  Ctrl+C    ", Style::default().fg(PEACH).bold()),
            Span::styled("Quit application", Style::default().fg(TEXT)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  Press any key to close",
            Style::default().fg(SUBTEXT).italic(),
        )),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(MAUVE))
            .title(" ❓ Help ")
            .style(Style::default().bg(SURFACE)),
    );

    f.render_widget(help, area);
}

// ─── Helpers ─────────────────────────────────────────────

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

/// Simple word-wrap with left indent
fn wrap_text(text: &str, indent: usize) -> Vec<String> {
    let prefix = " ".repeat(indent);
    text.split('\n')
        .flat_map(|line| {
            if line.is_empty() {
                vec![String::new()]
            } else {
                let mut result = Vec::new();
                let mut current = String::new();
                for word in line.split_whitespace() {
                    if current.is_empty() {
                        current = format!("{prefix}{word}");
                    } else if current.len() + word.len() + 1 > 80 + indent {
                        result.push(current);
                        current = format!("{prefix}{word}");
                    } else {
                        current.push(' ');
                        current.push_str(word);
                    }
                }
                if !current.is_empty() {
                    result.push(current);
                }
                result
            }
        })
        .collect()
}
