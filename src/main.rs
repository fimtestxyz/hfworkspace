mod app;
mod event;
mod models;
mod ui;

use std::io;
use std::sync::Arc;

use crossterm::{
    event::{self as crossterm_event, Event, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{Terminal, backend::CrosstermBackend};
use tokio::sync::mpsc;

use app::App;
use event::{AppEvent, handle_events};
use models::ModelManager;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ── Setup terminal ────────────────────────────────
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    // ── Setup channels & state ─────────────────────────
    let (tx, mut rx) = mpsc::channel::<AppEvent>(256);

    let model_manager = Arc::new(ModelManager::new()?);
    let mut app = App::new();

    // Populate model list from HF cache
    app.models = model_manager.list_models();
    if app.models.is_empty() {
        app.set_status("No models found. Press 'd' to download one.");
    } else {
        app.set_status(format!("Found {} models in catalog", app.models.len()));
    }

    // ── Spawn crossterm key listener ───────────────────
    let key_tx = tx.clone();
    tokio::spawn(async move {
        loop {
            if crossterm_event::poll(std::time::Duration::from_millis(50)).unwrap() {
                if let Ok(Event::Key(key)) = crossterm_event::read() {
                    if key.kind == KeyEventKind::Press {
                        let _ = key_tx.send(AppEvent::Key(key)).await;
                    }
                }
            }
        }
    });

    // ── Main event loop ────────────────────────────────
    loop {
        // Draw UI
        terminal.draw(|f| ui::draw(f, &mut app))?;

        // Process next event
        if let Some(app_event) = rx.recv().await {
            handle_events(&mut app, app_event, model_manager.clone(), tx.clone()).await;

            if app.should_quit {
                break;
            }
        }
    }

    // ── Restore terminal ───────────────────────────────
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;

    Ok(())
}
