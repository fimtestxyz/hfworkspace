use std::sync::Arc;
use crossterm::event::KeyCode;
use crossterm::event::KeyModifiers;
use crossterm::event::KeyEvent;
use tokio::sync::mpsc;

use crate::app::{App, ChatMessage, MessageRole, Mode, ModelStatus};
use crate::models::ModelManager;

pub enum AppEvent {
    Key(KeyEvent),
    DownloadProgress(String, f32),
    DownloadComplete(String),
    DownloadError(String, String),
    StreamToken(String),
    StreamComplete,
    StreamError(String),
}

pub async fn handle_events(
    app: &mut App,
    event: AppEvent,
    model_manager: Arc<ModelManager>,
    event_sender: mpsc::Sender<AppEvent>,
) {
    match event {
        AppEvent::Key(key_event) => {
            handle_key_event(app, key_event, model_manager, event_sender).await;
        }
        AppEvent::DownloadProgress(repo_id, progress) => {
            if let Some(m) = app.models.iter_mut().find(|m| m.repo_id == repo_id) {
                m.status = ModelStatus::Downloading(progress);
            }
        }
        AppEvent::DownloadComplete(repo_id) => {
            app.set_status(format!("Finished downloading {repo_id}"));
            if let Some(m) = app.models.iter_mut().find(|m| m.repo_id == repo_id) {
                m.status = ModelStatus::Downloaded;
                // Recalculate size
                m.size_mb = model_manager.list_models()
                    .into_iter()
                    .find(|x| x.repo_id == repo_id)
                    .and_then(|x| x.size_mb);
            }
        }
        AppEvent::DownloadError(repo_id, err) => {
            app.set_status(format!("Error downloading {repo_id}: {err}"));
            if let Some(m) = app.models.iter_mut().find(|m| m.repo_id == repo_id) {
                m.status = ModelStatus::Error(err);
            }
        }
        AppEvent::StreamToken(token) => {
            app.streaming_text.push_str(&token);
        }
        AppEvent::StreamComplete => {
            app.is_generating = false;
            let response = std::mem::take(&mut app.streaming_text);
            app.messages.push(ChatMessage {
                role: MessageRole::Assistant,
                content: response,
            });
        }
        AppEvent::StreamError(err) => {
            app.is_generating = false;
            app.messages.push(ChatMessage {
                role: MessageRole::System,
                content: format!("Generation error: {err}"),
            });
        }
    }
}

async fn handle_key_event(
    app: &mut App,
    key_event: KeyEvent,
    model_manager: Arc<ModelManager>,
    event_sender: mpsc::Sender<AppEvent>,
) {
    // Global quits
    if key_event.modifiers.contains(KeyModifiers::CONTROL) && key_event.code == KeyCode::Char('c') {
        app.should_quit = true;
        return;
    }

    if app.mode == Mode::Help {
        app.mode = app.previous_mode.clone();
        return;
    }

    match app.mode {
        Mode::Models => handle_models_keys(app, key_event, model_manager, event_sender).await,
        Mode::Chat => handle_chat_keys(app, key_event, event_sender),
        Mode::DownloadPopup => handle_download_popup_keys(app, key_event, model_manager, event_sender),
        Mode::Help => {}
    }
}

async fn handle_models_keys(
    app: &mut App,
    key_event: KeyEvent,
    model_manager: Arc<ModelManager>,
    event_sender: mpsc::Sender<AppEvent>,
) {
    match key_event.code {
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Tab => {
            app.previous_mode = Mode::Models;
            app.mode = Mode::Chat;
        }
        KeyCode::Char('?') => {
            app.previous_mode = Mode::Models;
            app.mode = Mode::Help;
        }
        KeyCode::Down | KeyCode::Char('j') => {
            let i = match app.model_list_state.selected() {
                Some(i) => {
                    if i >= app.models.len().saturating_sub(1) {
                        0
                    } else {
                        i + 1
                    }
                }
                None => 0,
            };
            app.model_list_state.select(Some(i));
        }
        KeyCode::Up | KeyCode::Char('k') => {
            let i = match app.model_list_state.selected() {
                Some(i) => {
                    if i == 0 {
                        app.models.len().saturating_sub(1)
                    } else {
                        i - 1
                    }
                }
                None => 0,
            };
            app.model_list_state.select(Some(i));
        }
        KeyCode::Char('d') => {
            app.download_input.clear();
            app.mode = Mode::DownloadPopup;
        }
        KeyCode::Char('x') => {
            if let Some(index) = app.model_list_state.selected() {
                if let Some(m) = app.models.get(index) {
                    let repo_id = m.repo_id.clone();
                    match model_manager.delete_model(&repo_id) {
                        Ok(msg) => {
                            app.set_status(msg);
                            if let Some(m_mut) = app.models.get_mut(index) {
                                m_mut.status = ModelStatus::NotDownloaded;
                                m_mut.size_mb = None;
                            }
                        }
                        Err(e) => app.set_status(format!("Error: {e}")),
                    }
                }
            }
        }
        KeyCode::Enter | KeyCode::Char('l') => {
            if let Some(index) = app.model_list_state.selected() {
                // Clone repo_id before mutating
                let repo_id = app.models.get(index).map(|m| m.repo_id.clone());
                let status = app.models.get(index).map(|m| m.status.clone());

                if let (Some(repo_id), Some(status)) = (repo_id, status) {
                    match status {
                        ModelStatus::Downloaded | ModelStatus::Loaded => {
                            // Unload previous
                            if let Some(loaded_repo) = &app.loaded_model {
                                if let Some(old) = app.models.iter_mut().find(|x| x.repo_id == *loaded_repo) {
                                    old.status = ModelStatus::Downloaded;
                                }
                            }
                            app.loaded_model = Some(repo_id.clone());
                            app.set_status(format!("Loaded model: {repo_id}"));
                            if let Some(m_mut) = app.models.get_mut(index) {
                                m_mut.status = ModelStatus::Loaded;
                            }
                            // Auto switch to chat
                            app.mode = Mode::Chat;
                        }
                        ModelStatus::NotDownloaded => {
                            app.set_status(format!("Starting download for {repo_id}..."));
                            trigger_download(repo_id, model_manager, event_sender);
                        }
                        _ => {}
                    }
                }
            }
        }
        _ => {}
    }
}

fn handle_chat_keys(
    app: &mut App,
    key_event: KeyEvent,
    event_sender: mpsc::Sender<AppEvent>,
) {
    // Use Escape to switch to Models, so all printable characters work in chat
    match key_event.code {
        KeyCode::Esc => {
            app.previous_mode = Mode::Chat;
            if app.is_generating {
                // Cancel generation
                app.is_generating = false;
                let partial = std::mem::take(&mut app.streaming_text);
                if !partial.is_empty() {
                    app.messages.push(ChatMessage {
                        role: MessageRole::Assistant,
                        content: partial,
                    });
                }
            } else {
                app.mode = Mode::Models;
            }
        }
        KeyCode::F(1) => {
            app.previous_mode = Mode::Chat;
            app.mode = Mode::Help;
        }
        KeyCode::Enter if !app.is_generating => {
            let prompt = std::mem::take(&mut app.chat_input);
            if !prompt.trim().is_empty() {
                app.messages.push(ChatMessage {
                    role: MessageRole::User,
                    content: prompt.clone(),
                });
                app.is_generating = true;
                app.streaming_text.clear();
                let model_repo = app.loaded_model.clone();
                trigger_chat_generation(prompt, model_repo, event_sender);
            }
        }
        KeyCode::Backspace if !app.is_generating => {
            app.chat_input.pop();
        }
        KeyCode::Char(c) if !app.is_generating => {
            app.chat_input.push(c);
        }
        // Ignore other keys during generation
        _ => {}
    }
}

fn handle_download_popup_keys(
    app: &mut App,
    key_event: KeyEvent,
    model_manager: Arc<ModelManager>,
    event_sender: mpsc::Sender<AppEvent>,
) {
    match key_event.code {
        KeyCode::Esc => {
            app.mode = Mode::Models;
        }
        KeyCode::Char(c) => {
            app.download_input.push(c);
        }
        KeyCode::Backspace => {
            app.download_input.pop();
        }
        KeyCode::Enter => {
            let repo_id = std::mem::take(&mut app.download_input);
            if !repo_id.trim().is_empty() {
                if !app.models.iter().any(|m| m.repo_id == repo_id) {
                    app.models.push(crate::app::ModelEntry {
                        repo_id: repo_id.clone(),
                        status: ModelStatus::NotDownloaded,
                        size_mb: None,
                        description: "Custom model".to_string(),
                        last_used: None,
                    });
                }
                app.set_status(format!("Querying metadata for {repo_id}..."));
                app.mode = Mode::Models;
                trigger_download(repo_id, model_manager, event_sender);
            }
        }
        _ => {}
    }
}

fn trigger_download(
    repo_id: String,
    model_manager: Arc<ModelManager>,
    event_sender: mpsc::Sender<AppEvent>,
) {
    tokio::spawn(async move {
        let _ = event_sender.send(AppEvent::DownloadProgress(repo_id.clone(), 0.10)).await;
        match model_manager.download_model(&repo_id).await {
            Ok(_) => {
                let _ = event_sender.send(AppEvent::DownloadComplete(repo_id)).await;
            }
            Err(e) => {
                let _ = event_sender.send(AppEvent::DownloadError(repo_id, e.to_string())).await;
            }
        }
    });
}

fn trigger_chat_generation(
    prompt: String,
    model_repo: Option<String>,
    event_sender: mpsc::Sender<AppEvent>,
) {
    tokio::spawn(async move {
        // Simulation mode when no model is loaded
        if model_repo.is_none() {
            let _ = event_sender
                .send(AppEvent::StreamToken("🤖 (Simulation) ".to_string()))
                .await;
            let mock_response = format!(
                "This is a simulated response to: \"{prompt}\". \
                 To chat with a real model, load a model from the Models screen (Tab)."
            );
            for word in mock_response.split_whitespace() {
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                let _ = event_sender
                    .send(AppEvent::StreamToken(format!("{word} ")))
                    .await;
            }
            let _ = event_sender.send(AppEvent::StreamComplete).await;
            return;
        }

        // If a model is marked as loaded, call Python for real inference
        let _repo = model_repo.unwrap();
        let output = tokio::process::Command::new("python3")
            .arg("-c")
            .arg(format!(
                "import sys; sys.path.insert(0, '/Volumes/wwk_nvme/Users/wwkoon/hfworkspace'); \
                 from gemma_simple import load_gemma_model, generate_text; \
                 t, m = load_gemma_model(); \
                 prompt = '<start_of_turn>user\\n{prompt}<end_of_turn>\\n<start_of_turn>model\\n'; \
                 print(generate_text(prompt, t, m, max_length=200), end='')"
            ))
            .output()
            .await;

        match output {
            Ok(out) => {
                let text = String::from_utf8_lossy(&out.stdout).to_string();
                let _ = event_sender
                    .send(AppEvent::StreamToken(text))
                    .await;
                let _ = event_sender.send(AppEvent::StreamComplete).await;
            }
            Err(e) => {
                let _ = event_sender
                    .send(AppEvent::StreamError(e.to_string()))
                    .await;
            }
        }
    });
}
