use std::fmt;

/// Application modes
#[derive(Debug, Clone, PartialEq)]
pub enum Mode {
    Models,
    Chat,
    DownloadPopup,
    Help,
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Mode::Models => write!(f, "Models"),
            Mode::Chat => write!(f, "Chat"),
            Mode::DownloadPopup => write!(f, "Download"),
            Mode::Help => write!(f, "Help"),
        }
    }
}

/// Model status
#[derive(Debug, Clone, PartialEq)]
pub enum ModelStatus {
    Downloaded,
    Loaded,
    Downloading(f32),
    NotDownloaded,
    Error(String),
}

impl fmt::Display for ModelStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelStatus::Downloaded => write!(f, "✓ Downloaded"),
            ModelStatus::Loaded => write!(f, "● Loaded"),
            ModelStatus::Downloading(p) => write!(f, "↓ {}%", (p * 100.0) as u32),
            ModelStatus::NotDownloaded => write!(f, "○ Not downloaded"),
            ModelStatus::Error(e) => write!(f, "✗ {e}"),
        }
    }
}

/// A local model entry
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ModelEntry {
    pub repo_id: String,
    pub status: ModelStatus,
    pub size_mb: Option<u64>,
    pub description: String,
    pub last_used: Option<String>,
}

/// Chat message
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

/// Application state
#[allow(dead_code)]
pub struct App {
    pub mode: Mode,
    pub previous_mode: Mode,
    pub should_quit: bool,

    // Model management
    pub models: Vec<ModelEntry>,
    pub model_list_state: ratatui::widgets::ListState,
    pub download_input: String,
    pub status_message: Option<(String, String)>,

    // Chat
    pub messages: Vec<ChatMessage>,
    pub chat_input: String,
    pub chat_scroll: u16,
    pub is_generating: bool,
    pub loaded_model: Option<String>,
    pub streaming_text: String,

    // Popup
    pub popup_scroll: u16,
}

impl App {
    pub fn new() -> Self {
        let mut model_list_state = ratatui::widgets::ListState::default();
        model_list_state.select(Some(0));

        Self {
            mode: Mode::Models,
            previous_mode: Mode::Models,
            should_quit: false,

            models: Vec::new(),
            model_list_state,
            download_input: String::new(),
            status_message: None,

            messages: Vec::new(),
            chat_input: String::new(),
            chat_scroll: 0,
            is_generating: false,
            loaded_model: None,
            streaming_text: String::new(),
            popup_scroll: 0,
        }
    }

    pub fn set_status(&mut self, msg: impl Into<String>) {
        let now = chrono::Local::now();
        let time_str = now.format("%H:%M:%S").to_string();
        self.status_message = Some((msg.into(), time_str));
    }

    #[allow(dead_code)]
    pub fn clear_status(&mut self) {
        self.status_message = None;
    }
}
