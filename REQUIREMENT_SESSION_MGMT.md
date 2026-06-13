# Session Management Requirements

## Overview

Add persistent session management to `gemma_search.py` so conversations survive restarts, are individually addressable by ID, and support bash-style message history navigation.

## Data Model

### Chat ID

- Format: `YYYYMMDD-HHMMSS` (e.g. `20260613-181500`)
- Auto-generated on session creation
- User may assign a friendly name via `/session rename <name>`
- Friendly names are stored as metadata, not used as filenames

### Session File

Each session is a single JSON file at `.sessions/<chat_id>.json`:

```json
{
  "chat_id": "20260613-181500",
  "name": null,
  "model": "google/gemma-3-4b-it",
  "created": "2026-06-13T18:15:00",
  "updated": "2026-06-13T18:22:34",
  "messages": [
    {"role": "user", "content": "What is AAPL doing?", "timestamp": "..."},
    {"role": "assistant", "content": "...", "timestamp": "..."}
  ],
  "search_history": [],
  "fetch_history": []
}
```

### Index File

`.sessions/index.json` for fast listing without loading every session file:

```json
{
  "sessions": [
    {
      "chat_id": "20260613-181500",
      "name": null,
      "model": "google/gemma-3-4b-it",
      "created": "2026-06-13T18:15:00",
      "updated": "2026-06-13T18:22:34",
      "turns": 5
    }
  ]
}
```

## Storage Layout

```
.sessions/
  index.json
  20260613-181500.json
  20260614-093000.json
  ...
```

- Directory is auto-created on first session save
- Add `.sessions/` to `.gitignore`
- All paths relative to the script's working directory

## Auto-Save Behavior

- A new session is created automatically when the chat starts
- Session is written to disk after every assistant response completes
- On interrupt (`KeyboardInterrupt` during generation), partial response is saved if non-empty
- On `/quit` or EOF, session is saved before exit
- No explicit "save" action needed from the user

## Commands

| Command | Description |
|---|---|
| `/session` | List all saved sessions (newest first), highlight current |
| `/session new` | Start a fresh session (auto-saves current first) |
| `/session save [name]` | Save current session with optional friendly name |
| `/session load <id_or_name>` | Restore a session by chat_id prefix or friendly name |
| `/session delete <id_or_name>` | Delete a session file and update index |
| `/session rename <name>` | Assign a friendly name to the current session |
| `/session info` | Show current session metadata (id, name, turns, duration) |

### Load Matching

- Prefix match on chat_id: `/session load 202606` loads the first match
- Exact match on friendly name (case-insensitive)
- If ambiguous, list matches and ask user to clarify
- After loading, conversation resumes with full message history passed to the model

### List Display

```
  Sessions
  ─────────────────────────────────────────────
  * 20260613-181500  gemma-3-4b-it  5 turns  "AAPL research"
    20260614-093000  gemma-3-4b-it  12 turns
    20260614-140000  gemma-2-2b-it  3 turns  "quick Q&A"
  ─────────────────────────────────────────────
  * = current session
```

## Message History Navigation

- Integrate Python `readline` module on the `input()` prompt
- Each session maintains its own readline history (separate from other sessions)
- Up/Down arrow keys cycle through previously sent user messages within the current session
- On `/session load`, the loaded session's user messages are pushed into readline history so Up arrow recalls them
- History is bounded to the last 500 messages per session to avoid memory bloat
- Ctrl+R is NOT used (avoid readline reverse-search collision)

## Banner Update

Update the banner to show the current session ID:

```
  ╭───────────────────────────────────────────────────╮
  │  🌐 Gemma-4 Search  ·  HuggingFace + Web          │
  │  session: 20260613-181500                         │
  ╰───────────────────────────────────────────────────╯
```

## Edge Cases

- If `.sessions/` doesn't exist, `/session` shows "No saved sessions"
- If a session file is corrupted JSON, skip it with a warning (don't crash)
- Loading a session that used a different model: warn the user, allow it anyway
- Very long conversations (>1000 messages): truncate context window for the model prompt, but keep full history in the session file
- `/session delete` on the current session: start a new session automatically

## Non-Goals

- No multi-user support
- No encryption of session files
- No session export/import formats beyond JSON
- No branching or forking sessions
