use anyhow::Result;
use hf_hub::{Cache, Repo, RepoType};
use std::path::PathBuf;

use super::app::{ModelEntry, ModelStatus, SearchResult};

/// Manages HuggingFace model cache and downloads
pub struct ModelManager {
    cache: Cache,
}

impl ModelManager {
    pub fn new() -> Result<Self> {
        let cache_dir = dirs::home_dir()
            .expect("No home directory")
            .join(".cache")
            .join("huggingface")
            .join("hub");

        let cache = Cache::new(cache_dir);
        Ok(Self { cache })
    }

    /// List popular models, checking which are already cached locally
    pub fn list_models(&self) -> Vec<ModelEntry> {
        let popular = vec![
            ("google/gemma-2-2b-it", "Google Gemma 2 2B Instruct — small & fast"),
            ("google/gemma-2-9b-it", "Google Gemma 2 9B Instruct — capable"),
            ("meta-llama/Llama-3.2-3B-Instruct", "Meta Llama 3.2 3B Instruct"),
            ("meta-llama/Llama-3.1-8B-Instruct", "Meta Llama 3.1 8B Instruct"),
            ("microsoft/Phi-3-mini-4k-instruct", "Microsoft Phi-3 Mini 4K"),
            ("HuggingFaceH4/zephyr-7b-beta", "Zephyr 7B Beta"),
            ("Qwen/Qwen2.5-3B-Instruct", "Alibaba Qwen 2.5 3B Instruct"),
            ("NousResearch/Hermes-3-Llama-3.1-8B", "Nous Hermes 3 Llama 3.1 8B"),
        ];

        popular
            .into_iter()
            .map(|(repo_id, desc)| {
                let status = if self.is_downloaded(repo_id) {
                    ModelStatus::Downloaded
                } else {
                    ModelStatus::NotDownloaded
                };

                let size_mb = self.get_local_size(repo_id);

                ModelEntry {
                    repo_id: repo_id.to_string(),
                    status,
                    size_mb,
                    description: desc.to_string(),
                    last_used: None,
                }
            })
            .collect()
    }

    /// Check if model exists in local cache
    fn is_downloaded(&self, repo_id: &str) -> bool {
        let cache_repo = self.cache.model(repo_id.to_string());
        // Check if any file exists in the cache for this model
        cache_repo.get("config.json").is_some()
            || cache_repo.get("model.safetensors").is_some()
            || cache_repo.get("model.safetensors.index.json").is_some()
    }

    /// Calculate total size of cached model files in MB
    fn get_local_size(&self, repo_id: &str) -> Option<u64> {
        let _cache_repo = self.cache.model(repo_id.to_string());
        // Walk the cache directory for this model
        let _cache_path = self.cache.path().join("models--").join(
            repo_id.replace('/', "--models--").replace('/', "--"),
        );

        // Use the actual hf-hub cache folder naming convention
        let folder_name = Repo::new(repo_id.to_string(), RepoType::Model).folder_name();
        let model_cache_dir = self.cache.path().join(folder_name);

        if !model_cache_dir.exists() {
            return None;
        }

        let mut total: u64 = 0;
        self.walk_dir(&model_cache_dir, &mut total);
        if total > 0 {
            Some(total / (1024 * 1024))
        } else {
            None
        }
    }

    /// Recursively walk a directory to sum file sizes
    fn walk_dir(&self, dir: &PathBuf, total: &mut u64) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    if meta.is_file() {
                        *total += meta.len();
                    } else if meta.is_dir() {
                        self.walk_dir(&entry.path(), total);
                    }
                }
            }
        }
    }

    /// Download a model via hf-hub
    pub async fn download_model(&self, repo_id: &str) -> Result<String> {
        use hf_hub::api::tokio::ApiBuilder;

        let api = ApiBuilder::new().build()?;
        let api_repo = api.model(repo_id.to_string());

        let info = api_repo.info().await?;
        let mut downloaded = Vec::new();

        for sibling in &info.siblings {
            let name = &sibling.rfilename;
            if name.ends_with(".safetensors")
                || name.ends_with(".gguf")
                || name.ends_with(".bin")
                || name == "config.json"
                || name == "tokenizer.json"
                || name == "tokenizer.model"
                || name == "tokenizer_config.json"
                || name == "generation_config.json"
            {
                match api_repo.download(name).await {
                    Ok(path) => downloaded.push(path),
                    Err(e) => eprintln!("Warning: failed to download {name}: {e}"),
                }
            }
        }

        Ok(format!("Downloaded {} files for {repo_id}", downloaded.len()))
    }

    /// Delete a model from local cache
    pub fn delete_model(&self, repo_id: &str) -> Result<String> {
        let folder_name = Repo::new(repo_id.to_string(), RepoType::Model).folder_name();
        let model_cache_dir = self.cache.path().join(folder_name);

        if model_cache_dir.exists() {
            std::fs::remove_dir_all(&model_cache_dir)?;
            Ok(format!("Deleted {repo_id}"))
        } else {
            Ok(format!("{repo_id} not found in cache"))
        }
    }

    /// Scan the HF cache directory for all installed models
    pub fn list_installed_models(&self) -> Vec<ModelEntry> {
        let cache_path = self.cache.path().to_path_buf();
        if !cache_path.exists() {
            return Vec::new();
        }

        let mut entries = Vec::new();
        let Ok(dir_entries) = std::fs::read_dir(&cache_path) else {
            return Vec::new();
        };

        for entry in dir_entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.starts_with("models--") {
                continue;
            }

            // Convert cache folder name back to repo_id:
            // "models--google--gemma-2-2b-it" -> "google/gemma-2-2b-it"
            let repo_id = name
                .strip_prefix("models--")
                .unwrap_or(&name)
                .replace("--", "/");

            let size_mb = self.get_local_size(&repo_id);

            entries.push(ModelEntry {
                repo_id,
                status: ModelStatus::Downloaded,
                size_mb,
                description: "Installed (local cache)".to_string(),
                last_used: None,
            });
        }

        entries.sort_by(|a, b| a.repo_id.cmp(&b.repo_id));
        entries
    }

    /// Search HuggingFace Hub API for models matching a query
    pub async fn search_models(&self, query: &str) -> Result<Vec<SearchResult>> {
        let client = reqwest::Client::new();
        let resp = client
            .get("https://huggingface.co/api/models")
            .query(&[
                ("search", query),
                ("limit", "20"),
                ("sort", "downloads"),
                ("direction", "-1"),
            ])
            .send()
            .await?
            .json::<Vec<serde_json::Value>>()
            .await?;

        let results = resp
            .into_iter()
            .filter_map(|v| {
                let repo_id = v.get("id")?.as_str()?.to_string();
                let downloads = v
                    .get("downloads")
                    .and_then(|d| d.as_u64())
                    .unwrap_or(0);
                let likes = v.get("likes").and_then(|l| l.as_u64()).unwrap_or(0);
                let pipeline_tag = v
                    .get("pipeline_tag")
                    .and_then(|t| t.as_str())
                    .map(|s| s.to_string());

                Some(SearchResult {
                    repo_id,
                    downloads,
                    likes,
                    pipeline_tag,
                })
            })
            .collect();

        Ok(results)
    }

    /// Get the local path for a downloaded model
    #[allow(dead_code)]
    pub fn get_model_path(&self, repo_id: &str) -> Option<PathBuf> {
        let cache_repo = self.cache.model(repo_id.to_string());
        cache_repo.get("config.json")
    }
}
