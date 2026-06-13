"""
Color-coded LLM chat with HuggingFace Gemma-4 models + Web Search
Uses Catppuccin Mocha-inspired terminal colors for a beautiful TUI experience.
Integrated with WebSearch for real-time information retrieval.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import sys
import os
import json
import atexit
from datetime import datetime
import readline

import requests
from markdownify import markdownify as md
from ddgs import DDGS

# ─── Catppuccin Mocha ANSI Colors ──────────────────────────
class C:
    """Catppuccin Mocha palette as ANSI 256-color / true-color codes."""
    # Base
    ROSEWATER = "\033[38;5;217m"   # f5e0dc
    FLAMINGO  = "\033[38;5;210m"   # f2cdcd
    PINK      = "\033[38;5;212m"   # f5c2e7
    MAUVE     = "\033[38;5;183m"   # cba6f7
    RED       = "\033[38;5;203m"   # f38ba8
    MAROON    = "\033[38;5;181m"   # eba0ac
    PEACH     = "\033[38;5;215m"   # fab387
    YELLOW    = "\033[38;5;221m"   # f9e2af
    GREEN     = "\033[38;5;150m"   # a6e3a1
    TEAL      = "\033[38;5;115m"   # 94e2d5
    SKY       = "\033[38;5;153m"   # 89dceb
    SAPPHIRE  = "\033[38;5;116m"   # 74c7ec
    BLUE      = "\033[38;5;111m"   # 89b4fa
    LAVENDER  = "\033[38;5;189m"   # b4befe
    # Surface
    TEXT      = "\033[38;5;188m"   # cdd6f4
    SUBTEXT1  = "\033[38;5;252m"   # bac2de
    SUBTEXT0  = "\033[38;5;246m"   # a6adc8
    OVERLAY0  = "\033[38;5;240m"   # 6c7086
    SURFACE0  = "\033[38;5;59m"    # 313244
    SURFACE1  = "\033[38;5;60m"    # 45475a
    BASE      = "\033[38;5;17m"    # 1e1e2e
    MANTLE    = "\033[38;5;16m"    # 181825
    # Styles
    BOLD      = "\033[1m"
    DIM       = "\033[2m"
    ITALIC    = "\033[3m"
    UNDERLINE = "\033[4m"
    RESET     = "\033[0m"
    # Backgrounds
    BG_BASE   = "\033[48;5;17m"    # 1e1e2e
    BG_SURFACE0 = "\033[48;5;59m" # 313244
    BG_SURFACE1 = "\033[48;5;60m" # 45475a
    BG_MANTLE = "\033[48;5;16m"    # 181825


def supports_truecolor():
    """Check if the terminal supports true-color (24-bit)."""
    colorterm = os.environ.get("COLORTERM", "")
    term = os.environ.get("TERM", "")
    return "24bit" in colorterm or "truecolor" in colorterm or "kitty" in term


if supports_truecolor():
    # Override with true-color escape codes for richer display
    C.ROSEWATER = "\033[38;2;245;224;220m"
    C.FLAMINGO  = "\033[38;2;242;205;205m"
    C.PINK      = "\033[38;2;245;194;231m"
    C.MAUVE     = "\033[38;2;203;166;247m"
    C.RED       = "\033[38;2;243;139;168m"
    C.MAROON    = "\033[38;2;235;160;172m"
    C.PEACH     = "\033[38;2;250;179;135m"
    C.YELLOW    = "\033[38;2;249;226;175m"
    C.GREEN     = "\033[38;2;166;227;161m"
    C.TEAL      = "\033[38;2;148;226;213m"
    C.SKY       = "\033[38;2;137;220;235m"
    C.SAPPHIRE  = "\033[38;2;116;199;236m"
    C.BLUE      = "\033[38;2;137;180;250m"
    C.LAVENDER  = "\033[38;2;180;190;254m"
    C.TEXT      = "\033[38;2;205;214;244m"
    C.SUBTEXT1  = "\033[38;2;186;194;222m"
    C.SUBTEXT0  = "\033[38;2;166;173;200m"
    C.OVERLAY0  = "\033[38;2;108;112;134m"
    C.SURFACE0  = "\033[38;2;49;50;68m"
    C.BG_BASE   = "\033[48;2;30;30;46m"
    C.BG_SURFACE0 = "\033[48;2;49;50;68m"
    C.BG_SURFACE1 = "\033[48;2;69;71;90m"
    C.BG_MANTLE = "\033[48;2;24;24;37m"


# ─── Styled Print Helpers ──────────────────────────────────

def styled(text, *codes):
    """Wrap text in ANSI escape sequences."""
    return f"{''.join(codes)}{text}{C.RESET}"

def banner():
    """Print the application banner."""
    print()
    print(styled("  ╭───────────────────────────────────────────────────╮", C.MAUVE, C.BOLD))
    print(styled("  │", C.MAUVE) + styled("  🌐 ", C.SKY) + styled("Gemma-4 Search", C.BOLD, C.TEXT) + styled("  ·  ", C.SUBTEXT0) + styled("HuggingFace + Web", C.ITALIC, C.SUBTEXT0) + styled("  │", C.MAUVE))
    print(styled("  ╰───────────────────────────────────────────────────╯", C.MAUVE, C.BOLD))
    print()

def spinner_frame(frame, msg="Loading"):
    """Return a single spinner frame string."""
    icons = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    icon = icons[frame % len(icons)]
    return styled(f"  {icon} {msg}...", C.MAUVE)

def print_system(msg):
    """Print a system-level message (model loading, status, etc)."""
    print(f"{C.SURFACE0}  ┃{C.RESET} {C.OVERLAY0}{msg}{C.RESET}")

def print_user_msg(msg):
    """Print a stylist user message label."""
    print(f"\n{C.PEACH}  ▎ You{C.RESET} {C.PEACH}{C.BOLD}»{C.RESET} ", end="", flush=True)

def print_assistant_label():
    """Print the assistant label before streaming."""
    print(f"{C.GREEN}  ▎ Gemma{C.RESET} {C.GREEN}{C.BOLD}»{C.RESET} ", end="", flush=True)

def print_web_search_label():
    """Print a web search label."""
    print(f"{C.SKY}  ▎ Web Search{C.RESET} {C.SKY}{C.BOLD}»{C.RESET} ", end="", flush=True)

def print_streaming_chunk(chunk):
    """Print a streaming chunk in assistant color."""
    print(f"{C.LAVENDER}{chunk}{C.RESET}", end="", flush=True)

def print_error(msg):
    """Print an error message."""
    print(f"\n{C.RED}  ✖ {msg}{C.RESET}\n")

def print_info(msg):
    """Print an informational hint."""
    print(f"{C.SUBTEXT0}  ℹ {msg}{C.RESET}")

def print_success(msg):
    """Print a success message."""
    print(f"{C.GREEN}  ✓ {msg}{C.RESET}")

def print_divider():
    """Print a subtle divider line."""
    print(f"{C.SURFACE0}  ─────────────────────────────────────────────────{C.RESET}")

def print_help():
    """Print help/controls."""
    print(f"\n{C.SURFACE1}  ── Controls ───────────────────────────────────────{C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}Enter{C.RESET}    {C.SUBTEXT0}Send message{C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}↑ / ↓{C.RESET}    {C.SUBTEXT0}Browse previous commands (persistent){C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}Ctrl+C{C.RESET}   {C.SUBTEXT0}Interrupt generation or quit{C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}/quit{C.RESET}    {C.SUBTEXT0}Exit the chat{C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}/help{C.RESET}    {C.SUBTEXT0}Show this help{C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}/clear{C.RESET}   {C.SUBTEXT0}Clear conversation history{C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}/model{C.RESET}   {C.SUBTEXT0}Show current model/info{C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}/switch{C.RESET}  {C.SUBTEXT0}Switch to a different installed model{C.RESET}")
    print(f"\n{C.SURFACE1}  ── Search ────────────────────────────────────────{C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}/search <query>{C.RESET}  {C.SUBTEXT0}General web search{C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}/news <query>{C.RESET}    {C.SUBTEXT0}News search with timestamps{C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}/fetch <url>{C.RESET}     {C.SUBTEXT0}Fetch and read a web page{C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}/site <domain> <query>{C.RESET}  {C.SUBTEXT0}Search a specific site{C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}/history{C.RESET}         {C.SUBTEXT0}View search/fetch history{C.RESET}")
    print(f"\n{C.SURFACE1}  ── Site Shortcuts (trader research) ──────────────{C.RESET}")
    print(f"  {C.SKY}{C.BOLD}/x <query>{C.RESET}         {C.SUBTEXT0}Search X / Twitter — sentiment, breaking news{C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}/reddit <query>{C.RESET}    {C.SUBTEXT0}Search Reddit — WSB, r/stocks, discussions{C.RESET}")
    print(f"  {C.YELLOW}{C.BOLD}/hn <query>{C.RESET}        {C.SUBTEXT0}Search Hacker News — tech & startup analysis{C.RESET}")
    print(f"  {C.RED}{C.BOLD}/yt <query>{C.RESET}        {C.SUBTEXT0}Search YouTube — earnings calls, breakdowns{C.RESET}")
    print(f"  {C.GREEN}{C.BOLD}/stocktwits <query>{C.RESET}{C.SUBTEXT0}Search StockTwits — trader chatter{C.RESET}")
    print(f"  {C.MAUVE}{C.BOLD}/research <ticker>{C.RESET} {C.SUBTEXT0}Multi-site sweep: X + Reddit + HN + YT + News{C.RESET}")
    print()


# ─── Web Search Implementation ──────────────────────────────

class WebSearchManager:
    """Manages web search operations and history."""

    def __init__(self):
        self.search_history = []
        self.fetch_history = []

    def search(self, query, max_results=5):
        """
        Perform a web search query via DuckDuckGo.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of search results with title, url, and snippet
        """
        print_web_search_label()
        print(f"{C.SKY}Searching for: \"{query}\"{C.RESET}\n")

        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))
        except Exception as exc:
            print_error(f"Search failed: {exc}")
            return []

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]

        self.search_history.append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results": results
        })

        print_divider()
        if not results:
            print(f"\n{C.SUBTEXT0}  No results found{C.RESET}")
        for i, result in enumerate(results, 1):
            print(f"\n{C.BLUE}{i}. {result['title']}{C.RESET}")
            print(f"   {C.SKY}{result['url']}{C.RESET}")
            print(f"   {C.SUBTEXT1}{result['snippet']}{C.RESET}")
        print_divider()

        return results

    SITE_PRESETS = {
        "x":       ("x.com",              C.SKY,      "X / Twitter"),
        "twitter": ("x.com",              C.SKY,      "X / Twitter"),
        "reddit":  ("reddit.com",         C.PEACH,    "Reddit"),
        "hn":      ("news.ycombinator.com", C.YELLOW, "Hacker News"),
        "yt":      ("youtube.com",        C.RED,      "YouTube"),
        "stocktwits": ("stocktwits.com",  C.GREEN,    "StockTwits"),
    }

    def search_site(self, query, site, max_results=5, label=None, color=None):
        """Search a specific site via DuckDuckGo site: operator."""
        display_label = label or site
        display_color = color or C.SKY

        print_web_search_label()
        print(f"{display_color}{display_label}{C.RESET} {C.SUBTEXT0}»{C.RESET} {C.TEXT}{query}{C.RESET}\n")

        site_query = f"site:{site} {query}"
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(site_query, max_results=max_results))
        except Exception as exc:
            print_error(f"Search failed: {exc}")
            return []

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]

        self.search_history.append({
            "query": f"[{display_label}] {query}",
            "timestamp": datetime.now().isoformat(),
            "results": results,
        })

        print_divider()
        if not results:
            print(f"\n{C.SUBTEXT0}  No results found on {display_label}{C.RESET}")
        for i, result in enumerate(results, 1):
            print(f"\n{display_color}{i}. {result['title']}{C.RESET}")
            print(f"   {C.SKY}{result['url']}{C.RESET}")
            print(f"   {C.SUBTEXT1}{result['snippet']}{C.RESET}")
        print_divider()

        return results

    def search_news(self, query, max_results=5):
        """Search recent news articles via DuckDuckGo News."""
        print_web_search_label()
        print(f"{C.YELLOW}News{C.RESET} {C.SUBTEXT0}»{C.RESET} {C.TEXT}{query}{C.RESET}\n")

        try:
            with DDGS() as ddgs:
                raw = list(ddgs.news(query, max_results=max_results))
        except Exception as exc:
            print_error(f"News search failed: {exc}")
            return []

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("body", ""),
                "date": r.get("date", ""),
            }
            for r in raw
        ]

        self.search_history.append({
            "query": f"[News] {query}",
            "timestamp": datetime.now().isoformat(),
            "results": results,
        })

        print_divider()
        if not results:
            print(f"\n{C.SUBTEXT0}  No news results found{C.RESET}")
        for i, result in enumerate(results, 1):
            date_str = result["date"][:16].replace("T", " ") if result["date"] else ""
            print(f"\n{C.YELLOW}{i}. {result['title']}{C.RESET}")
            if date_str:
                print(f"   {C.DIM}{date_str}{C.RESET}")
            print(f"   {C.SKY}{result['url']}{C.RESET}")
            print(f"   {C.SUBTEXT1}{result['snippet']}{C.RESET}")
        print_divider()

        return results

    def research(self, topic, max_results_per_site=3):
        """Multi-site research sweep for stock/market analysis."""
        print()
        print(styled(f"  ╭────────────────────────────────────────────╮", C.MAUVE, C.BOLD))
        print(styled(f"  │", C.MAUVE) + styled("  📊 Research Sweep", C.BOLD, C.TEXT) + styled(f"  »  {topic}", C.PEACH) + styled("  │", C.MAUVE))
        print(styled(f"  ╰────────────────────────────────────────────╯", C.MAUVE, C.BOLD))
        print()

        sweep_sites = [
            ("x.com",              C.SKY,    "X / Twitter",    "market sentiment"),
            ("reddit.com",         C.PEACH,  "Reddit",         "community discussion"),
            ("news.ycombinator.com", C.YELLOW, "Hacker News", "tech analysis"),
            ("youtube.com",        C.RED,    "YouTube",        "earnings & breakdowns"),
        ]

        all_results = {}

        for site, color, label, desc in sweep_sites:
            results = self.search_site(
                topic, site,
                max_results=max_results_per_site,
                label=label, color=color,
            )
            all_results[label] = results
            print()

        print(styled("  📰 Latest News", C.YELLOW, C.BOLD))
        print()
        news_results = self.search_news(topic, max_results=max_results_per_site)
        all_results["News"] = news_results

        return all_results

    def fetch(self, url, max_chars=50000):
        """
        Fetch content from a URL and convert to markdown.

        Args:
            url: URL to fetch
            max_chars: Maximum characters to return (prevents memory issues on huge pages)

        Returns:
            Page content as markdown, or None on failure
        """
        print_web_search_label()
        print(f"{C.SKY}Fetching: {url}{C.RESET}\n")

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            resp = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0 (compatible; GemmaSearch/1.0)"},
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            print_error(f"Failed to fetch {url}: {exc}")
            return None

        content_type = resp.headers.get("Content-Type", "")
        if "html" not in content_type and "text" not in content_type:
            print_error(f"Unsupported content type: {content_type}")
            return None

        markdown = md(
            resp.text,
            heading_style="ATX",
            strip=["script", "style", "nav", "footer", "aside"],
            convert=["p", "h1", "h2", "h3", "h4", "h5", "h6", "a", "li", "ol", "ul",
                     "pre", "code", "blockquote", "strong", "em", "br", "hr", "table",
                     "thead", "tbody", "tr", "th", "td", "img"],
        )

        lines = [line for line in markdown.splitlines() if line.strip()]
        markdown = "\n".join(lines)

        if len(markdown) > max_chars:
            markdown = markdown[:max_chars] + "\n\n... (truncated)"

        self.fetch_history.append({
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "chars": len(markdown),
        })

        preview = markdown[:600] + ("..." if len(markdown) > 600 else "")
        print_divider()
        print(f"\n{C.TEXT}{preview}{C.RESET}\n")
        print(f"{C.SUBTEXT0}  {len(markdown):,} chars fetched{C.RESET}")
        print_divider()

        return markdown


    def format_search_context(self, results):
        """
        Format search results for inclusion in conversation context.

        Args:
            results: List of search result dictionaries

        Returns:
            Formatted string for the model
        """
        context_parts = ["Search results:"]
        for i, result in enumerate(results, 1):
            context_parts.append(f"\n{i}. {result['title']}")
            context_parts.append(f"   URL: {result['url']}")
            context_parts.append(f"   {result['snippet']}")
        return "\n".join(context_parts)

    def show_history(self, limit=10):
        """Display search and fetch history."""
        print(f"\n{C.BOLD}Search History{C.RESET}")
        print_divider()

        for entry in self.search_history[-limit:]:
            print(f"{C.SUBTEXT0}{entry['timestamp'][:19]}{C.RESET} {C.SKY}{entry['query']}{C.RESET}")
            print(f"   {C.SUBTEXT1}{len(entry['results'])} results{C.RESET}")
            print()

        if not self.search_history:
            print(f"{C.SUBTEXT0}No search history yet{C.RESET}\n")

        print(f"\n{C.BOLD}Fetch History{C.RESET}")
        print_divider()

        for entry in self.fetch_history[-limit:]:
            print(f"{C.SUBTEXT0}{entry['timestamp'][:19]}{C.RESET}")
            print(f"   {C.SKY}{entry['url']}{C.RESET}\n")

        if not self.fetch_history:
            print(f"{C.SUBTEXT0}No fetch history yet{C.RESET}\n")


# ─── Model Loading ─────────────────────────────────────────

def list_installed_models():
    """Scan the HuggingFace cache directory for installed models."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if not os.path.isdir(cache_dir):
        return []

    models = []
    for entry in sorted(os.listdir(cache_dir)):
        if entry.startswith("models--"):
            repo_id = entry[len("models--"):].replace("--", "/")
            # Estimate size from snapshots directory
            snap_dir = os.path.join(cache_dir, entry, "snapshots")
            size_mb = 0
            if os.path.isdir(snap_dir):
                for root, _dirs, files in os.walk(snap_dir):
                    for f in files:
                        fp = os.path.join(root, f)
                        try:
                            size_mb += os.path.getsize(fp)
                        except OSError:
                            pass
                size_mb = size_mb // (1024 * 1024)
            models.append({"repo_id": repo_id, "size_mb": size_mb})
    return models


def unload_model(model):
    """Free GPU/CPU memory by deleting the model and clearing CUDA cache."""
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_gemma_model(model_name="google/gemma-3-4b-it"):
    """
    Load a Gemma model from HuggingFace.

    Args:
        model_name: The HuggingFace model identifier

    Returns:
        (tokenizer, model) tuple
    """
    print_system(f"Initializing {model_name}")
    print()

    # Load tokenizer
    print(spinner_frame(0, "Loading tokenizer"), end="\r", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(styled(f"  ✓ Tokenizer loaded", C.GREEN), " " * 20)

    # Determine the best data type for stability and performance
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        dtype_str = "bfloat16 (CUDA)"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dtype = torch.bfloat16
        dtype_str = "bfloat16 (MPS)"
    else:
        dtype = torch.float32
        dtype_str = "float32 (CPU fallback)"

    print_system(f"Selected precision: {dtype_str}")

    # Load model
    print(spinner_frame(3, "Loading model (this may take a few minutes)"), end="\r", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    print(styled(f"  ✓ Model loaded to {model.device}", C.GREEN), " " * 20)

    # Device info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print_system(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        print_system("Running on CPU (consider GPU for faster inference)")

    return tokenizer, model


# ─── Generation ────────────────────────────────────────────

def generate_text_stream(prompt, tokenizer, model, max_length=512, temperature=0.7):
    """
    Generate text with streaming output.

    Args:
        prompt: Input text prompt (pre-formatted chat template string)
        tokenizer: Loaded tokenizer
        model: Loaded model
        max_length: Maximum new tokens to generate
        temperature: Sampling temperature

    Yields:
        Text chunks as they are generated
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Ensure input_ids are long type
    if inputs["input_ids"].dtype != torch.long:
        inputs["input_ids"] = inputs["input_ids"].long()

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=30.0,
    )

    use_sample = temperature > 0.01

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_length,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    if use_sample:
        generation_kwargs["temperature"] = max(temperature, 0.1)
        generation_kwargs["do_sample"] = True
        generation_kwargs["top_p"] = 0.9
        generation_kwargs["use_cache"] = True
    else:
        generation_kwargs["do_sample"] = False

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    try:
        for chunk in streamer:
            if chunk:
                yield chunk
    finally:
        thread.join(timeout=60)


# ─── Conversation Formatting ────────────────────────────────

def format_chat_prompt(messages, tokenizer, search_context=None):
    """
    Format conversation history using the tokenizer's native chat template.

    Args:
        messages: List of (role, content) tuples
        tokenizer: The loaded tokenizer
        search_context: Optional search context to include

    Returns:
        Formatted prompt string
    """
    # Use the tokenizer's built-in chat template
    try:
        chat_messages = [
            {"role": role, "content": content}
            for role, content in messages
        ]
        prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt
    except Exception:
        # Fallback: manual Gemma-2 style formatting
        prompt = ""
        for role, content in messages:
            if role == "user":
                prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif role == "assistant":
                prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"
        prompt += "<start_of_turn>model\n"
        return prompt


# ─── Input History ─────────────────────────────────────────

def setup_input_history():
    """Configure readline arrow-up/down history with cross-session persistence."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "gemma_search")
    os.makedirs(cache_dir, exist_ok=True)
    history_path = os.path.join(cache_dir, "input_history")

    readline.set_history_length(1000)
    readline.parse_and_bind("set disable-completion on")
    readline.parse_and_bind(r'"\e[A": previous-history')
    readline.parse_and_bind(r'"\e[B": next-history')
    readline.parse_and_bind(r'"\eOA": previous-history')
    readline.parse_and_bind(r'"\eOB": next-history')

    try:
        readline.read_history_file(history_path)
    except FileNotFoundError:
        pass

    def _save():
        try:
            readline.write_history_file(history_path)
        except OSError:
            pass

    atexit.register(_save)

    return history_path, _save


# ─── Interactive Chat ──────────────────────────────────────

def interactive_chat(model_name="google/gemma-3-4b-it"):
    """
    Run a color-coded interactive chat session with Gemma + Web Search.
    """
    # Initialize web search manager
    web_search = WebSearchManager()

    # Configure readline history (arrow up/down + persistence)
    _history_path, _save_history = setup_input_history()

    # Clear screen and show banner
    print("\033[2J\033[H", end="")
    banner()
    print_system(f"Model: {model_name}")
    print_system("Web Search: Enabled")
    print()

    # Load model
    tokenizer, model = load_gemma_model(model_name)

    print()
    print(styled("  ✓ Ready!", C.GREEN, C.BOLD))
    print_help()

    # Conversation history
    messages = []
    turn_count = 0
    last_search_results = None

    while True:
        try:
            print_user_msg("")
            user_input = input().strip()

            # Strip auto-added blank entries so arrow-up stays clean
            hist_len = readline.get_current_history_length()
            while hist_len > 0:
                entry = readline.get_history_item(hist_len)
                if entry is None or entry.strip() == "":
                    readline.remove_history_item(hist_len - 1)
                    hist_len -= 1
                else:
                    break

            if user_input:
                _save_history()

            # Slash commands
            if user_input.startswith("/"):
                if user_input.lower() in ["/quit", "/exit", "/q"]:
                    print()
                    print(styled("  👋 Goodbye!", C.MAUVE))
                    print()
                    break
                elif user_input.lower() == "/help":
                    print_help()
                    continue
                elif user_input.lower() == "/clear":
                    messages.clear()
                    turn_count = 0
                    last_search_results = None
                    print()
                    print(styled("  ✓ Conversation cleared", C.GREEN))
                    print()
                    continue
                elif user_input.lower() == "/model":
                    print()
                    print_system(f"Model: {model_name}")
                    print_system(f"Device: {model.device}")
                    print_system(f"Turns: {turn_count}")
                    print_system(f"Messages: {len(messages)}")
                    print_system(f"Searches: {len(web_search.search_history)}")
                    print()
                    continue
                elif user_input.lower().startswith("/switch"):
                    parts = user_input.split(None, 1)
                    target = parts[1].strip() if len(parts) > 1 else ""

                    installed = list_installed_models()
                    if not installed and not target:
                        print_error("No models found in HuggingFace cache")
                        print_info("Use the Rust TUI or 'huggingface-cli download' to install models")
                        continue

                    if target:
                        chosen = target
                    else:
                        print()
                        print(styled("  ── Installed Models ──────────────────────────────────", C.MAUVE))
                        current_idx = None
                        for i, m in enumerate(installed, 1):
                            marker = styled(" ● ", C.GREEN) if m["repo_id"] == model_name else styled("   ", C.SUBTEXT0)
                            size = f"{m['size_mb']} MB" if m['size_mb'] else "unknown"
                            print(f"  {marker}{C.PEACH}{i:2d}{C.RESET}  {C.TEXT}{m['repo_id']}{C.RESET}  {C.SUBTEXT0}({size}){C.RESET}")
                            if m["repo_id"] == model_name:
                                current_idx = i
                        print(styled("  ─────────────────────────────────────────────────────", C.MAUVE))
                        if current_idx:
                            print(f"  {C.SUBTEXT0}● = currently loaded{C.RESET}")
                        print()
                        print(f"  {C.TEXT}Enter number or repo ID to switch (Esc to cancel):{C.RESET} ", end="", flush=True)
                        choice = input().strip()
                        if not choice:
                            print_info("Switch cancelled")
                            continue
                        if choice.isdigit():
                            idx = int(choice) - 1
                            if 0 <= idx < len(installed):
                                chosen = installed[idx]["repo_id"]
                            else:
                                print_error(f"Invalid number: {choice}")
                                continue
                        else:
                            chosen = choice

                    if chosen == model_name:
                        print_info(f"Already using {chosen}")
                        continue

                    print()
                    print_system(f"Switching from {model_name} → {chosen}")
                    print_system("Unloading current model...")
                    unload_model(model)
                    tokenizer = None
                    model = None
                    print_system("Loading new model...")
                    print()

                    try:
                        tokenizer, model = load_gemma_model(chosen)
                        model_name = chosen
                        messages.clear()
                        turn_count = 0
                        last_search_results = None
                        print()
                        print(styled(f"  ✓ Switched to {model_name}", C.GREEN, C.BOLD))
                        print_info("Conversation cleared (new model context)")
                        print()
                    except Exception as e:
                        print_error(f"Failed to load {chosen}: {e}")
                        print_system("Attempting to reload previous model...")
                        try:
                            tokenizer, model = load_gemma_model(model_name)
                            print_success(f"Restored {model_name}")
                        except Exception as e2:
                            print_error(f"Failed to restore previous model: {e2}")
                            print_error("Session is in an invalid state. Restart recommended.")
                    continue
                elif user_input.lower() == "/history":
                    web_search.show_history()
                    continue
                elif user_input.lower().startswith("/search "):
                    query = user_input[8:].strip()
                    if query:
                        results = web_search.search(query)
                        last_search_results = results

                        # Ask if user wants to use these results
                        print(f"\n{C.TEXT}Use these results in the conversation? {C.PEACH}[Y/n]{C.RESET} ", end="", flush=True)
                        response = input().strip().lower()
                        if not response or response == 'y':
                            # Add search results to conversation
                            search_context = web_search.format_search_context(results)
                            messages.append(("user", f"Search question: {query}\n\n{search_context}"))
                            turn_count += 1
                            print_success("Added to conversation")
                        print()
                    else:
                        print_info("Usage: /search <query>")
                    continue
                elif user_input.lower().startswith("/fetch "):
                    url = user_input[7:].strip()
                    if url:
                        content = web_search.fetch(url)
                        if content is None:
                            print_error("Could not fetch that URL")
                            continue

                        # Ask if user wants to use this content
                        print(f"\n{C.TEXT}Use this content in the conversation? {C.PEACH}[Y/n]{C.RESET} ", end="", flush=True)
                        response = input().strip().lower()
                        if not response or response == 'y':
                            messages.append(("user", f"I read this content from {url}:\n\n{content}"))
                            turn_count += 1
                            print_success("Added to conversation")
                        print()
                    else:
                        print_info("Usage: /fetch <url>")
                    continue
                elif any(user_input.lower() == cmd or user_input.lower().startswith(cmd + " ")
                         for cmd in ("/x", "/reddit", "/hn", "/yt", "/stocktwits")):
                    site_cmd_map = {
                        "/x": "x", "/reddit": "reddit", "/hn": "hn",
                        "/yt": "yt", "/stocktwits": "stocktwits",
                    }
                    parts = user_input.split(None, 1)
                    cmd_lower = parts[0].lower()
                    preset_key = site_cmd_map.get(cmd_lower)
                    if preset_key and preset_key in WebSearchManager.SITE_PRESETS:
                        site, color, label = WebSearchManager.SITE_PRESETS[preset_key]
                        query = parts[1].strip() if len(parts) > 1 else ""
                        if query:
                            results = web_search.search_site(query, site, label=label, color=color)
                            last_search_results = results

                            print(f"\n{C.TEXT}Use these results in the conversation? {C.PEACH}[Y/n]{C.RESET} ", end="", flush=True)
                            response = input().strip().lower()
                            if not response or response == 'y':
                                search_context = web_search.format_search_context(results)
                                messages.append(("user", f"Search ({label}) for: {query}\n\n{search_context}"))
                                turn_count += 1
                                print_success("Added to conversation")
                            print()
                        else:
                            print_info(f"Usage: {cmd_lower} <query>")
                    else:
                        print_error(f"Unknown command: {parts[0]}")
                    continue
                elif user_input.lower().startswith("/news "):
                    query = user_input[6:].strip()
                    if query:
                        results = web_search.search_news(query)
                        last_search_results = results

                        print(f"\n{C.TEXT}Use these results in the conversation? {C.PEACH}[Y/n]{C.RESET} ", end="", flush=True)
                        response = input().strip().lower()
                        if not response or response == 'y':
                            search_context = web_search.format_search_context(results)
                            messages.append(("user", f"News search: {query}\n\n{search_context}"))
                            turn_count += 1
                            print_success("Added to conversation")
                        print()
                    else:
                        print_info("Usage: /news <query>")
                    continue
                elif user_input.lower().startswith("/site "):
                    rest = user_input[6:].strip()
                    space_idx = rest.find(" ")
                    if space_idx > 0:
                        domain = rest[:space_idx]
                        query = rest[space_idx+1:].strip()
                        results = web_search.search_site(query, domain)
                        last_search_results = results

                        print(f"\n{C.TEXT}Use these results in the conversation? {C.PEACH}[Y/n]{C.RESET} ", end="", flush=True)
                        response = input().strip().lower()
                        if not response or response == 'y':
                            search_context = web_search.format_search_context(results)
                            messages.append(("user", f"Search ({domain}) for: {query}\n\n{search_context}"))
                            turn_count += 1
                            print_success("Added to conversation")
                        print()
                    else:
                        print_info("Usage: /site <domain> <query>")
                    continue
                elif user_input.lower().startswith("/research "):
                    topic = user_input[10:].strip()
                    if topic:
                        all_results = web_search.research(topic)

                        print(f"\n{C.TEXT}Add research sweep to conversation? {C.PEACH}[Y/n]{C.RESET} ", end="", flush=True)
                        response = input().strip().lower()
                        if not response or response == 'y':
                            context_parts = [f"Research sweep for: {topic}\n"]
                            for site_label, results in all_results.items():
                                if results:
                                    context_parts.append(f"\n--- {site_label} ---")
                                    context_parts.append(web_search.format_search_context(results))
                            messages.append(("user", "\n".join(context_parts)))
                            turn_count += 1
                            print_success("Added to conversation")
                        print()
                    else:
                        print_info("Usage: /research <ticker or topic>")
                    continue
                else:
                    print_error(f"Unknown command: {user_input}")
                    print_info("Type /help for available commands")
                    continue

            # Skip empty input
            if not user_input:
                continue

            # Add to conversation
            messages.append(("user", user_input))
            turn_count += 1

            # Format prompt
            prompt = format_chat_prompt(messages, tokenizer)

            # Stream response
            print_assistant_label()
            full_response = ""

            try:
                for chunk in generate_text_stream(
                    prompt, tokenizer, model,
                    max_length=512, temperature=0.7
                ):
                    print_streaming_chunk(chunk)
                    full_response += chunk
                print(f"{C.RESET}\n")

                messages.append(("assistant", full_response.strip()))

            except KeyboardInterrupt:
                print(f"\n\n{C.YELLOW}  ⚠ Generation interrupted{C.RESET}\n")
                if full_response.strip():
                    messages.append(("assistant", full_response.strip()))

        except KeyboardInterrupt:
            print(f"\n\n{C.MAUVE}  👋 Goodbye!{C.RESET}\n")
            break
        except EOFError:
            print(f"\n\n{C.MAUVE}  👋 Goodbye!{C.RESET}\n")
            break
        except Exception as e:
            print_error(str(e))
            continue


# ─── Entry Point ────────────────────────────────────────────

if __name__ == "__main__":
    default_model = os.environ.get("GEMMA_MODEL", "google/gemma-3-4b-it")
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        selected_model = sys.argv[1]
    else:
        selected_model = default_model

    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        banner()
        print(f"  {C.TEXT}Usage:{C.RESET}")
        print(f"    {C.SAPPHIRE}python gemma_search.py{C.RESET}              {C.SUBTEXT0}Interactive chat with web search{C.RESET}")
        print(f"    {C.SAPPHIRE}GEMMA_MODEL=... python gemma_search.py{C.RESET}  {C.SUBTEXT0}Override model via env{C.RESET}")
        print()
        print(f"  {C.TEXT}Web Commands:{C.RESET}")
        print(f"    {C.SKY}/search <query>{C.RESET}     {C.SUBTEXT0}General web search{C.RESET}")
        print(f"    {C.YELLOW}/news <query>{C.RESET}       {C.SUBTEXT0}News search with timestamps{C.RESET}")
        print(f"    {C.SKY}/fetch <url>{C.RESET}        {C.SUBTEXT0}Fetch and read a webpage{C.RESET}")
        print(f"    {C.SKY}/site <domain> <q>{C.RESET}  {C.SUBTEXT0}Search a specific site{C.RESET}")
        print(f"    {C.SKY}/history{C.RESET}            {C.SUBTEXT0}View search/fetch history{C.RESET}")
        print()
        print(f"  {C.TEXT}Model:{C.RESET}")
        print(f"    {C.PEACH}/model{C.RESET}              {C.SUBTEXT0}Show current model info{C.RESET}")
        print(f"    {C.PEACH}/switch{C.RESET}             {C.SUBTEXT0}Switch to a different installed model{C.RESET}")
        print(f"    {C.PEACH}/switch <repo_id>{C.RESET}   {C.SUBTEXT0}Switch directly to a model{C.RESET}")
        print()
        print(f"  {C.TEXT}Trader Research:{C.RESET}")
        print(f"    {C.SKY}/x <query>{C.RESET}          {C.SUBTEXT0}Search X / Twitter{C.RESET}")
        print(f"    {C.PEACH}/reddit <query>{C.RESET}     {C.SUBTEXT0}Search Reddit{C.RESET}")
        print(f"    {C.YELLOW}/hn <query>{C.RESET}         {C.SUBTEXT0}Search Hacker News{C.RESET}")
        print(f"    {C.RED}/yt <query>{C.RESET}         {C.SUBTEXT0}Search YouTube{C.RESET}")
        print(f"    {C.GREEN}/stocktwits <query>{C.RESET} {C.SUBTEXT0}Search StockTwits{C.RESET}")
        print(f"    {C.MAUVE}/research <ticker>{C.RESET}  {C.SUBTEXT0}Multi-site sweep{C.RESET}")
        print()
        print(f"  {C.TEXT}Models:{C.RESET}")
        print(f"    {C.LAVENDER}google/gemma-3-4b-it{C.RESET}   {C.SUBTEXT0}Gemma 4, 4B instruct (recommended){C.RESET}")
        print(f"    {C.LAVENDER}google/gemma-2-2b-it{C.RESET}   {C.SUBTEXT0}Gemma 2, 2B instruct (fast){C.RESET}")
        print(f"    {C.LAVENDER}google/gemma-2-9b-it{C.RESET}   {C.SUBTEXT0}Gemma 2, 9B instruct (capable){C.RESET}")
        print()
    else:
        interactive_chat(selected_model)