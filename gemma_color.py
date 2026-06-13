"""
Color-coded LLM chat with HuggingFace Gemma-4 models
Uses Catppuccin Mocha-inspired terminal colors for a beautiful TUI experience.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import sys
import os

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
    print(styled("  ╭─────────────────────────────────────────────╮", C.MAUVE, C.BOLD))
    print(styled("  │", C.MAUVE) + styled("  🤖 ", C.PEACH) + styled("Gemma-4 Chat  ", C.BOLD, C.TEXT) + styled("·  HuggingFace", C.ITALIC, C.SUBTEXT0) + styled("  │", C.MAUVE))
    print(styled("  ╰─────────────────────────────────────────────╯", C.MAUVE, C.BOLD))
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

def print_streaming_chunk(chunk):
    """Print a streaming chunk in assistant color."""
    print(f"{C.LAVENDER}{chunk}{C.RESET}", end="", flush=True)

def print_error(msg):
    """Print an error message."""
    print(f"\n{C.RED}  ✖ {msg}{C.RESET}\n")

def print_info(msg):
    """Print an informational hint."""
    print(f"{C.SUBTEXT0}  ℹ {msg}{C.RESET}")

def print_divider():
    """Print a subtle divider line."""
    print(f"{C.SURFACE0}  ─────────────────────────────────────────{C.RESET}")

def print_help():
    """Print help/controls."""
    print(f"\n{C.SURFACE1}  ── Controls ──────────────────────────────{C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}Enter{C.RESET}    {C.SUBTEXT0}Send message{C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}Ctrl+C{C.RESET}   {C.SUBTEXT0}Interrupt generation or quit{C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}/quit{C.RESET}    {C.SUBTEXT0}Exit the chat{C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}/help{C.RESET}    {C.SUBTEXT0}Show this help{C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}/clear{C.RESET}   {C.SUBTEXT0}Clear conversation history{C.RESET}")
    print(f"  {C.PEACH}{C.BOLD}/model{C.RESET}   {C.SUBTEXT0}Show current model info{C.RESET}")
    print()

# ─── Model Loading ─────────────────────────────────────────

def load_gemma_model(model_name="google/gemma-3-4b-it"):
    """
    Load a Gemma model from HuggingFace.

    Args:
        model_name: The HuggingFace model identifier
                   Options include:
                   - google/gemma-3-4b-it   (Gemma 4, recommended)
                   - google/gemma-2-2b-it   (smaller, faster)
                   - google/gemma-2-9b-it   (more capable)

    Returns:
        (tokenizer, model) tuple
    """
    print_system(f"Initializing {model_name}")
    print()

    # Load tokenizer
    print(spinner_frame(0, "Loading tokenizer"), end="\r", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(styled(f"  ✓ Tokenizer loaded", C.GREEN), " " * 20)

    # Determine the best data type for stability and performance.
    # Gemma-2 and Gemma-3 are numerically unstable in float16 due to logit soft-capping
    # (resulting in NaNs). bfloat16 is native and highly recommended.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        dtype_str = "bfloat16 (CUDA)"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon supports bfloat16 in macOS 14+ / PyTorch 2.x
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

    # Ensure input_ids are long type (required by some models)
    if inputs["input_ids"].dtype != torch.long:
        inputs["input_ids"] = inputs["input_ids"].long()

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=30.0,
    )

    # Use greedy decoding at very low temp to avoid multinomial NaN issues
    # on CPU or when probs contain NaN/Inf values
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
        # Clamp logits to prevent NaN in softmax/multinomial
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


def generate_text(prompt, tokenizer, model, max_length=512, temperature=0.7):
    """Generate text (non-streaming, waits for complete response)."""
    chunks = []
    for chunk in generate_text_stream(prompt, tokenizer, model, max_length, temperature):
        chunks.append(chunk)
    return "".join(chunks)


# ─── Conversation Formatting ────────────────────────────────

def format_chat_prompt(messages, tokenizer):
    """
    Format conversation history using the tokenizer's native chat template.

    Falls back to manual Gemma formatting if apply_chat_template is unavailable.

    Args:
        messages: List of (role, content) tuples
        tokenizer: The loaded tokenizer (used for apply_chat_template)

    Returns:
        Formatted prompt string
    """
    # Use the tokenizer's built-in chat template (handles Gemma-3/Gemma-4 correctly)
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


# ─── Interactive Chat ──────────────────────────────────────

def interactive_chat(model_name="google/gemma-3-4b-it"):
    """
    Run a color-coded interactive chat session with Gemma.
    """
    # Clear screen and show banner
    print("\033[2J\033[H", end="")  # Clear screen
    banner()
    print_system(f"Model: {model_name}")
    print()

    # Load model
    tokenizer, model = load_gemma_model(model_name)

    print()
    print(styled("  ✓ Ready!", C.GREEN, C.BOLD))
    print_help()

    # Conversation history for multi-turn
    messages = []
    turn_count = 0

    while True:
        try:
            # User input with styled prompt
            print_user_msg("")
            user_input = input().strip()

            # Slash commands
            if user_input.startswith("/"):
                cmd = user_input.lower()
                if cmd in ["/quit", "/exit", "/q"]:
                    print()
                    print(styled("  👋 Goodbye!", C.MAUVE))
                    print()
                    break
                elif cmd == "/help":
                    print_help()
                    continue
                elif cmd == "/clear":
                    messages.clear()
                    turn_count = 0
                    print()
                    print(styled("  ✓ Conversation cleared", C.GREEN))
                    print()
                    continue
                elif cmd == "/model":
                    print()
                    print_system(f"Model: {model_name}")
                    print_system(f"Device: {model.device}")
                    print_system(f"Turns: {turn_count}")
                    print_system(f"Messages in history: {len(messages)}")
                    print()
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

                # Store assistant response for multi-turn context
                messages.append(("assistant", full_response.strip()))

            except KeyboardInterrupt:
                print(f"\n\n{C.YELLOW}  ⚠ Generation interrupted{C.RESET}\n")
                # Still store partial response
                if full_response.strip():
                    messages.append(("assistant", full_response.strip()))

        except KeyboardInterrupt:
            # Double Ctrl+C exits
            print(f"\n\n{C.MAUVE}  👋 Goodbye!{C.RESET}\n")
            break
        except EOFError:
            print(f"\n\n{C.MAUVE}  👋 Goodbye!{C.RESET}\n")
            break
        except Exception as e:
            print_error(str(e))
            continue


# ─── Single Demo Mode ──────────────────────────────────────

def single_prompt_example(model_name="google/gemma-3-4b-it"):
    """Run preset prompts with color-coded output."""
    banner()
    print_system(f"Model: {model_name}")
    print()

    tokenizer, model = load_gemma_model(model_name)
    print()

    prompts = [
        "Write a short poem about artificial intelligence.",
        "Explain quantum computing in simple terms.",
        "What are the benefits of renewable energy?",
    ]

    for i, prompt_text in enumerate(prompts, 1):
        print_divider()
        print(f"\n  {C.PEACH}{C.BOLD}Prompt {i}:{C.RESET} {C.TEXT}{prompt_text}{C.RESET}\n")
        print_assistant_label()

        formatted = format_chat_prompt([("user", prompt_text)], tokenizer)

        for chunk in generate_text_stream(formatted, tokenizer, model, max_length=200):
            print_streaming_chunk(chunk)
        print(f"{C.RESET}\n")

    print_divider()
    print()


# ─── Entry Point ────────────────────────────────────────────

if __name__ == "__main__":
    # Select model (override with env var or argument)
    default_model = os.environ.get("GEMMA_MODEL", "google/gemma-3-4b-it")
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        selected_model = sys.argv[1]
    else:
        selected_model = default_model

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        single_prompt_example(selected_model)
    elif len(sys.argv) > 1 and sys.argv[1] == "--help":
        banner()
        print(f"  {C.TEXT}Usage:{C.RESET}")
        print(f"    {C.SAPPHIRE}python gemma_color.py{C.RESET}              {C.SUBTEXT0}Interactive chat{C.RESET}")
        print(f"    {C.SAPPHIRE}python gemma_color.py --demo{C.RESET}      {C.SUBTEXT0}Run preset prompts{C.RESET}")
        print(f"    {C.SAPPHIRE}python gemma_color.py <model_id>{C.RESET}  {C.SUBTEXT0}Use specific model{C.RESET}")
        print(f"    {C.SAPPHIRE}GEMMA_MODEL=... python gemma_color.py{C.RESET}  {C.SUBTEXT0}Override via env{C.RESET}")
        print()
        print(f"  {C.TEXT}Models:{C.RESET}")
        print(f"    {C.LAVENDER}google/gemma-3-4b-it{C.RESET}   {C.SUBTEXT0}Gemma 4, 4B instruct (recommended){C.RESET}")
        print(f"    {C.LAVENDER}google/gemma-2-2b-it{C.RESET}   {C.SUBTEXT0}Gemma 2, 2B instruct (fast){C.RESET}")
        print(f"    {C.LAVENDER}google/gemma-2-9b-it{C.RESET}   {C.SUBTEXT0}Gemma 2, 9B instruct (capable){C.RESET}")
        print()
    else:
        interactive_chat(selected_model)
