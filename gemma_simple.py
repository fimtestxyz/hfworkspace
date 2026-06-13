"""
Simple LLM program using HuggingFace Gemma-2 models
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import sys

def load_gemma_model(model_name="google/gemma-2-2b-it"):
    """
    Load a Gemma model from HuggingFace
    
    Args:
        model_name: The HuggingFace model identifier
                   Options include:
                   - google/gemma-2-2b-it (smaller, faster)
                   - google/gemma-2-9b-it (more capable)
                   - google/gemma-7b
                   - google/gemma-2b
    
    Returns:
        tokenizer and model
    """
    print(f"Loading model: {model_name}")
    print("This may take a few minutes on first run...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 for efficiency
        device_map="auto",  # Automatically use available GPU/CPU
    )
    
    return tokenizer, model


def generate_text_stream(prompt, tokenizer, model, max_length=512, temperature=0.7):
    """
    Generate text using the model with streaming output
    
    Args:
        prompt: Input text prompt
        tokenizer: Loaded tokenizer
        model: Loaded model
        max_length: Maximum length of generated text
        temperature: Sampling temperature (lower = more focused)
    
    Yields:
        Text chunks as they are generated
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Create streamer for real-time output
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=10.0
    )
    
    # Start generation in a separate thread
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_length,
        "temperature": temperature,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream output
    generated_text = ""
    for chunk in streamer:
        if chunk:
            generated_text += chunk
            yield chunk
    
    thread.join()


def generate_text(prompt, tokenizer, model, max_length=512, temperature=0.7):
    """
    Generate text using the model (non-streaming, waits for complete response)
    
    Args:
        prompt: Input text prompt
        tokenizer: Loaded tokenizer
        model: Loaded model
        max_length: Maximum length of generated text
        temperature: Sampling temperature (lower = more focused)
    
    Returns:
        Generated text
    """
    # Use streaming internally and combine chunks
    generated_chunks = []
    for chunk in generate_text_stream(prompt, tokenizer, model, max_length, temperature):
        generated_chunks.append(chunk)
    
    return ''.join(generated_chunks)


def interactive_chat():
    """
    Run an interactive chat session with Gemma (with streaming)
    """
    print("=" * 60)
    print("Simple LLM Chat with HuggingFace Gemma-2")
    print("(Streaming responses enabled)")
    print("=" * 60)
    print()
    
    # Load model
    tokenizer, model = load_gemma_model()
    
    print("\n✓ Model loaded successfully!")
    print("Type 'quit' or 'exit' to end the conversation.\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for exit
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! 👋")
            break
        
        if not user_input:
            continue
        
        # Format conversation prompt
        # Gemma-2 uses a turn-based format
        prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
        
        print("Gemma: ", end="", flush=True)
        
        # Stream response for real-time display
        try:
            for chunk in generate_text_stream(
                prompt, 
                tokenizer, 
                model, 
                max_length=256, 
                temperature=0.7
            ):
                print(chunk, end="", flush=True)
            
            print("\n")
        except KeyboardInterrupt:
            print("\n\nGeneration interrupted. Press Ctrl+C again to quit.\n")
            continue
        except Exception as e:
            print(f"\n\nError: {e}\n")


def single_prompt_example():
    """
    Run a single prompt example (with streaming)
    """
    print("Running single prompt example...\n")
    
    # Load model (use smaller model for quick demo)
    tokenizer, model = load_gemma_model("google/gemma-2-2b-it")
    
    # Example prompts
    prompts = [
        "Write a short poem about artificial intelligence.",
        "Explain quantum computing in simple terms.",
        "What are the benefits of renewable energy?",
    ]
    
    for i, prompt_text in enumerate(prompts, 1):
        print(f"\n{'='*60}")
        print(f"Prompt {i}: {prompt_text}")
        print(f"{'='*60}")
        
        # Format for Gemma-2
        formatted_prompt = f"<start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"
        
        print("\nResponse (streaming):")
        generated_text = ""
        for chunk in generate_text_stream(formatted_prompt, tokenizer, model, max_length=150):
            print(chunk, end="", flush=True)
            generated_text += chunk
        
        print("\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Run demo mode
        single_prompt_example()
    else:
        # Run interactive mode (default)
        interactive_chat()