"""
Quick test script for Gemma-2 model with streaming
Runs a single prompt to verify streaming works
"""

from gemma_simple import load_gemma_model, generate_text_stream


def test_streaming():
    """Run a streaming test to verify the model works"""
    print("🧪 Testing Gemma-2 model with streaming...\n")
    
    # Load the small model
    print("Loading model...")
    tokenizer, model = load_gemma_model("google/gemma-2-2b-it")
    print("✓ Model loaded!\n")
    
    # Test prompt
    test_prompt = "Write a short poem about the ocean."
    formatted_prompt = f"<start_of_turn>user\n{test_prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    print(f"Prompt: {test_prompt}\n")
    print("Streaming response:")
    print("-" * 60)
    
    # Stream the response token by token
    full_response = ""
    for chunk in generate_text_stream(
        formatted_prompt,
        tokenizer,
        model,
        max_length=100,
        temperature=0.8
    ):
        print(chunk, end="", flush=True)
        full_response += chunk
    
    print()
    print("-" * 60)
    print(f"\n✓ Streaming test completed!")
    print(f"  Total characters streamed: {len(full_response)}")


if __name__ == "__main__":
    test_streaming()
