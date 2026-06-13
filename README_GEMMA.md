# Simple LLM Program with HuggingFace Gemma-2

A minimal example of using Google's Gemma-2 language models through HuggingFace.

## Models Available

- `google/gemma-2-2b-it` - Small and fast (recommended for testing)
- `google/gemma-2-9b-it` - More capable, more resource-intensive
- `google/gemma-7b` - Original Gemma 7B
- `google/gemma-2b` - Original Gemma 2B

## Installation

The necessary dependencies are already in `pyproject.toml`:

```bash
cd /Volumes/wwk_nvme/Users/wwkoon/hfworkspace
uv sync  # or use your preferred environment manager
```

## Usage

### Interactive Chat Mode

```bash
# Activate your environment
source .venv/bin/activate

# Run the chat bot
python gemma_simple.py
```

### Demo Mode (Multiple Prompts)

```bash
python gemma_simple.py --demo
```

### Using the Script as a Module

```python
from gemma_simple import load_gemma_model, generate_text

# Load model
tokenizer, model = load_gemma_model("google/gemma-2-2b-it")

# Generate text
prompt = "<start_of_turn>user\nWrite a haiku about coding<end_of_turn>\n<start_of_turn>model\n"
response = generate_text(prompt, tokenizer, model, max_length=100)
print(response)
```

## Examples

### Basic Usage

```python
from gemma_simple import load_gemma_model, generate_text

# Load a small model
tokenizer, model = load_gemma_model()

# Ask questions
prompt = "<start_of_turn>user\nWhat is machine learning?<end_of_turn>\n<start_of_turn>model\n"
result = generate_text(prompt, tokenizer, model)
print(result)
```

### Text Completion

```python
prompt = """<start_of_turn>user
Complete this sentence: The future of AI is...
<end_of_turn>
<start_of_turn>model
"""

response = generate_text(prompt, tokenizer, model)
print(response)
```

### Code Generation

```python
prompt = """<start_of_turn>user
Write a Python function to calculate factorial:
<end_of_turn>
<start_of_turn>model
"""

code = generate_text(prompt, tokenizer, model)
print(code)
```

## Tips

1. **Start with the smaller model** `google/gemma-2-2b-it` for faster inference
2. **Adjust temperature**: Lower (0.1-0.3) for more focused answers, higher (0.7-1.0) for creativity
3. **Adjust max_length**: Shorter values for quick responses, longer for detailed answers
4. **Model format**: Gemma-2 uses `<start_of_turn>` and `<end_of_turn>` tokens for conversation turns
5. **Hardware**: Use GPU for faster inference; the script automatically detects and uses available GPU

## Requirements

- Python 3.13+
- PyTorch
- Transformers
- HuggingFace Hub

All specified in `pyproject.toml`.

## Notes

- First run will download the model (~3GB for gemma-2-2b-it)
- Ensure adequate disk space and memory
- HF_TOKEN environment variable may be required for some models