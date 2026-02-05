# AGENTS.md

## Project Overview

BookNLP is a natural language processing pipeline designed to scale to books and other long documents in English. It provides a comprehensive set of NLP tasks including:

- Part-of-speech tagging
- Dependency parsing
- Entity recognition (people, locations, facilities, organizations, vehicles)
- Supersense tagging (semantic role annotations like "animal", "artifact", "body", "cognition")
- Event tagging

The project is built in Python and uses PyTorch, Hugging Face Transformers, and spaCy as core dependencies. It ships with two pre-trained models: a smaller model optimized for speed and a larger model optimized for accuracy.

### Key Technologies

- **Language**: Python 3.7+
- **ML Framework**: PyTorch (1.7.1+)
- **NLP Libraries**: Hugging Face Transformers (4.11.3+), spaCy (3+)
- **Package Manager**: pip/setuptools
- **Main Library**: booknlp

## Setup Commands

### Installation

```sh
# Create a Python 3.7+ environment (using conda is recommended)
conda create --name booknlp python=3.7
conda activate booknlp

# If using GPU, install PyTorch for your CUDA version:
# Visit https://pytorch.org and follow their installation instructions

# Install booknlp and dependencies
pip install booknlp

# Download required spaCy language model
python -m spacy download en_core_web_sm
```

### Development Installation

For development and contribution to the project:

```sh
# Clone the repository
git clone https://github.com/dbamman/book-nlp.git
cd booknlp

# Install in development mode
pip install -e .

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Development Workflow

### Project Structure

```
booknlp/
├── booknlp.py              # Main entry point and BookNLP class
├── common/                 # Shared utilities and core NLP components
│   ├── core.py            # Core data structures and BookNLPResult
│   ├── crf.py             # Conditional Random Field implementation
│   ├── layered_reader.py  # Text processing utilities
│   ├── logger.py          # Logging configuration
│   ├── pipelines.py       # Pipeline orchestration
│   ├── sequence_eval.py   # Evaluation metrics
│   └── sequence_layered_reader.py  # Sequence processing utilities
├── english/               # English-language specific modules
│   ├── english_booknlp.py # English pipeline implementation
│   ├── entity_tagger.py   # Entity recognition model
│   ├── tagger.py          # POS tagger and dependency parser
│   └── data/              # Data files and tagsets
└── data/                  # Sample data and pre-trained models
```

### Running the Pipeline

#### As a Python Library

```python
from booknlp.booknlp import BookNLP

# Configure the model
model_params = {
    "pipeline": "entity,supersense,event",  # Specify which tasks to run
    "model": "big"                           # Use "big" or "small"
}

# Initialize the pipeline
booknlp = BookNLP("en", model_params)

# Process a document
input_file = "input_dir/sample.txt"
output_directory = "output_dir/"
book_id = "sample"

result = booknlp.process(input_file, output_directory, book_id)
```

#### Via Command Line

```sh
python -m booknlp.booknlp \
    --language en \
    --inputFile input.txt \
    --outputFolder output_dir/ \
    --id my_document
```

#### Processing Options

- **pipeline**: Comma-separated list of tasks to run (e.g., "entity,event" to run only entity and event tagging)
- **model**: Choose between "big" (better accuracy, requires GPU/multi-core) or "small" (faster, suitable for personal computers)

### Understanding Model Parameters

The `model_params` dictionary accepts:
- `pipeline`: String specifying which NLP tasks to execute
- `model`: String specifying which pre-trained model to use

### Output Files

After processing, the following files are generated in the output directory:

- **`{book_id}.tokens`**: Word-level information with POS tags, lemmas, dependencies, and events
- **`{book_id}.entities`**: Typed entities with position, type (NOM/PROP/PRON), and category (PER/LOC/FAC/GPE/VEH/ORG)
- **`{book_id}.supersense`**: Semantic role annotations with start/end token positions and supersense categories

## Code Style Guidelines

### Python Conventions

Follow PEP 8 standards with these specific guidelines for this project:

- **Type Hints**: Use type hints for function parameters and return types (Python 3.7+)
- **Naming**: Use snake_case for functions/variables, PascalCase for classes
- **Documentation**: Add docstrings to public methods explaining parameters, return types, and behavior
- **Imports**: Organize imports alphabetically; use absolute imports from the booknlp package
- **Comments**: Focus on explaining *why* the code works a certain way, not *what* it does (code should be self-documenting)

### Module Organization

- Keep related functionality together in appropriate submodules (common/, english/, etc.)
- Use `__init__.py` to expose public APIs
- Avoid circular imports by organizing code hierarchically

### Configuration and Constants

- Store model paths and configuration in data/ directory
- Use configuration classes (like `EnglishBookNLPConfig`) to manage parameters
- Document non-obvious constants or magic numbers

## Building and Package Management

### Setup File

The project uses `setup.py` for packaging. Key configuration:

- **Package Name**: booknlp
- **Current Version**: 1.0.7
- **Python Version**: 3.7+ (inferred from requirements)
- **Core Dependencies**:
  - torch >= 1.7.1
  - spacy >= 3
  - transformers >= 4.11.3

### Building from Source

```sh
# Build distribution packages
python setup.py sdist bdist_wheel

# Install in development mode
pip install -e .
```

### Package Data

The `MANIFEST.in` file includes data files (tagsets, wordnet data, etc.) in the package distribution.

## Testing Instructions

Currently, there are no automated unit tests in the repository. For development and validation:

### Manual Testing

Test the pipeline with sample data:

```python
from booknlp.booknlp import BookNLP

# Test with small model for quick validation
model_params = {"pipeline": "entity", "model": "small"}
booknlp = BookNLP("en", model_params)

# Use provided sample data
booknlp.process(
    "booknlp/data/english/pride_and_prejudice.txt",
    "test_output/",
    "pride_test"
)
```

### Using Example Files

Sample notebooks and scripts are available in `examples/`:

- `run_booknlp.py`: Example script showing how to use the library
- `Read character file.ipynb`: Jupyter notebook demonstrating output parsing

### Validation Checklist

When adding new features:

- [ ] Test with both "small" and "big" model configurations
- [ ] Verify output file formats match expected schemas
- [ ] Test with documents of varying lengths
- [ ] Check performance on GPU and CPU environments
- [ ] Ensure backward compatibility with existing output format
- [ ] Validate new entities, supersenses, or events are correctly annotated

## Performance Considerations

### Model Selection

| Metric                  | Small Model | Big Model |
| ----------------------- | ----------- | --------- |
| Entity tagging (F1)     | 88.2        | 90.0      |
| Supersense tagging (F1) | 73.2        | 76.2      |
| Event tagging (F1)      | 70.6        | 74.1      |

Choose "small" for faster processing on CPU; use "big" on GPU for higher accuracy.

### Memory Requirements

- **Small model**: ~2-4 GB for typical books
- **Big model**: ~8-12 GB for typical books

### Processing Speed

- Small model processes ~1000 tokens/second on CPU
- Big model processes ~500 tokens/second on CPU
- GPU acceleration provides 5-10x speedup

## Common Patterns

### Processing Multiple Documents

```python
from booknlp.booknlp import BookNLP
import os

model_params = {"pipeline": "entity,supersense,event", "model": "small"}
booknlp = BookNLP("en", model_params)

input_dir = "books/"
output_dir = "processed_books/"

for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        book_id = filename[:-4]  # Remove .txt extension
        booknlp.process(
            os.path.join(input_dir, filename),
            output_dir,
            book_id
        )
```

### Working with Text Strings

```python
from booknlp.booknlp import BookNLP

model_params = {"pipeline": "entity", "model": "small"}
booknlp = BookNLP("en", model_params)

text = "Harry Potter walked through the castle."
result = booknlp.process(text=text, out_folder="output/", doc_id="harry")
```

## Dependencies and Troubleshooting

### GPU Setup

If you encounter GPU-related issues:

1. Verify PyTorch is installed for your CUDA version: `python -c "import torch; print(torch.cuda.is_available())"`
2. Check CUDA compatibility with your GPU
3. For CPU-only environments, the small model is recommended

### Missing spaCy Model

```sh
# Download if not already installed
python -m spacy download en_core_web_sm
```

### Dependency Version Conflicts

If installation fails due to version conflicts:

```sh
# Create a fresh environment
conda create --name booknlp python=3.7
conda activate booknlp

# Install PyTorch first (critical for other dependencies)
# Follow instructions at https://pytorch.org

# Then install booknlp
pip install booknlp
```

### Memory Issues

If processing large documents causes out-of-memory errors:

1. Use the "small" model instead of "big"
2. Consider splitting very long documents
3. Process documents on a machine with more available RAM

## Contributing

When making changes to the codebase:

- **Code Organization**: Follow the existing module structure
- **Type Safety**: Add type hints to new functions and methods
- **Documentation**: Update docstrings and AGENTS.md if behavior changes
- **Backward Compatibility**: Maintain existing output formats and API signatures
- **Testing**: Manually test with sample documents before committing

## Repository Structure Summary

- **booknlp/**: Main package containing all source code
  - **common/**: Core NLP utilities and pipeline orchestration
  - **english/**: English language-specific implementations
  - **data/**: Sample data and resource files
- **examples/**: Sample usage scripts and Jupyter notebooks
- **img/**: Documentation images
- **setup.py**: Package configuration and dependency specification
- **MANIFEST.in**: Include non-Python files in distribution

## Additional Resources

- GitHub Repository: https://github.com/dbamman/book-nlp
- Hugging Face Transformers: https://huggingface.co/transformers/
- spaCy Documentation: https://spacy.io/
- PyTorch: https://pytorch.org/
