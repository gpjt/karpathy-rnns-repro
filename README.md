# karpathy-repro

A reproduction of Andrej Karpathy's famous ["The Unreasonable Effectiveness of Recurrent Neural Networks"](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) blog post. Train character-level LSTM models on various text corpora and generate samples in the learned style.

## Quick Start

1. **Setup environment**: `uv sync` (or use pip with `pyproject.toml`)
2. **Train a model**: `uv run python train_rnn.py experiments/gilesthomas.com large-model`
3. **Monitor training**: Open `experiments/gilesthomas.com/runs/large-model/training_run.html` in browser
4. **Generate samples**: `uv run python generate_sample_text.py experiments/gilesthomas.com large-model best -p "<|article-start|>" -t 0.7`

## Environment Setup

### Using uv (recommended)
```bash
uv sync
```

### Using pip
```bash
pip install -e .
```

### Requirements
- Python 3.13+
- PyTorch 2.8+
- CUDA (optional, for GPU acceleration)

## Available Datasets

The project includes four text corpora for training:

### 1. gilesthomas.com (Ready to use)
- **Data**: Personal blog posts in markdown format (1.8MB)
- **Status**: ✅ Included in repository
- **Style**: Technical writing, programming tutorials

### 2. Shakespeare
- **Data**: Complete works of William Shakespeare
- **Status**: ❌ Manual download required
- **Setup**:
  1. Download from [Project Gutenberg](https://www.gutenberg.org/ebooks/100)
  2. Save as `experiments/shakespeare/data/input.txt`
  3. Remove Project Gutenberg metadata

### 3. War and Peace
- **Data**: Leo Tolstoy's novel
- **Status**: ❌ Manual download required
- **Setup**:
  1. Download from [Project Gutenberg](https://www.gutenberg.org/ebooks/2600)
  2. Save as `experiments/war-and-peace/data/input.txt`
  3. Remove Project Gutenberg metadata

### 4. Paul Graham Essays
- **Data**: All essays from paulgraham.com
- **Status**: ❌ Script download required
- **Setup**:
  ```bash
  cd experiments/paul-graham/data
  uv run python download.py
  ```

## Training Models

### Basic Training
```bash
uv run python train_rnn.py <experiment_directory> <run_name>
```

Examples:
```bash
# Train on blog data (ready to use)
uv run python train_rnn.py experiments/gilesthomas.com my-run

# Train on other datasets (after data setup)
uv run python train_rnn.py experiments/shakespeare karpathy-params-run
uv run python train_rnn.py experiments/paul-graham larger-model-as-more-data
```

### Configuration

Each run requires two JSON config files in `experiments/<dataset>/runs/<run_name>/`:

**train.json** - Training hyperparameters:
```json
{
  "seq_length": 100,
  "batch_size": 100,
  "val_batch_percent": 5,
  "optimizer": "Adam",
  "lr": 0.0003,
  "weight_decay": 0.0,
  "epochs": 10000,
  "max_grad_norm": 5.0,
  "patience": 5
}
```

**model.json** - Model architecture:
```json
{
  "hidden_size": 512,
  "num_layers": 3,
  "dropout": 0.5
}
```

### Monitoring Training

The system provides real-time visual monitoring:

1. **Start training** in one terminal
2. **Open monitoring page** in browser: `experiments/<dataset>/runs/<run_name>/training_run.html`
3. **Auto-refreshing chart** shows:
   - Training loss (blue line)
   - Validation loss (orange line)
   - Best epoch marker (red line)
   - Updates every second

Example monitoring workflow:
```bash
# Terminal 1: Start training
uv run python train_rnn.py experiments/gilesthomas.com my-run

# Terminal 2: Open monitoring (or use browser)
open experiments/gilesthomas.com/runs/my-run/training_run.html
```

## Generating Sample Text

### Basic Usage
```bash
uv run python generate_sample_text.py <experiment_directory> <run_name> <checkpoint> [options]
```

### Examples
```bash
# Generate with greedy sampling (most conservative)
uv run python generate_sample_text.py experiments/gilesthomas.com large-model best

# Generate 500 bytes with creative sampling
uv run python generate_sample_text.py experiments/shakespeare karpathy-params-run best -n 500 -t 0.8

# Start generation with specific primer text
uv run python generate_sample_text.py experiments/paul-graham larger-model-as-more-data best \
  -p "<|article-start|>" -t 0.7 -n 1000
```

### Options
- **`-n, --length`**: Number of bytes to generate (default: 100)
- **`-t, --temperature`**: Sampling creativity (default: 0.0)
  - `0.0` = Greedy (most predictable)
  - `0.1-0.5` = Conservative but varied
  - `0.8-1.2` = Creative and diverse
  - `>1.5` = Very experimental
- **`-p, --primer_text`**: Starting text (default: random character)

### Available Checkpoints
- **`best`** - Best validation loss checkpoint (recommended)
- **`latest`** - Most recent checkpoint
- **`epoch-N`** - Specific epoch (e.g., `epoch-42`)

### Recommended Primers by Dataset
- **gilesthomas.com**: `"<|article-start|>"` (generates new blog post)
- **paul-graham**: `"<|article-start|>"` (generates new essay)
- **shakespeare**: `"HAMLET:"` or `"Enter "` (generates dialogue/stage direction)
- **war-and-peace**: `"The "` or `"Prince "` (generates narrative)

## Project Structure

```
experiments/
├── <dataset>/
│   ├── data/
│   │   ├── input.txt          # Training text
│   │   └── SOURCE.md          # Data acquisition instructions
│   └── runs/
│       └── <run_name>/
│           ├── train.json     # Training config
│           ├── model.json     # Model config
│           ├── training_run.html  # Live monitoring page
│           ├── training_run.png   # Training loss chart
│           └── checkpoints/   # Saved model states
│               ├── best/      # Best validation checkpoint
│               ├── latest/    # Most recent checkpoint
│               └── epoch-*/   # Per-epoch checkpoints
```

## Tips

- **Start with gilesthomas.com** - data is included and ready to train
- **Monitor training** - open the HTML page for real-time loss tracking
- **Use early stopping** - set `patience` to stop when validation loss plateaus
- **Experiment with temperature** - higher values = more creative but less coherent text
- **GPU acceleration** - models automatically use CUDA if available
