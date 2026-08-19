# LLM DP Evaluation

This repository generates Python solutions for differential privacy tasks with optional RAG context, then evaluates the generated code by running each generated script.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For local models, make sure Ollama is running and the selected models are available. The default task runner uses:

- `nomic-embed-text` for embeddings
- `codellama:7b` for code generation

For hosted models, provide the matching API key file in the project root (create a file and enter the key in there):

- `openai_key.txt`
- `anthropic_key.txt`
- `gemini_key.txt`

## Build Or Load The RAG Index

When using RAG, build the LlamaIndex vector index before querying:

```bash
python llamaindex_indexing.py --embed nomic-embed-text
```

The index is saved under `storage/` according to the selected embedding model. If the index already exists, the script loads it instead of rebuilding it.

## Generate Code

Edit `run_tasks.sh` to adjust the run configuration:

- `RAG`: use retrieval augmented generation (`True` or `False`)
- `EMBED_MODEL`: embedding model used when `RAG=True`
- `LLM_MODEL`: model used to generate code
- `FRAMEWORK`: DP framework to request in the prompt (`OpenDP`, `Diffprivlib`, or `PipelineDP`)
- `RUN_COUNT`: number of generations per task
- `tasks`: task prompt names from `prompts/*.txt`

Then run:

```bash
./run_tasks.sh
```

The script saves generated Python files under:

```text
generated_code/<model>_<rag>_<framework>_<timestamp>/
```

For example:

```text
generated_code/codellama-7b_rag_OpenDP_1785277012414351176/
```

Keep the timestamp from the folder name. You will use it for evaluation.

## Evaluate Generated Code

Run the evaluator with a timestamp substring that uniquely matches one generated folder:

```bash
python code_evaluation.py --ts 1785277012414351176
```

The evaluator runs each `.py` file in the matching `generated_code` folder and writes a JSON log to `evaluation_logs/`. The console summary reports how many scripts were runnable versus failed. The JSON log includes each script path, return code, stdout, and stderr so failures and errors can be inspected.

## Important Files

### `llamaindex_indexing.py`

Builds or loads the RAG index. It preprocesses text files in `rag_corpus/`, converts document sections into LlamaIndex nodes, creates a `VectorStoreIndex`, and persists it under the embedding-specific directory in `storage/`.

### `llamaindex_querying.py`

Generates code for one task. It loads a prompt from `prompts/{task}.txt`, injects the requested DP framework, optionally retrieves relevant documentation from the saved LlamaIndex index, and sends the final prompt to the selected LLM. Generated code is saved to the requested output directory.

### `run_tasks.sh`

Batch runner for code generation. It sets default model/framework parameters, defines the list of tasks to run, creates one timestamped output folder in `generated_code/`, and calls `llamaindex_querying.py` for each task and run index.

### `code_evaluation.py`

Evaluates a timestamped generated-code folder. It finds the folder in `generated_code/`, runs each generated Python file with a timeout, prints a runnable/failed summary, and saves detailed execution logs under `evaluation_logs/`.

### Supporting Files

- `helper.py`: shared model configuration, prompt loading, RAG corpus preprocessing, retrieval printing, and context postprocessing utilities.
- `metadata.py`: extracts document sections and code blocks from the RAG text corpus.
- `weave_trace.py`: wraps generation and optional evaluation helpers with Weave tracing, and handles generated code file creation.
- `prompts/`: task prompts used for code generation.
- `rag_corpus/`: documentation text used as retrieval context.
