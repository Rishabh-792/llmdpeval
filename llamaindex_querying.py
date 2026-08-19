import argparse
import re
from pathlib import Path

from helper import (
    RETRIEVAL_TOP_K,
    SIMILARITY_THRESHOLD,
    RAGQueryPostprocessor,
    configure_models,
    load_prompt,
    persist_dir_for_embedding,
    print_generation_summary,
    print_prompt,
    print_retrieved_nodes,
    selected_embed_model,
    selected_llm_model,
)
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from weave_trace import (
    DEFAULT_WEAVE_PROJECT,
    create_generation_output_dir,
    init_weave_tracing,
    retrieved_node_trace_records,
    trace_code_generation,
)


QUERY_TYPE_PATTERN = re.compile(r"^Query type:\s*(.+)$", flags=re.MULTILINE)
QUERY_DEFINITION_PATTERN = re.compile(
    r"^Query definition:\s*(.+)$",
    flags=re.MULTILINE,
)


def build_retrieval_query(prompt: str) -> str:
    query_type = QUERY_TYPE_PATTERN.search(prompt)
    query_definition = QUERY_DEFINITION_PATTERN.search(prompt)
    if query_type is None or query_definition is None:
        raise ValueError("Prompt is missing Query type or Query definition.")
    return "\n".join(
        [
            f"Query type: {query_type.group(1)}",
            f"Query definition: {query_definition.group(1)}",
        ]
    )


def positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("--n must be at least 1")
    return parsed_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embed",
        default=None,
        help="Embedding model to use. Examples: nomic-embed-text, text-embedding-3-large.",
    )
    parser.add_argument(
        "--llm",
        default=None,
        help=(
            "LLM model to use. Examples: llama3.2, gpt-5.4, gpt-5.4-nano, "
            "gemini-3.1-flash-lite, claude-haiku-4-5, qwen2.5-coder:7b, "
            "codellama:7b."
        ),
    )
    parser.add_argument(
        "--task",
        default="stat_count",
        help="Prompt file to use from prompts/{task}.txt. Examples: stat_count, ml_pca, selection_topk.",
    )
    parser.add_argument(
        "--framework",
        default="OpenDP",
        choices=["OpenDP", "Diffprivlib", "PipelineDP"],
        help="DP framework to inject into requirement 4 of the selected prompt.",
    )
    parser.add_argument(
        "--rag",
        default="True",
        choices=["True", "False"],
        help="Use retrieval-augmented generation. Set to False for a direct LLM call.",
    )
    parser.add_argument(
        "--run-index",
        type=positive_int,
        default=1,
        help="Run number to use in the generated filename.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where generated code should be saved. If omitted, a new directory is created.",
    )
    parser.add_argument(
        "--weave",
        action="store_true",
        help="Trace the RAG retrieval/postprocessing step with Weave.",
    )
    parser.add_argument(
        "--weave-project",
        default=DEFAULT_WEAVE_PROJECT,
        help=f"Weave project name to use when --weave is set. Default: {DEFAULT_WEAVE_PROJECT}.",
    )
    return parser.parse_args()


def run_querying(args: argparse.Namespace) -> None:
    init_weave_tracing(args)

    llm_model_name = selected_llm_model(args)
    use_rag = args.rag == "True"
    embed_model_name = selected_embed_model(args) if use_rag else None
    configure_models(embed_model_name=embed_model_name, llm_model_name=llm_model_name)

    if use_rag:
        persist_dir = persist_dir_for_embedding(embed_model_name)
        print(f"Using persist directory: {persist_dir}")

    print(f"Using task: {args.task}")
    print(f"Using DP framework: {args.framework}")

    prompt = load_prompt(args.task, framework=args.framework)
    print_prompt(prompt)

    output_dir = Path(args.output_dir) if args.output_dir else create_generation_output_dir(
        llm_model=llm_model_name,
        rag_enabled=use_rag,
        framework=args.framework,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving generated code to: {output_dir}")

    if not use_rag:
        run_generation(
            embed_model_name=embed_model_name,
            llm_model_name=llm_model_name,
            use_rag=use_rag,
            task_name=args.task,
            prompt=prompt,
            output_dir=str(output_dir),
            retrieved_nodes=[],
            unique_retrieved_paths=[],
            final_prompt=prompt,
            run_index=args.run_index,
        )
        return

    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    print(f"Loaded existing index from {persist_dir} for querying")

    retriever = index.as_retriever(similarity_top_k=RETRIEVAL_TOP_K)
    retrieval_query = build_retrieval_query(prompt)
    print_prompt(retrieval_query, title="Retrieval query")
    retrieved_nodes = retriever.retrieve(retrieval_query)
    print_retrieved_nodes(retrieved_nodes)

    postprocessor = RAGQueryPostprocessor(
        prompt=prompt,
        candidate_nodes=retrieved_nodes,
        all_nodes=index.docstore.docs.values(),
        threshold=SIMILARITY_THRESHOLD,
        max_context_documents=1,
    )
    final_prompt = postprocessor.postprocess()

    print(f"\nFinal context nodes: {len(postprocessor.final_nodes)}")
    print_prompt(final_prompt)

    run_generation(
        embed_model_name=embed_model_name,
        llm_model_name=llm_model_name,
        use_rag=use_rag,
        task_name=args.task,
        prompt=prompt,
        output_dir=str(output_dir),
        retrieved_nodes=retrieved_node_trace_records(retrieved_nodes),
        unique_retrieved_paths=postprocessor.final_document_paths,
        final_prompt=final_prompt,
        run_index=args.run_index,
    )


def run_generation(
    *,
    embed_model_name: str | None,
    llm_model_name: str,
    use_rag: bool,
    task_name: str,
    prompt: str,
    output_dir: str,
    retrieved_nodes: list,
    unique_retrieved_paths: list[str],
    final_prompt: str,
    run_index: int,
) -> None:
    print(f"\nGeneration run {run_index}")
    generation = trace_code_generation(
        embed_model=embed_model_name,
        llm_model=llm_model_name,
        rag_enabled=use_rag,
        task_name=task_name,
        output_dir=output_dir,
        run_index=run_index,
        retrieved_nodes=retrieved_nodes,
        unique_retrieved_paths=unique_retrieved_paths,
        final_prompt=final_prompt,
    )
    # print_generation_summary(generation)


def main() -> None:
    run_querying(parse_args())


if __name__ == "__main__":
    main()
