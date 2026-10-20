"""
CLI entry point: ask a question against the ingested documents.

Runs the full pipeline (embed -> hybrid retrieve -> generate with citations
-> faithfulness check -> log) via src/pipeline.py, and prints the result.

Usage:
    python src/ask.py "What does this document say about remote work?"
    python src/ask.py "What does Section 4 say?" --doc vendor_contract_a.txt
"""

import sys

from pipeline import answer_question


def parse_args(argv: list[str]) -> tuple[str, list[str] | None]:
    if "--doc" in argv:
        idx = argv.index("--doc")
        doc_names = argv[idx + 1].split(",")
        question_parts = argv[:idx] + argv[idx + 2 :]
        return " ".join(question_parts), doc_names
    return " ".join(argv), None


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/ask.py "your question" [--doc file1.txt,file2.pdf]')
        sys.exit(1)

    question, sources = parse_args(sys.argv[1:])

    try:
        result = answer_question(question, sources=sources)
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    print("\n--- Answer ---")
    print(result["answer"])

    print("\n--- Sources ---")
    for i, meta in enumerate(result["metadatas"], 1):
        print(f"[{i}] {meta['source']}, page {meta['page']}")

    score = result["faithfulness_score"]
    print(f"\n--- Faithfulness: {score if score is not None else 'N/A'} ---")
    if result["unsupported_claims"]:
        print("Unsupported claims flagged by the checker:")
        for claim in result["unsupported_claims"]:
            print(f"  - {claim}")

    t = result["timing_ms"]
    tok = result["tokens"]
    print(
        f"\n--- Timing: embed {t['embed']}ms · retrieve {t['retrieve']}ms · "
        f"generate {t['generate']}ms · evaluate {t['evaluate']}ms · total {t['total']}ms ---"
    )
    print(f"--- Tokens: {tok['input']} in / {tok['output']} out ---")


if __name__ == "__main__":
    main()
