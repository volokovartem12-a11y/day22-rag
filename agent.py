"""
День 22. Агент с двумя режимами: с RAG и без RAG.

    python agent.py                                  # диалог, режим по умолчанию — rag
    python agent.py --ask "Какие слои памяти в дне 11?" --mode both

Команды в диалоге:
    /rag      — отвечать с RAG (по базе документов)
    /plain    — отвечать без RAG (только знания модели)
    /both     — показывать оба ответа рядом
    /sources  — показать найденные фрагменты последнего RAG-ответа
    /exit
"""

import argparse
import sys

from rag import LLM, ask_plain, ask_rag, load_index


def show(result: dict) -> None:
    title = "С RAG" if result["mode"] == "rag" else "Без RAG"
    print(f"\n── {title} ──\n{result['answer']}")
    if result["mode"] == "rag":
        print(f"\nИсточники: {', '.join(result['cited']) or 'модель не сослалась на фрагменты'}")


def answer(llm, index, question: str, mode: str, k: int) -> dict:
    last = None
    if mode in ("plain", "both"):
        show(ask_plain(llm, question))
    if mode in ("rag", "both"):
        last = ask_rag(llm, index, question, k)
        show(last)
    return last


def main(args) -> None:
    llm = LLM(args.model)
    index = load_index(args.backend, "structure")
    print(f"Модель: {llm.model} · база: {index.meta['chunks']} чанков ({index.meta['embedder']['label']})")

    if args.ask:
        answer(llm, index, args.ask, args.mode, args.k)
        return

    mode, last = args.mode, None
    print("Команды: /rag, /plain, /both, /sources, /exit\n")
    while True:
        try:
            q = input(f"[{mode}] Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q == "/exit":
            break
        if q in ("/rag", "/plain", "/both"):
            mode = q[1:]
            print(f"Режим: {mode}")
            continue
        if q == "/sources":
            if not last:
                print("Пока не было RAG-ответов.")
            for c in (last or {}).get("chunks", []):
                print(f"  [{c['n']}] {c['source']} — {c['section']} (score {c['score']})")
            continue
        last = answer(llm, index, q, mode, args.k) or last
        print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Агент с режимами RAG / без RAG")
    ap.add_argument("--ask", help="задать один вопрос и выйти")
    ap.add_argument("--mode", choices=["rag", "plain", "both"], default="rag")
    ap.add_argument("--model", help="модель AITunnel")
    ap.add_argument("--backend", choices=["aitunnel", "local"], default="aitunnel")
    ap.add_argument("--k", type=int, default=4)
    main(ap.parse_args())
