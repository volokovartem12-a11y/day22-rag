"""
День 22. Сравнение ответов без RAG и с RAG на 10 контрольных вопросах.

Для каждого вопроса и режима считается:
  факты     — доля обязательных фактов из eval_questions.json, найденных в ответе;
  статус    — верно (все факты) / частично / честно не знает / неверно;
  источники — (только RAG) нашёл ли поиск нужный файл и сослалась ли на него модель;
  токены    — сколько стоил ответ.

Запуск:
    python evaluate.py                        # модель по умолчанию, эмбеддинги AITunnel
    python evaluate.py --model gpt-5.6-luna
    python evaluate.py --backend local        # поиск на локальных эмбеддингах
"""

import argparse
import json
import os
import sys
from datetime import datetime

from rag import BASE, LLM, ask_plain, ask_rag, is_refusal, load_index


def load_questions() -> list:
    with open(os.path.join(BASE, "eval_questions.json"), encoding="utf-8") as f:
        return json.load(f)["questions"]


def fact_check(answer: str, must: list) -> tuple:
    text = answer.lower().replace("\u00a0", " ")
    matched = [any(v.lower() in text for v in group) for group in must]
    return sum(matched) / len(must), [g[0] for g, ok in zip(must, matched) if not ok]


def status(coverage: float, answer: str) -> str:
    if coverage == 1:
        return "верно"
    if is_refusal(answer):
        return "честно не знает"
    return "частично" if coverage > 0 else "неверно"


def source_match(paths: list, expected: list):
    if not expected:
        return None  # источник не обязателен
    return any(e in p for p in paths for e in expected)


def score(result: dict, q: dict) -> dict:
    cov, missing = fact_check(result["answer"], q["must"])
    out = {**result, "coverage": round(cov, 2), "missing": missing, "status": status(cov, result["answer"])}
    if result["mode"] == "rag":
        out["retrieved_ok"] = source_match([c["source"] for c in result["chunks"]], q["sources"])
        out["cited_ok"] = source_match(result["cited"], q["sources"])
    return out


def summarize(rows: list, mode: str) -> dict:
    rs = [r[mode] for r in rows]
    by_type = {t: [r[mode]["coverage"] for r in rows if r["type"] == t] for t in ("project", "general")}
    s = {
        "avg_coverage": round(sum(r["coverage"] for r in rs) / len(rs), 2),
        "project_coverage": round(sum(by_type["project"]) / len(by_type["project"]), 2) if by_type["project"] else None,
        "general_coverage": round(sum(by_type["general"]) / len(by_type["general"]), 2) if by_type["general"] else None,
        "statuses": {st: sum(r["status"] == st for r in rs) for st in ("верно", "частично", "честно не знает", "неверно")},
        "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in rs),
        "completion_tokens": sum(r["usage"]["completion_tokens"] for r in rs),
    }
    if mode == "rag":
        with_src = [r for r in rs if r["retrieved_ok"] is not None]
        s["retrieved_ok"] = f"{sum(r['retrieved_ok'] for r in with_src)}/{len(with_src)}"
        s["cited_ok"] = f"{sum(bool(r['cited_ok']) for r in with_src)}/{len(with_src)}"
    return s


def mark(v) -> str:
    return "—" if v is None else ("✅" if v else "❌")


def render(rows: list, summ: dict, meta: dict) -> str:
    p, r = summ["plain"], summ["rag"]
    head = (
        f"# Сравнение: ответы без RAG и с RAG\n\n"
        f"Модель: `{meta['model']}` · поиск: `{meta['index']}`, топ-{meta['k']} чанков · {meta['date']}\n\n"
        f"## Итог\n\n"
        f"| Метрика | Без RAG | С RAG |\n|---|---|---|\n"
        f"| Средняя доля фактов (все 10) | {p['avg_coverage']:.0%} | **{r['avg_coverage']:.0%}** |\n"
        f"| Вопросы по проектам (8) | {p['project_coverage']:.0%} | **{r['project_coverage']:.0%}** |\n"
        f"| Общие вопросы (2) | {p['general_coverage']:.0%} | {r['general_coverage']:.0%} |\n"
        f"| Верно (все факты) | {p['statuses']['верно']} | **{r['statuses']['верно']}** |\n"
        f"| Частично | {p['statuses']['частично']} | {r['statuses']['частично']} |\n"
        f"| Честно не знает | {p['statuses']['честно не знает']} | {r['statuses']['честно не знает']} |\n"
        f"| Неверно | {p['statuses']['неверно']} | {r['statuses']['неверно']} |\n"
        f"| Поиск нашёл нужный источник | — | {r['retrieved_ok']} |\n"
        f"| Модель сослалась на нужный источник | — | {r['cited_ok']} |\n"
        f"| Токены: вход / выход | {p['prompt_tokens']} / {p['completion_tokens']} | "
        f"{r['prompt_tokens']} / {r['completion_tokens']} |\n\n"
        f"## По вопросам\n\n| # | Тип | Без RAG | С RAG | Источник найден | Источник процитирован |\n"
        f"|---|---|---|---|---|---|\n"
    )
    table = "\n".join(
        f"| {x['id']} | {x['type']} | {x['plain']['status']} ({x['plain']['coverage']:.0%}) | "
        f"{x['rag']['status']} ({x['rag']['coverage']:.0%}) | {mark(x['rag']['retrieved_ok'])} | "
        f"{mark(x['rag']['cited_ok'])} |" for x in rows)
    details = []
    for x in rows:
        chunks = "\n".join(f"  - [{c['n']}] `{c['source']}` — {c['section']} (score {c['score']})"
                           for c in x["rag"]["chunks"])
        details.append(
            f"### {x['id']}. {x['q']}\n\n"
            f"**Ожидание:** {x['expected']}  \n"
            f"**Ожидаемые источники:** {', '.join(f'`{s}`' for s in x['sources']) or 'не обязательны'}\n\n"
            f"**Без RAG** — {x['plain']['status']}, факты {x['plain']['coverage']:.0%}"
            f"{' (не хватает: ' + ', '.join(x['plain']['missing']) + ')' if x['plain']['missing'] else ''}\n\n"
            f"> {x['plain']['answer'].replace(chr(10), chr(10) + '> ')}\n\n"
            f"**С RAG** — {x['rag']['status']}, факты {x['rag']['coverage']:.0%}"
            f"{' (не хватает: ' + ', '.join(x['rag']['missing']) + ')' if x['rag']['missing'] else ''}\n\n"
            f"> {x['rag']['answer'].replace(chr(10), chr(10) + '> ')}\n\n"
            f"Найденные фрагменты:\n{chunks}\n\n"
            f"Процитированы: {', '.join(f'`{s}`' for s in x['rag']['cited']) or 'нет'}\n")
    return head + table + "\n\n## Ответы целиком\n\n" + "\n".join(details)


def main(args) -> int:
    questions = load_questions()
    llm = LLM(args.model)
    index = load_index(args.backend, "structure")
    meta = {"model": llm.model, "index": f"structure / {index.meta['embedder']['label']}", "k": args.k,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
    print(f"Модель: {meta['model']} · поиск: {meta['index']}, топ-{args.k}\n")

    rows = []
    for q in questions:
        print(f"[{q['id']:>2}/10] {q['q'][:70]}…", flush=True)
        plain = score(ask_plain(llm, q["q"]), q)
        rag = score(ask_rag(llm, index, q["q"], args.k), q)
        print(f"        без RAG: {plain['status']:<16} {plain['coverage']:>4.0%}   │   с RAG: {rag['status']:<16} "
              f"{rag['coverage']:>4.0%}   источник: {mark(rag['retrieved_ok'])} найден, "
              f"{mark(rag['cited_ok'])} процитирован")
        rows.append({**{k: q[k] for k in ("id", "type", "q", "expected", "sources")}, "plain": plain, "rag": rag})

    summ = {"plain": summarize(rows, "plain"), "rag": summarize(rows, "rag")}
    p, r = summ["plain"], summ["rag"]
    print(f"\n{'=' * 72}\nИТОГ  без RAG → с RAG\n{'=' * 72}")
    print(f"Доля фактов (все 10):     {p['avg_coverage']:>5.0%} → {r['avg_coverage']:.0%}")
    print(f"  по проектам (8):        {p['project_coverage']:>5.0%} → {r['project_coverage']:.0%}")
    print(f"  общие вопросы (2):      {p['general_coverage']:>5.0%} → {r['general_coverage']:.0%}")
    print(f"Верно / частично / не знает / неверно:  "
          f"{'/'.join(str(p['statuses'][s]) for s in p['statuses'])} → {'/'.join(str(r['statuses'][s]) for s in r['statuses'])}")
    print(f"Источник найден поиском: {r['retrieved_ok']}, процитирован моделью: {r['cited_ok']}")
    print(f"Токены вход/выход:        {p['prompt_tokens']}/{p['completion_tokens']} → "
          f"{r['prompt_tokens']}/{r['completion_tokens']}")

    os.makedirs(os.path.join(BASE, "output"), exist_ok=True)
    md = os.path.join(BASE, "output", "rag_comparison.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(render(rows, summ, meta))
    with open(os.path.join(BASE, "output", "rag_comparison.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summ, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\nОтчёт с полными ответами: {os.path.relpath(md, BASE)}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Сравнение ответов без RAG и с RAG")
    ap.add_argument("--model", help="модель AITunnel (по умолчанию gpt-4o-mini)")
    ap.add_argument("--backend", choices=["aitunnel", "local"], default="aitunnel", help="эмбеддинги для поиска")
    ap.add_argument("--k", type=int, default=4, help="сколько чанков подставлять в промпт")
    sys.exit(main(ap.parse_args()))
