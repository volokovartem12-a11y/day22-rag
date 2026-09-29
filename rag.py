"""
День 22. Первый RAG-запрос.

    вопрос ──► поиск релевантных чанков в индексе (день 21)
           ──► объединение чанков с вопросом в один промпт
           ──► запрос к LLM ──► ответ со ссылками на источники [1], [2] …

Два режима с одинаковой моделью:
  ask_plain() — без RAG: модель отвечает только из своих знаний;
  ask_rag()   — с RAG: модель отвечает по найденным фрагментам базы.
"""

import os
import re

from build_index import build, index_path
from index_store import Index

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "gpt-4o-mini"

PLAIN_SYSTEM = (
    "Ты — ассистент. Ответь на вопрос кратко и по существу, на русском. "
    "Если точно не знаешь ответа, прямо скажи, что не знаешь, и не выдумывай."
)

RAG_SYSTEM = (
    "Ты отвечаешь на вопросы по базе документации проектов. Используй ТОЛЬКО фрагменты "
    "из раздела «Контекст». После каждого утверждения ставь ссылку на фрагмент в виде [n]. "
    "Приводи точные названия, числа и имена из контекста. Если в контексте нет ответа — "
    "прямо скажи, что в базе нет ответа на этот вопрос, и ничего не придумывай. "
    "Отвечай кратко, на русском."
)

REFUSAL = re.compile(r"не знаю|нет информации|не располагаю|не могу (точно )?(ответить|сказать)|"
                     r"в базе нет|в контексте нет|нет данных|недостаточно (данных|информации)|"
                     r"не упоминается|не указан", re.IGNORECASE)


class LLM:
    """Тонкая обёртка над AITunnel (OpenAI-совместимый API)."""

    def __init__(self, model: str = None):
        key = os.environ.get("AITUNNEL_API_KEY")
        if not key:
            raise SystemExit("Не задан AITUNNEL_API_KEY — ключ нужен для запросов к модели.")
        from openai import OpenAI
        self.model = model or os.environ.get("AITUNNEL_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(api_key=key, base_url="https://api.aitunnel.ru/v1/")

    def chat(self, system: str, user: str) -> tuple:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except Exception as e:  # noqa: BLE001
            text = str(e)
            if "model" in text.lower() and ("not" in text.lower() or "не найден" in text.lower() or "404" in text):
                raise SystemExit(f"Модель {self.model} недоступна в AITunnel: {text}\n"
                                 f"Попробуйте другую, например -Model gpt-4o-mini") from e
            raise
        usage = getattr(resp, "usage", None)
        return (resp.choices[0].message.content or "").strip(), {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }


def load_index(backend: str = "aitunnel", strategy: str = "structure") -> Index:
    """Индекс из дня 21. Если его ещё нет — собирается автоматически."""
    path = index_path(strategy, backend)
    if not os.path.exists(path):
        print(f"Индекс {os.path.relpath(path, BASE)} не найден — собираю ...")
        from types import SimpleNamespace as NS
        build(NS(docs=os.path.join(BASE, "docs"), backend=backend, model=None,
                 size=1000, overlap=150, max_chars=1500, min_chars=200))
    return Index(path)


def retrieve(index: Index, question: str, k: int = 4) -> list:
    """Шаг 1: поиск релевантных чанков."""
    return index.search(question, k)


def build_prompt(question: str, chunks: list) -> str:
    """Шаг 2: объединение найденных чанков с вопросом."""
    blocks = [f"[{i}] источник: {c['source']} | раздел: {c['section']}\n{c['text']}"
              for i, c in enumerate(chunks, 1)]
    return "Контекст:\n\n" + "\n\n---\n\n".join(blocks) + f"\n\nВопрос: {question}"


def cited_sources(answer: str, chunks: list) -> list:
    """Какие фрагменты модель реально процитировала: [1], [2, 3] → пути файлов."""
    nums = {int(n) for group in re.findall(r"\[([\d,\s]+)\]", answer) for n in re.findall(r"\d+", group)}
    return sorted({chunks[n - 1]["source"] for n in nums if 1 <= n <= len(chunks)})


def ask_plain(llm: LLM, question: str) -> dict:
    answer, usage = llm.chat(PLAIN_SYSTEM, question)
    return {"mode": "plain", "question": question, "answer": answer, "usage": usage}


def ask_rag(llm: LLM, index: Index, question: str, k: int = 4) -> dict:
    """Полный RAG-запрос: вопрос → поиск чанков → объединение → запрос к LLM."""
    chunks = retrieve(index, question, k)
    prompt = build_prompt(question, chunks)
    answer, usage = llm.chat(RAG_SYSTEM, prompt)       # шаг 3: запрос к LLM
    return {
        "mode": "rag", "question": question, "answer": answer, "usage": usage,
        "chunks": [{"n": i, "chunk_id": c["chunk_id"], "source": c["source"], "section": c["section"],
                    "score": round(c["score"], 3)} for i, c in enumerate(chunks, 1)],
        "cited": cited_sources(answer, chunks),
        "prompt_chars": len(prompt),
    }


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL.search(answer))
