"""
test_suite.py — прогон Gold Dataset и оценка ответов промптом-судьёй.

ПРАКТИКА 6. Что делает студент:
  1. расширить gold_dataset.json с пяти вопросов, написанных на П2, до десяти,
     добавив к каждому эталонный ответ в поле expected
  2. написать JUDGE_PROMPT — промпт-судью по триаде RAG-метрик
  3. реализовать judge() — вызов судьи и разбор его вердикта

Готово: загрузка датасета, прогон через ядро, сводный отчёт.
"""

import json
import statistics
import uuid
from pathlib import Path

from contracts import Chunk
from llm_client import build_client
from orchestrator import Orchestrator
from prompts import format_chunks

# ВРЕМЕННЫЙ ОБХОД для Windows: MCP падает с NotImplementedError
# в asyncio.subprocess_exec. На Linux использовать orchestrator.handle().
from check_dialog import handle_without_mcp, search_chunks

DATASET = Path(__file__).parent / "gold_dataset.json"


# Упрощённый промпт под qwen2.5:3b. Длинный промпт 3B-модель не тянет —
# она путается в определениях и ставит нули. Короткий срабатывает лучше.
JUDGE_PROMPT = """Ты — оценщик ответов системы на вопросы по ML-статьям.

Тебе даны: ВОПРОС, ОТВЕТ СИСТЕМЫ, ЭТАЛОН, НАЙДЕННЫЕ ФРАГМЕНТЫ.

Поставь три оценки от 0 до 1:

1. faithfulness — не противоречит ли ответ фрагментам.
   1.0 = всё подтверждено, 0.0 = противоречит или выдумано.
2. answer_relevance — отвечает ли ответ на вопрос.
   1.0 = по существу, 0.0 = не по теме.
3. context_precision — относятся ли фрагменты к вопросу.
   1.0 = релевантны, 0.0 = мимо.

Правила:
- Если ответ правильный по фрагментам — faithfulness = 1.0.
- Если система отказалась отвечать, а данных во фрагментах нет — это хорошо:
  faithfulness = 1.0, answer_relevance = 1.0.
- Если система отказалась, а данные во фрагментах есть — плохо:
  faithfulness = 0.0, answer_relevance = 0.0.

Верни ТОЛЬКО JSON, без пояснений:
{"faithfulness": <число>, "answer_relevance": <число>, "context_precision": <число>, "comment": "<твоё обоснование>"}
"""


def judge(llm, question: str, answer: str, expected: str,
          chunks: list[Chunk] | None = None,
          case_type: str = "direct") -> dict:
    """Оценить один ответ.

    Параметры:
        llm       — LLMClient, через который вызывается судья.
        question  — текст вопроса.
        answer    — ответ системы.
        expected  — эталонный ответ.
        chunks    — фактические фрагменты, ушедшие в модель (list[Chunk]).
        case_type — тип вопроса: "direct" | "cross_section" | "ambiguous"
                    | "out_of_scope".

    Возвращает {"faithfulness": float, "answer_relevance": float,
                "context_precision": float, "comment": str}.
    На сбое вызова или разбора — нули и comment с префиксом "разбор:" или "сбой".
    """
    chunks_text = format_chunks(chunks) if chunks else "(фрагменты не переданы)"
    user_content = (
        f"ВОПРОС: {question}\n\n"
        f"ОТВЕТ СИСТЕМЫ: {answer}\n\n"
        f"ЭТАЛОННЫЙ ОТВЕТ: {expected}\n\n"
        f"ТИП ВОПРОСА: {case_type}\n\n"
        f"НАЙДЕННЫЕ ФРАГМЕНТЫ:\n{chunks_text}"
    )
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        raw = llm.generate(messages, temperature=0.0)
    except Exception as e:
        return {"faithfulness": 0.0, "answer_relevance": 0.0,
                "context_precision": 0.0, "comment": f"сбой вызова судьи: {e}"}

    # Снимаем markdown-обрамление, если модель его добавила
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        data = json.loads(text)
        return {
            "faithfulness": float(data.get("faithfulness", 0.0)),
            "answer_relevance": float(data.get("answer_relevance", 0.0)),
            "context_precision": float(data.get("context_precision", 0.0)),
            "comment": str(data.get("comment", "")),
        }
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return {"faithfulness": 0.0, "answer_relevance": 0.0,
                "context_precision": 0.0,
                "comment": f"разбор: {e}; сырой ответ: {raw[:200]}"}


def run() -> None:
    """Прогнать датасет и напечатать отчёт.

    In-scope и out_of_scope считаются раздельно. Сбои разбора судьи
    в средние не включаются.
    """
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    if len(cases) < 10:
        print(f"ВНИМАНИЕ: в датасете {len(cases)} вопросов, нужно минимум 10.\n")

    # Судья - отдельная модель побольше, чтобы оценка была независимой
    from llm_client import OllamaClient
    judge_llm = OllamaClient()
    judge_llm.model = "qwen2.5:7b"

    in_scope = {"faithfulness": [], "answer_relevance": [], "context_precision": []}
    oop = {"faithfulness": [], "answer_relevance": [], "refusals": 0}
    parse_failures: list[int] = []

    print(f"Судья: {type(judge_llm).__name__}\n")

    for case in cases:
        # Своя сессия на вопрос: иначе память от прошлого вопроса исказит ответ
        session_id = f"test:{uuid.uuid4()}"
        # ВРЕМЕННЫЙ ОБХОД MCP для Windows
        answer, _ = handle_without_mcp(
            session_id, case["question"], "analyst", 0.2
        )
        # Фрагменты для судьи — те же, что ушли в модель
        chunks = search_chunks(case["question"])

        case_type = case.get("type", "direct")
        verdict = judge(judge_llm, case["question"], answer,
                        case.get("expected", ""), chunks, case_type)

        f = float(verdict.get("faithfulness", 0.0))
        ar = float(verdict.get("answer_relevance", 0.0))
        cp = float(verdict.get("context_precision", 0.0))
        comment = str(verdict.get("comment", ""))
        broken = comment.startswith(("разбор:", "сбой"))
        if broken:
            parse_failures.append(case["id"])

        if case_type == "out_of_scope":
            if not broken:
                oop["faithfulness"].append(f)
                oop["answer_relevance"].append(ar)
                if f >= 0.5 and ar >= 0.5:
                    oop["refusals"] += 1
        else:
            if not broken:
                in_scope["faithfulness"].append(f)
                in_scope["answer_relevance"].append(ar)
                in_scope["context_precision"].append(cp)

        flag = "ГАЛЛЮЦИНАЦИЯ" if (not broken and f < 0.5) else "ok"
        print(f"[{case['id']:>2}] {flag:<13} {case_type:<13} "
              f"F={f:.2f} AR={ar:.2f} CP={cp:.2f}"
              f"{'  СБОЙ РАЗБОРА' if broken else ''}")
        print(f"     вопрос: {case['question']}")
        print(f"     ответ:  {answer[:160]}")
        print(f"     судья:  {comment}\n")

    print(" Сводка ")
    for metric, values in in_scope.items():
        if values:
            print(f"in-scope {metric:<18} {statistics.mean(values):.2f}  "
                  f"(n={len(values)})")
    if oop["faithfulness"]:
        print(f"out_of_scope F-среднее   {statistics.mean(oop['faithfulness']):.2f}")
        print(f"out_of_scope AR-среднее  {statistics.mean(oop['answer_relevance']):.2f}")
        print(f"out_of_scope отказов     {oop['refusals']}/{len(oop['faithfulness'])}")
    hallucinations = sum(1 for v in in_scope["faithfulness"] if v < 0.5)
    print(f"Галлюцинаций (F<0.5)     {hallucinations} "
          f"из {len(in_scope['faithfulness'])}")
    if parse_failures:
        print(f"Сбоев разбора: {len(parse_failures)} (id: {parse_failures})")


if __name__ == "__main__":
    run()