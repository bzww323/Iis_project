"""
Проверка параграфа 3.6 методички практики №5: диалог из трёх реплик с опорой на историю
"""
import uuid

import config
import memory
import prompts
from contracts import Chunk
from llm_client import build_client


def search_chunks(question: str, k: int = 5) -> list[Chunk]:
    """Поиск фрагментов напрямую через ChromaDB"""
    c = config.get_collection()
    r = c.query(query_texts=[question], n_results=k)
    metas = r["metadatas"][0]
    docs = r["documents"][0]
    dists = r["distances"][0]
    return [
        Chunk(
            text=docs[i],
            source=metas[i]["source"],
            page=metas[i].get("page", 0),
            category=metas[i].get("category", ""),
            score=round(1 - dists[i], 4),
        )
        for i in range(len(docs))
    ]


def handle_without_mcp(session_id: str, question: str, role: str, temperature: float):
    """Аналог orchestrator.handle() без MCP. Возвращает (ответ, дамп сообщений)"""
    # 1. читаем историю ДО сохранения текущей реплики
    history = memory.window(session_id)
    # 2. ищем фрагменты
    chunks = search_chunks(question)
    # 3. сохраняем вопрос
    memory.save(session_id, "user", question)
    # 4. собираем сообщения
    messages = prompts.build_messages(role, question, chunks, history)
    # 5. вызываем LLM
    llm = build_client()
    answer = llm.generate(messages, temperature=temperature)
    # 6. сохраняем ответ
    memory.save(session_id, "assistant", answer)
    return answer, messages


def main():
    memory.create_schema()
    sid = f"demo:{uuid.uuid4()}"
    memory.clear(sid)

    questions = [
        "Что такое self-attention?",
        "А чем он отличается от обычного attention?",
        "А в каких задачах это применяется?",
    ]

    messages_dump = None
    for i, q in enumerate(questions, 1):
        print(f"\n--- Реплика {i}: {q}")
        answer, messages = handle_without_mcp(sid, q, "analyst", 0.2)
        print(f"Ответ: {answer[:300]}")
        if i == 3:
            messages_dump = messages

    print("\n\n=== Дамп build_messages() для третьей реплики ===")
    for i, m in enumerate(messages_dump, 1):
        print(f"{i}. [{m['role']}] {m['content'][:120]}")

    print("\n=== История сессии из БД ===")
    for msg in memory.window(sid):
        print(f"[{msg.role}] {msg.content[:80]}")

    memory.clear(sid)


if __name__ == "__main__":
    main()