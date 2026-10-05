import logging
logging.basicConfig(level=logging.INFO)
from orchestrator import Orchestrator

o = Orchestrator()
phrases = [
    "Что такое self-attention?",
    "Расскажи про Adam",
    "Чем attention отличается от self-attention?",
    "Сравни BERT и ResNet",
    "Ты уверен, что это правильно?",
    "Перепроверь предыдущий ответ",
    "asdf qwerty",
]
for p in phrases:
    intent = o.detect_intent(p)
    role = o.route("test", intent)
    print(f"{p[:40]:<42} интент {intent:<12} роль {role}")
o.close()