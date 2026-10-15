"""
llm_client.py — доступ к языковой модели за общим интерфейсом.

ПРАКТИКА 4. Что делает студент:
  1. GigaChatClient.generate() — вызов Сбер GigaChat через официальный SDK
  2. OllamaClient.generate()   — вызов локальной модели через HTTP API Ollama
  3. count_tokens()            — оценка длины промпта до отправки

Готово и трогать не нужно: build_client() и выбор бэкенда по LLM_BACKEND.

Критерии приёмки:
  - смена LLM_BACKEND с gigachat на ollama не требует правки orchestrator.py
  - длина промпта считается до отправки и логируется
  - оба клиента принимают один и тот же список messages и возвращают строку

НЕ МЕНЯТЬ: сигнатуру generate(messages, temperature) — она объявлена в contracts.py
и на неё опирается оркестратор.

Зачем абстракция. Учебные квоты кончаются, сеть в аудитории падает, а занятие
должно продолжаться. Ollama — фолбэк, который работает офлайн. Заодно это
материал для сравнения авторегрессионной модели и reasoning-модели из Л4:
у DeepSeek-R1 в ответе виден блок рассуждений, у GigaChat — нет.

Третий бэкенд — fake_llm.FakeClient, офлайн-заглушка без сети и ключей. Она
нужна для отладки, когда недоступны оба клиента, и ваши реализации не заменяет:
работа, сданная на LLM_BACKEND=fake, не принимается.
"""

import config
from contracts import LLMClient
import requests
import re

class GigaChatClient:
    """Клиент Сбер GigaChat API."""

    def __init__(self):
        self.model = config.GIGACHAT_MODEL
        # TODO (П4): создать клиент
        # from gigachat import GigaChat
        # self._client = GigaChat(credentials=config.GIGACHAT_CREDENTIALS,
        #                         model=self.model,
        #                         verify_ssl_certs=config.GIGACHAT_VERIFY_SSL)

    def generate(self, messages: list[dict], temperature: float = 0.3) -> str:
        """Отправить диалог в GigaChat и вернуть текст ответа.

        SDK принимает структуру вида {"messages": [...], "temperature": ...}.
        Ответ лежит в response.choices[0].message.content.
        Не забудьте обработать сетевую ошибку: занятие не должно падать трассировкой.
        """
        raise NotImplementedError("П4: реализуйте вызов GigaChat")

    def count_tokens(self, text: str) -> int:
        """Оценка длины в токенах.

        У GigaChat есть метод tokens_count. Если он недоступен, дайте оценку:
        для русского текста примерно один токен на 3 символа. Точность здесь
        не главное — важно видеть порядок величины и связать его со стоимостью.
        """
        raise NotImplementedError("П4: реализуйте подсчёт токенов")


class OllamaClient:
    # клиент локального инференса через Ollama

    def __init__(self):
        self.model = config.OLLAMA_MODEL
        self.host = config.OLLAMA_HOST

    def generate(self, messages: list[dict], temperature: float = 0.3) -> str:
        """POST на {host}/api/chat с телом {"model", "messages", "stream": False}.

        Ответ — в data["message"]["content"].
        У reasoning-моделей ответ приходит с блоком <think>...</think> — решите,
        показывать его пользователю или вырезать, и обоснуйте выбор на защите.
        """
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False, # ответ целиком, а не потоком
                    "options": {"temperature": temperature}, # температура роли из ROLES
                },
                timeout=120, # локальная модель может думать долго, 2 минуты с запасом
            )
            response.raise_for_status() # 4xx и 5xx превратит в исключение
            answer = response.json()["message"]["content"]
            return self._strip_thinking(answer)
        except requests.exceptions.Timeout:
            return "[ошибка Ollama: таймаут 120 секунд, модель не ответила]"
        except requests.exceptions.ConnectionError:
            return "[ошибка Ollama: сервис недоступен, проверьте, запущен ли ollama]"
        except (KeyError, ValueError) as e:
            return f"[ошибка Ollama: неожиданный формат ответа: {e}]"
        except requests.exceptions.RequestException as e:
            return f"[ошибка Ollama: {e}]"

    def count_tokens(self, text: str) -> int:
        """Оценка длины в токенах — та же логика, что и в GigaChatClient.

        Для русского текста примерно один токен на три символа.
        Точность здесь не главное — важен порядок величины.
        """
        return max(1, len(text) // 3)

    def _strip_thinking(self, text: str) -> str:
        """Вырезать блок <think>...</think> из ответа reasoning-модели.

        DeepSeek-R1 вставляет блок рассуждений перед ответом. Пользователю он
        не нужен — там черновик. Показываем только итоговый ответ.
        """
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def build_client() -> LLMClient:
    """Выбрать бэкенд по конфигурации. Готово, менять не нужно."""
    if config.LLM_BACKEND == "ollama":
        return OllamaClient()
    if config.LLM_BACKEND == "gigachat":
        return GigaChatClient()
    if config.LLM_BACKEND == "fake":
        # Импорт внутри ветки: без него fake_llm тянулся бы при любом бэкенде
        from fake_llm import FakeClient

        return FakeClient()
    raise ValueError(
        f"Неизвестный LLM_BACKEND: {config.LLM_BACKEND!r}. "
        f"Допустимо: gigachat | ollama | fake"
    )
