import json
import openai
import google.generativeai as genai
import requests


class GenerationCancelled(Exception):
    # Internal "stop now" signal raised while a model response is still in progress.
    """Raised when an in-flight generation is cancelled by the caller."""


class LLMClient:
    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-5.4-mini-2026-03-17", provider: str = "openai"):
        self.provider = provider.lower()
        self.model = model
        self.base_url = base_url

        if self.provider in ("openai", "deepseek"):
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        elif self.provider == "gemini":
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        elif self.provider == "ollama":
            # Ollama uses a simple HTTP API; no SDK client object
            self.session = requests.Session()
            self.ollama_url = (base_url or "http://localhost:11434").rstrip("/")
        else:
            raise ValueError("Unsupported provider: choose 'openai', 'deepseek', 'gemini', or 'ollama'")

    @staticmethod
    def _should_cancel(cancel_check) -> bool:
        # Small helper: ask one yes/no question, "did the user request Stop?"
        return bool(cancel_check and cancel_check())

    def _chat_openai_compatible(self, messages: list, temperature: float, cancel_check=None) -> str:
        # OpenAI-compatible streaming path: keep checking whether Stop was requested while tokens arrive.
        if not cancel_check:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            return resp.choices[0].message.content

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        chunks = []
        try:
            for chunk in stream:
                if self._should_cancel(cancel_check):
                    raise GenerationCancelled()
                delta = ""
                if getattr(chunk, "choices", None):
                    delta = getattr(chunk.choices[0].delta, "content", "") or ""
                if delta:
                    chunks.append(delta)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

        if self._should_cancel(cancel_check):
            raise GenerationCancelled()
        return "".join(chunks)

    def _chat_gemini(self, messages: list, temperature: float, cancel_check=None) -> str:
        # Gemini path: best-effort cancel checks around the model call.
        conversation = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
        if self._should_cancel(cancel_check):
            raise GenerationCancelled()
        resp = self.client.generate_content(conversation, generation_config={"temperature": temperature})
        if self._should_cancel(cancel_check):
            raise GenerationCancelled()
        return resp.text

    def _chat_ollama(self, messages: list, temperature: float, cancel_check=None) -> str:
        # Ollama streaming path: keep checking whether Stop was requested while chunks arrive.
        url = f"{self.ollama_url}/api/chat"
        if not cancel_check:
            payload = {"model": self.model, "messages": messages, "stream": False, "options": {"temperature": temperature}}
            r = self.session.post(url, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data["message"]["content"]

        payload = {"model": self.model, "messages": messages, "stream": True, "options": {"temperature": temperature}}
        r = self.session.post(url, json=payload, timeout=60, stream=True)
        r.raise_for_status()
        parts = []
        try:
            for line in r.iter_lines(decode_unicode=True):
                if self._should_cancel(cancel_check):
                    raise GenerationCancelled()
                if not line:
                    continue
                data = json.loads(line)
                message = data.get("message", {})
                content = message.get("content", "")
                if content:
                    parts.append(content)
                if data.get("done"):
                    break
        finally:
            r.close()

        if self._should_cancel(cancel_check):
            raise GenerationCancelled()
        return "".join(parts)

    def chat(self, user_prompt: str = None, system_prompt: str = None, messages: list = None,temperature = 0.1, cancel_check=None) -> str:
        # Main model entry point: pass the Stop callback through to whichever provider is active.

        if messages is not None:
            if self.provider in ("openai", "deepseek"):
                return self._chat_openai_compatible(messages=messages, temperature=temperature, cancel_check=cancel_check)
            elif self.provider == "gemini":
                return self._chat_gemini(messages=messages, temperature=temperature, cancel_check=cancel_check)
            elif self.provider == "ollama":
                return self._chat_ollama(messages=messages, temperature=temperature, cancel_check=cancel_check)
        else:
            if self.provider in ("openai", "deepseek"):
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                if user_prompt:
                    messages.append({"role": "user", "content": user_prompt})
                return self._chat_openai_compatible(messages=messages, temperature=temperature, cancel_check=cancel_check)

            elif self.provider == "gemini":
                conversation_messages = []
                if system_prompt:
                    conversation_messages.append({"role": "system", "content": system_prompt})
                if user_prompt:
                    conversation_messages.append({"role": "user", "content": user_prompt})
                return self._chat_gemini(messages=conversation_messages, temperature=temperature, cancel_check=cancel_check)

            elif self.provider == "ollama":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})
                return self._chat_ollama(messages=messages, temperature=temperature, cancel_check=cancel_check)

"""
sample usage: 

# DeepSeek-R1 7B
llm = LLMClient(api_key="", provider="ollama", model="deepseek-r1:7b")

# Llama 3.3 70B
llm = LLMClient(api_key="", provider="ollama", model="llama3.3:70b")

# llama3.2:3b
llm = LLMClient(api_key="", provider="ollama", model="llama3.2:3b")
"""
