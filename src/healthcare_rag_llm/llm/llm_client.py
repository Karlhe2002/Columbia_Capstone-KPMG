import os

import google.generativeai as genai
import openai
import requests


class LLMClient:
    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-5", provider: str = "openai"):
        self.provider = provider.lower()
        self.model = model
        self.base_url = base_url

        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        elif self.provider == "gemini":
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        elif self.provider == "ollama":
            # Ollama uses a simple HTTP API; no SDK client object
            self.session = requests.Session()
            self.ollama_url = (base_url or "http://localhost:11434").rstrip("/")
        else:
            raise ValueError("Unsupported provider: choose 'openai', 'gemini', or 'ollama'")

    @classmethod
    def from_env(
        cls,
        default_provider: str = "ollama",
        default_model: str = "llama3.2:3b",
        default_base_url: str = None,
    ) -> "LLMClient":
        provider = os.getenv("LLM_PROVIDER", default_provider).strip().lower()
        model = os.getenv("LLM_MODEL", default_model).strip()
        base_url = os.getenv("LLM_BASE_URL", default_base_url)

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
        elif provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "")
        else:
            api_key = ""
            base_url = os.getenv("OLLAMA_BASE_URL", base_url)

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider=provider,
        )

    def _ollama_chat(self, messages: list, temperature: float = 0.1) -> str:
        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            r = self.session.post(url, json=payload, timeout=300)
            r.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Could not connect to Ollama at "
                f"{self.ollama_url}. Start the Ollama server, or set "
                "`OLLAMA_BASE_URL`/`LLM_BASE_URL` to a reachable endpoint. "
                f"Current model: {self.model}."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Ollama request failed for model '{self.model}' at {url}: {exc}"
            ) from exc

        data = r.json()
        return data["message"]["content"]

    def chat(self, user_prompt: str = None, system_prompt: str = None, messages: list = None,temperature = 0.1) -> str:

        if messages is not None:
            if self.provider == "openai":
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature
                )
                return resp.choices[0].message.content
            elif self.provider == "gemini":
                conversation = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
                resp = self.client.generate_content(conversation,generation_config={"temperature":temperature})
                return resp.text
            elif self.provider == "ollama":
                return self._ollama_chat(messages=messages, temperature=temperature)
        else:
            if self.provider == "openai":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                if user_prompt:
                    messages.append({"role": "user", "content": user_prompt})
                resp = self.client.chat.completions.create(model=self.model, messages=messages)
                return resp.choices[0].message.content

            elif self.provider == "gemini":
                conversation = (system_prompt + "\n" if system_prompt else "") + user_prompt
                resp = self.client.generate_content(conversation)
                return resp.text

            elif self.provider == "ollama":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})
                return self._ollama_chat(messages=messages, temperature=temperature)

"""
sample usage: 

# DeepSeek-R1 7B
llm = LLMClient(api_key="", provider="ollama", model="deepseek-r1:7b")

# Llama 3.3 70B
llm = LLMClient(api_key="", provider="ollama", model="llama3.3:70b")

# llama3.2:3b
llm = LLMClient(api_key="", provider="ollama", model="llama3.2:3b")
"""
