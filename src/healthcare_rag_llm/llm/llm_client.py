class LLMClient:
    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-5.4-mini-2026-03-17", provider: str = "openai"):
        self.provider = provider.lower()
        self.model = model
        self.base_url = base_url

        if self.provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        elif self.provider == "ollama":
            import requests
            # Ollama uses a simple HTTP API; no SDK client object
            self.session = requests.Session()
            self.ollama_url = (base_url or "http://localhost:11434").rstrip("/")
        else:
            raise ValueError("Unsupported provider: choose 'openai', 'gemini', or 'ollama'")

    def chat(
        self,
        user_prompt: str = None,
        system_prompt: str = None,
        messages: list = None,
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        """Send a chat completion request.

        json_mode=True opts into provider-native structured-output enforcement
        so the model is guaranteed to return a syntactically valid JSON object
        (no markdown fences, no leading/trailing prose, no truncated braces).
        Currently only honoured by the OpenAI provider; other providers ignore
        the flag because they don't expose an equivalent guarantee here.
        Caller must still ensure the prompt itself describes the expected JSON
        shape — OpenAI also requires the word "json" to appear in the prompt.
        """

        if messages is not None:
            if self.provider == "openai":
                kwargs = {"model": self.model, "messages": messages, "temperature": temperature}
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            elif self.provider == "gemini":
                conversation = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
                resp = self.client.generate_content(conversation,generation_config={"temperature":temperature})
                return resp.text
            elif self.provider == "ollama":
                url = f"{self.ollama_url}/api/chat"
                payload = {"model": self.model, "messages": messages, "stream": False,"options":{"temperature":temperature}}
                r = self.session.post(url, json=payload, timeout=60)
                r.raise_for_status()
                data = r.json()
                return data["message"]["content"]
        else:
            if self.provider == "openai":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                if user_prompt:
                    messages.append({"role": "user", "content": user_prompt})
                kwargs = {"model": self.model, "messages": messages}
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self.client.chat.completions.create(**kwargs)
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
                url = f"{self.ollama_url}/api/chat"
                payload = {"model": self.model, "messages": messages, "stream": False}
                r = self.session.post(url, json=payload, timeout=60)
                r.raise_for_status()
                data = r.json()
                return data["message"]["content"]

"""
sample usage: 

# Llama 3.3 70B
llm = LLMClient(api_key="", provider="ollama", model="llama3.3:70b")

# llama3.2:3b
llm = LLMClient(api_key="", provider="ollama", model="llama3.2:3b")
"""