import requests
import os
import openai

class LLMClient:
    def __init__(self, provider="deepseek", model="deepseek-chat"):
        self.provider = provider
        self.model = model

    def chat(self, messages):
        if self.provider == "deepseek":
            return self._chat_deepseek(messages)
        elif self.provider == "ollama":
            return self._chat_ollama(messages)
        else:
            raise ValueError("Unsupported provider")

    def _chat_deepseek(self, messages):
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"   # DeepSeek 的接口地址
        )
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return resp.choices[0].message.content

    def _chat_ollama(self, messages):
        url = "http://localhost:11434/api/chat"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        resp = requests.post(url, headers=headers, json=data)
        return resp.json()["message"]["content"]
