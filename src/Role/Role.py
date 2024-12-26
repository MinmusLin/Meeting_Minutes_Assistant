from GPT.ChatGPT import ChatGPT
import os


class Role:
    def __init__(self, role_file: str, model: ChatGPT) -> None:
        self.model = model
        self.role_prompt = self._read_role_prompt(role_file)

    def _read_role_prompt(self, role_file):
        path = os.path.dirname(os.path.abspath(__file__))
        with open(f'{path}/prompts/{role_file}', 'r', encoding='utf-8') as f:
            return f.read()

    def request(self, text: str):
        return self.model.request(text, self.role_prompt)
