from .Role import Role
from GPT.ChatGPT import ChatGPT
import os


class MeetingSecretary(Role):
    def __init__(self, model, template: str = None) -> None:
        super().__init__('MeetingSecretary.md', model)
        self.raw_prompt = self.role_prompt
        self.set_template(template if template is not None else self._get_default_template())

    def set_template(self, template: str) -> str:
        self.role_prompt = self.raw_prompt.replace('${template}', template)

    def _get_default_template(self):
        path = os.path.dirname(os.path.abspath(__file__))
        with open(f'{path}/prompts/DefaultMinutesTemplate.md', 'r', encoding='utf-8') as f:
            return f.read()


class SummaryWriter(Role):
    def __init__(self, model: ChatGPT) -> None:
        super().__init__('SummaryWriter.md', model)


class MeetingMinutesEditor(Role):
    def __init__(self, model, template: str = None) -> None:
        super().__init__('MeetingMinutesEditor.md', model)
        self.raw_prompt = self.role_prompt
        self.set_template(template if template is not None else self._get_default_template())

    def set_template(self, template: str) -> str:
        self.role_prompt = self.raw_prompt.replace('${template}', template)

    def _get_default_template(self):
        path = os.path.dirname(os.path.abspath(__file__))
        with open(f'{path}/prompts/DefaultMinutesTemplate.md', 'r', encoding='utf-8') as f:
            return f.read()
