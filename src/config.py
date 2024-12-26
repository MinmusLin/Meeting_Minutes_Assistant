from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    api_key: str = 'sk-...FnEA'
