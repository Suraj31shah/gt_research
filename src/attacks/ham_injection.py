from .base import Attack
import random

HAM_PHRASES = [
    "attached is the report",
    "let's schedule a meeting",
    "best regards",
    "please review the document",
    "following up on our conversation",
    "the project update is ready",
    "thanks for your help",
    "sincerely",
    "per our discussion"
]

class HamInjectionAttack(Attack):
    def __init__(self, strength=1.0):
        super().__init__("HamInjection", strength)
        self.n_phrases = max(1, int(strength))
        
    def apply(self, email: str) -> str:
        lines = email.split('\n')
        
        for _ in range(self.n_phrases):
            phrase = random.choice(HAM_PHRASES)
            pos = random.choice(['start', 'end', 'middle'])
            if pos == 'start':
                lines.insert(0, phrase)
            elif pos == 'end':
                lines.append(phrase)
            else:
                idx = random.randint(0, max(0, len(lines)-1))
                lines.insert(idx, phrase)
                
        return "\n".join(lines)
