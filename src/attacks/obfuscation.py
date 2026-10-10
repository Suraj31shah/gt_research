from .base import Attack
import random

class ObfuscationAttack(Attack):
    def __init__(self, strength=0.1):
        super().__init__("Obfuscation", strength)
        
    def apply(self, email: str) -> str:
        words = email.split()
        if not words:
            return email
            
        n_modify = int(len(words) * self.strength)
        indices = random.sample(range(len(words)), min(n_modify, len(words)))
        
        for idx in indices:
            word = words[idx]
            if len(word) > 3:
                # randomly insert a zero-width space or replace char
                if random.random() < 0.5:
                    split_point = random.randint(1, len(word)-1)
                    words[idx] = word[:split_point] + " " + word[split_point:]
                else:
                    word = word.replace('a', '@').replace('o', '0').replace('e', '3').replace('i', '1')
                    words[idx] = word
                    
        return " ".join(words)
