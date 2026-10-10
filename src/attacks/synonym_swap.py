from .base import Attack
import random

SYNONYMS = {
    'free': 'complimentary',
    'buy': 'purchase',
    'click': 'press',
    'win': 'earn',
    'cash': 'money',
    'guarantee': 'assurance',
    'cheap': 'inexpensive',
    'offer': 'deal',
    'prize': 'reward',
    'discount': 'markdown',
    'urgent': 'pressing',
    'bonus': 'perk'
}

class SynonymSwapAttack(Attack):
    def __init__(self, strength=0.5):
        super().__init__("SynonymSwap", strength)
        
    def apply(self, email: str) -> str:
        words = email.split()
        if not words:
            return email
            
        modified = False
        for i, word in enumerate(words):
            clean_word = word.lower().strip(',.!?')
            if clean_word in SYNONYMS and random.random() < self.strength:
                if word.istitle():
                    words[i] = SYNONYMS[clean_word].capitalize()
                elif word.isupper():
                    words[i] = SYNONYMS[clean_word].upper()
                else:
                    words[i] = SYNONYMS[clean_word]
                modified = True
                
        return " ".join(words) if modified else email
