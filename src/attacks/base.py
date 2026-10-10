from abc import ABC, abstractmethod

class Attack(ABC):
    def __init__(self, name: str, strength: float):
        self.name = name
        self.strength = strength
        
    @abstractmethod
    def apply(self, email: str) -> str:
        pass
        
    def get_config(self) -> dict:
        return {
            "name": self.name,
            "strength": self.strength
        }

class NoAttack(Attack):
    def __init__(self):
        super().__init__("None", 0.0)
    
    def apply(self, email: str) -> str:
        return email
