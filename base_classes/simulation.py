# This is the file in which the artificial environment parent class is implemented.

from typing import List, Tuple
import pygame


class Simulation:

    def __init__(self):
        pass

    def step(self, actions: List, delta_t: float):
        """
        This function is called to perform a step in the simulation.
        It should be implemented in the child class.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def reset(self):
        """
        This function is called to reset the simulation to its initial state.
        It should be implemented in the child class.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def render(self, screen: pygame.Surface):
        """
        This function is called to render the current state of the simulation.
        It should be implemented in the child class.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def state(self, normalized: bool = False) -> Tuple:
        """
        This function returns the current state of the simulation.
        It should be implemented in the child class.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def events_mapping(self, events: List[pygame.event.Event]) -> List:
        """
        This function returns the mapping of events to actions.
        It should be implemented in the child class.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def score(self) -> float:
        """
        This function returns the current score of the simulation.
        It should be implemented in the child class.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @classmethod
    def instanciate(cls, *args, **kwargs) -> 'Simulation':
        # Returns a new instance of the same child class
        return cls(*args, **kwargs)
