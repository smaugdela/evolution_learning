# This file implements a cart balancer simulation using the Simulation base class.

from enum import Enum
import random
from typing import List, Tuple
import math
import pygame
from icecream import ic

from base_classes.simulation import Simulation




class CartBalancer(Simulation):

    class Action(Enum):
        NONE = 0
        LEFT = 1
        RIGHT = 2

    # Simulation constants
    gravity = 9.81  # Acceleration due to gravity (m/s^2)
    mass_cart = 1.0  # Mass of the cart (kg)
    length_pole = 1  # Length of the pole (m)
    force_mag = 15.0  # Magnitude of the force applied to the cart (N)
    simulation_width = 5  # Width of the simulation area (m)
    friction_constant = 0.5  # Friction constant for the cart (N/m/s)

    # Render constants, in window ratios
    cart_width = 0.05
    cart_width_to_height = 0.5
    rail_height = 0.2

    # Learning hyperparameters
    max_score_per_second = 10.0

    def __init__(self):
        super().__init__()

        # State variables
        self.cart_position = 0
        self.cart_velocity = 0
        self.pole_angle = 0
        self.pole_angular_velocity = 0

        self._state = (self.cart_position, self.cart_velocity, self.pole_angle, self.pole_angular_velocity)

        self.actions = []
        self._score = 0
        self.left_bound_collision = False
        self.right_bound_collision = False

        self.reset()

    def events_mapping(self, events: List[pygame.event.Event]) -> List[Action]:
        """
        Maps Pygame events to actions.
        :param events: List of Pygame events.
        :return: List of actions corresponding to the events.
        """
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.actions.append(CartBalancer.Action.LEFT)
                elif event.key == pygame.K_RIGHT:
                    self.actions.append(CartBalancer.Action.RIGHT)
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    self.actions.remove(CartBalancer.Action.LEFT)
                elif event.key == pygame.K_RIGHT:
                    self.actions.remove(CartBalancer.Action.RIGHT)
        return self.actions


    def step(self, actions: List[Action], delta_t: float = 0.016) -> None:
        """
        Performs a step in the simulation based on the action taken.
        :param action: Action taken (LEFT, RIGHT, NONE).
        :param delta_t: Time step for the simulation.
        :return: None
        """

        force = 0
        if CartBalancer.Action.LEFT in actions and self.left_bound_collision is False:
            force -= self.force_mag
        elif CartBalancer.Action.RIGHT in actions and self.right_bound_collision is False:
            force += self.force_mag

        self.left_bound_collision = False
        self.right_bound_collision = False

        # Physics calculations
        # Update cart position and velocity
        cart_acceleration = (force - self.cart_velocity * self.friction_constant) / self.mass_cart
        self.cart_velocity += cart_acceleration * delta_t
        self.cart_position += self.cart_velocity * delta_t
        # Ensure the cart stays within the bounds of the simulation
        if self.cart_position <= -self.simulation_width / 2:
            self.cart_position = -self.simulation_width / 2
            self.cart_velocity = 0
            self.left_bound_collision = True
        elif self.cart_position >= self.simulation_width / 2:
            self.cart_position = self.simulation_width / 2
            self.cart_velocity = 0
            self.right_bound_collision = True

        # Update pole angle and angular velocity
        angular_acceleration = ((-cart_acceleration * math.cos(self.pole_angle)) + ((self.gravity / self.length_pole) * math.sin(self.pole_angle)) - (self.pole_angular_velocity * self.friction_constant)) / self.mass_cart
        self.pole_angular_velocity += angular_acceleration * delta_t
        new_pole_angle = self.pole_angle + self.pole_angular_velocity * delta_t
        if new_pole_angle < -math.pi:
            new_pole_angle += 2 * math.pi
        elif new_pole_angle > math.pi:
            new_pole_angle -= 2 * math.pi
        self.pole_angle = new_pole_angle

        self._state = (self.cart_position, self.cart_velocity, self.pole_angle, self.pole_angular_velocity)

        self._score += self.step_score(delta_t)

    def state(self, normalized: bool = False) -> Tuple:
        """
        Returns the current state of the simulation.

        :param normalized bool: If True, returns the state variables normalized to the simulation area (-1 to 1) (velocities are standardized).

        :return: Current state of the simulation (cart position, cart velocity, pole angle, pole angular velocity).
        """
        if normalized:
            # Normalize the state variables
            cart_position = self.cart_position / (self.simulation_width / 2)
            cart_velocity = self.cart_velocity / self.simulation_width
            pole_angle = self.pole_angle / math.pi
            pole_angular_velocity = self.pole_angular_velocity / (2 * math.pi)

            return (cart_position, cart_velocity, pole_angle, pole_angular_velocity)
        return self._state

    def render(self, surface: pygame.Surface):
        """
        Renders the current state of the simulation using Pygame.
        :param pygame.Surface: The Pygame surface to render on.
        """

        # Get screen dimensions
        width = surface.get_width()
        height = surface.get_height()

        pixels_per_meter = width / self.simulation_width

        # Draw a rail as a line in the bottom of the screen
        pygame.draw.line(surface, (0, 0, 0), (0, height - int(height * self.rail_height)), (width, height - int(height * self.rail_height)), 4)

        # Draw the cart
        cart_center_x = int(width * (self.cart_position + self.simulation_width / 2) / self.simulation_width)
        cart_center_y = int(height - height * self.rail_height)
        cart_width = int(width * self.cart_width)
        cart_height = int(cart_width * self.cart_width_to_height)
        cart_x = cart_center_x - int(cart_width / 2)
        cart_y = cart_center_y - int(cart_height / 2)
        cart = pygame.Rect(cart_x, cart_y, cart_width, cart_height)
        pygame.draw.rect(surface, (10, 60, 210), cart)

        # Draw the pole
        pole_length = int(pixels_per_meter * self.length_pole)
        pole_end_x = cart_center_x + int(pole_length * math.sin(self.pole_angle))
        pole_end_y = cart_center_y - int(pole_length * math.cos(self.pole_angle))
        pygame.draw.line(surface, (210, 60, 10), (cart_center_x, cart_center_y), (pole_end_x, pole_end_y), 5)

    def reset(self):
        """
        Resets the simulation to its initial state.
        """
        self.cart_position = 0
        self.cart_velocity = 0
        self.pole_angle = 0
        self.pole_angular_velocity = random.uniform(-0.1, 0.1)  # Random initial angular velocity
        while self.pole_angular_velocity == 0.0:
            self.pole_angular_velocity = random.uniform(-0.1, 0.1)

        self._state = (self.cart_position, self.cart_velocity, self.pole_angle, self.pole_angular_velocity)
        self.actions = []
        self._score = 0
        self.left_bound_collision = False
        self.right_bound_collision = False

        self.simulation_duration = 0

        return self._state

    def step_score(self, delta_t: float) -> float:

        def formulae(x: float, k: float) -> float:
            """
            https://www.desmos.com/calculator/kf1c5i7pes
            """
            return (-(x ** 2) + 1) * k if (-1. <= x <= 1.) else 0

        cart_position, cart_velocity, pole_angle, pole_angular_velocity = self.state(normalized=True)
        return formulae(pole_angle, self.max_score_per_second * delta_t) * formulae(cart_position, self.max_score_per_second * delta_t)

    def score(self) -> float:
        """
        Returns the current score of the simulation.
        :return: Current score of the simulation.
        """
        return self._score
