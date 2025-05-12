import os
import pygame
from argparse import ArgumentParser
from icecream import ic

from base_classes.simulation import Simulation
from cart_balancing import CartBalancer, Action
from evolution_logic import fit_evolutionary_model


def interactive_mode(simulation: Simulation):

    # Initialize Pygame
    pygame.init()
    # ic(pygame.font.get_fonts())

    # Set up the display
    framerate = 60
    width, height = 800, 600
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Cart Balancing Simulation")

    # Main loop
    running = True
    simulation_duration = 0.0
    while running:

        ### Handle events
        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
    
        actions = simulation.events_mapping(events)

        ### Update the simulation
        delta_t = clock.tick(framerate) / 1000.0  # Convert milliseconds to seconds
        simulation.step(actions, delta_t)
        simulation_duration += delta_t

        # Render the simulation
        simulation.render(screen)

        # ic(simulation.state())

        ### Text rendering
        font = pygame.font.Font(None, 30)

        # Print the state of the simulation on screen
        # state = simulation.state(True)
        # state_text = f"Angle: {state[2]:.2f}, Cart position: {state[0]:.2f}, Cart velocity: {state[1]:.2f}, Pole angular velocity: {state[3]:.2f}"
        # state_surface = font.render(state_text, True, (0, 0, 0))
        # screen.blit(state_surface, (10, 100))

        # Print FPS on screen
        fps_text = font.render(f"FPS: {clock.get_fps():.2f}", True, (0, 0, 0))
        screen.blit(fps_text, (10, 10))

        # Print score on screen
        score_text = font.render(f"Score: {simulation.score():.2f}", True, (0, 0, 0))
        screen.blit(score_text, (10, 50))

        # Print simulation duration on screen
        duration_text = font.render(f"Duration: {simulation_duration:.2f}s", True, (0, 0, 0))
        screen.blit(duration_text, (10, 80))

        # Update the display
        pygame.display.flip()

    pygame.quit()


def training_mode(simulation: Simulation):
    """
    Run the simulation in training mode.
    """
    # Set up the simulation parameters
    framerate = 60
    simulation_duration = 10.0  # seconds
    input_size = len(simulation.state())  # Number of state variables
    output_size = len(Action)  # Number of actions

    # Run the evolutionary model
    top_individuals = fit_evolutionary_model(
        simulation,
        input_size=input_size,
        output_size=output_size,
        framerate=framerate,
        simulation_duration=simulation_duration,
        population_size=10,
        num_generations=5,
        selection_rate=0.5,
        mutation_rate=0.1,
        return_top_n=1,
    )

    # Save the top individual locally
    os.makedirs("top_models", exist_ok=True)
    for ann, score in top_individuals:
        ann.save_model(f"top_models/best_model_{score:.0f}.pkl")

    print("Top individuals saved.")


def main():
    parser = ArgumentParser(description="Cart Balancing Simulation")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run the simulation in interactive mode.",
    )
    args = parser.parse_args()

    # Create the simulation
    simulation = CartBalancer()

    if args.interactive:
        interactive_mode(simulation)
    else:
        training_mode(simulation)



if __name__ == "__main__":
    main()
