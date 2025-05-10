import pygame
from icecream import ic
from argparse import ArgumentParser

from cart_balancing import CartBalancer, Action


def interactive_mode():
    # Initialize Pygame
    pygame.init()

    # ic(pygame.font.get_fonts())

    # Set up the display
    framerate = 0
    width, height = 800, 600
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Cart Balancing Simulation")

    # Create the simulation
    simulation = CartBalancer()

    # Main loop
    running = True
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
        duration_text = font.render(f"Duration: {simulation.simulation_duration:.2f}s", True, (0, 0, 0))
        screen.blit(duration_text, (10, 80))

        # Update the display
        pygame.display.flip()

    pygame.quit()


def training_mode():
    pass

def main():
    parser = ArgumentParser(description="Cart Balancing Simulation")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run the simulation in interactive mode.",
    )
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    else:
        training_mode()



if __name__ == "__main__":
    main()
