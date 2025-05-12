from typing import List, Tuple
import numpy as np
from rich.console import Console
from multiprocessing.pool import Pool
import os

from base_classes.ann import ANN
from base_classes.simulation import Simulation
from cart_balancing import Action


def fit_evolutionary_model(
        simulation: Simulation,
        input_size: int,
        output_size: int,
        return_top_n: int = 3,
        framerate: int = 60,
        simulation_duration: float = 10.0,
        num_generations: int = 100,
        population_size: int = 100,
        selection_rate: float = 0.5,
        mutation_rate: float = 0.05,
        console = Console()
    ) -> List[Tuple[ANN, float]]:

    for generation in range(1, num_generations + 1):
        console.log(f"[bold blue]Generation #{generation}[/bold blue]")

        if generation == 1:
            # Initial population
            population = [ANN(input_size, output_size, random_hidden_neurons=True) for _ in range(population_size)]
        else:
            population = selection(result_population, scores, selection_rate)
            population = breeding(population, population_size, input_size, output_size)
            population = mutation(population, mutation_rate)

        console.log("[bold yellow]Simulations...[/bold yellow]")

        result_population: List[ANN] = []
        scores: List[float] = []
        num_processes = max(os.cpu_count() * 2, 1)
        # num_processes = 1  # For debugging purposes, use a single process
        with Pool(num_processes) as pool:
            for individual in population:
                starmap_args = [(individual, simulation.instanciate(), framerate, simulation_duration) for individual in population]

            for ann, score in pool.starmap(ann_life, starmap_args):
                result_population.append(ann)
                scores.append(score)

        total_virtual_simulation_duration = simulation_duration * population_size
        console.log(f"[bold green]Done simulating for {total_virtual_simulation_duration / 60:.2f} virtual minutes.[/bold green]")
        console.log(f"[bold green]Best generation score: {max(scores)}[/bold green]")

    total_virtual_simulation_duration = simulation_duration * population_size * num_generations
    console.log(f"[bold green]Total simulation time: {total_virtual_simulation_duration / 3600:.2f} virtual hours.[/bold green]")

    # Select top N individuals
    top_n = sorted(zip(result_population, scores), key=lambda x: x[1], reverse=True)[:return_top_n]
    # top_individuals = [individual for individual, _ in top_n]
    # console.log(f"[bold green]Top individuals: {top_individuals}[/bold green]")
    top_scores = [score for _, score in top_n]
    console.log(f"[bold green]Top scores: {top_scores}[/bold green]")
    return top_n


def ann_life(
        ann: ANN,
        simulation: Simulation,
        framerate: int,
        simulation_duration: float,
    ) -> Tuple[ANN, float]:

    timestep = 1.0 / framerate

    time = 0.0
    while time < simulation_duration:
        time += timestep

        inputs = np.array(simulation.state(normalized=True))

        outputs = ann.compute_outputs(inputs)

        actions = [Action(outputs.argmax())]

        simulation.step(actions, timestep)

    return ann, simulation.score()


def selection(population: List[ANN], scores: List[float], selection_rate: float) -> List[ANN]:
    """
    Selects a subset of the population based on their scores.
    The selection rate determines the proportion of the population to keep.
    """
    num_selected = int(len(population) * selection_rate)
    selected_indices = np.argsort(scores)[-num_selected:]
    return [population[i] for i in selected_indices]


def breeding(population: List[ANN], population_size: int, input_size: int, output_size: int) -> List[ANN]:
    """
    Breeds new individuals from the selected population.
    The breeding rate determines the proportion of the population to breed.
    """
    new_population = []
    for idx, individual in enumerate(population):
        for potential_id, potential_mate in enumerate(population):
            if potential_id == idx:
                continue
            if individual.dimension == potential_mate.dimension:
                # print(f"Breeding {idx} with {potential_id}")
                new_population.append(individual.breed(potential_mate))

    # Add the new individuals to the population
    population.extend(new_population)

    # Ensure the population size is maintained
    if len(population) > population_size:
        population = population[:population_size]
    elif len(population) < population_size:
        # If the population is smaller than the desired size, fill it with random individuals
        while len(population) < population_size:
            population.append(ANN(input_size, output_size, random_hidden_neurons=True))
    return population

def mutation(population: List[ANN], mutation_rate: float) -> List[ANN]:
    """
    Mutates the population based on the mutation rate.
    The mutation rate determines the probability of each individual being mutated.
    """
    for individual in population:
        if np.random.rand() < mutation_rate:
            individual.mutate()
    return population
