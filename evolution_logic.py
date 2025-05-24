import random
from typing import List, Literal, Tuple
import numpy as np
from rich.console import Console
from multiprocessing.pool import Pool
import os
from matplotlib import pyplot as plt

from base_classes.ann import ANN
from base_classes.simulation import Simulation


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

    # Create a unique hash string from the parameters
    os.makedirs("runs", exist_ok=True)
    run_hash = hex(hash((f"evolution_{input_size}_{output_size}_{framerate}_{simulation_duration}_{num_generations}_{population_size}_{selection_rate}_{mutation_rate}_{random.randint(0, 100000)}")))
    run_hash = "run_" + str(run_hash).replace("-", "m")
    run_dir = f"runs/{run_hash}"
    while os.path.exists(run_dir):
        run_hash = hex(hash((f"evolution_{input_size}_{output_size}_{framerate}_{simulation_duration}_{num_generations}_{population_size}_{selection_rate}_{mutation_rate}_{random.randint(0, 100000)}")))
        run_hash = "run_" + str(run_hash).replace("-", "m")
        run_dir = f"runs/{run_hash}"
    os.makedirs(run_dir, exist_ok=False)

    # Init and launch generations loop
    max_score = -np.inf
    max_scores = []
    average_scores = []
    breakthrough_generations = []
    for generation in range(1, num_generations + 1):
        console.print(f"[bold blue]Generation #{generation}[/bold blue]")

        if generation == 1:
            # Initial population
            population = [ANN.create_individual(input_size, output_size) for _ in range(population_size)]
        else:
            population = selection(result_population, scores, selection_rate)
            population = breeding(population, population_size, input_size, output_size)
            population = mutation(population, mutation_rate)

        console.print("[bold yellow]Simulation...[/bold yellow]")

        result_population: List[ANN] = []
        scores: List[float] = []
        num_processes = max(os.cpu_count() * 2, 1)
        # num_processes = 1  # For debugging purposes, use a single process
        with Pool(num_processes) as pool:
            starmap_args = [(individual, simulation.instanciate(), framerate, simulation_duration) for individual in population]

            for ann, score in pool.starmap(ann_life, starmap_args):
                result_population.append(ann)
                scores.append(score)

        best_score = max(scores)
        max_scores.append(best_score)
        average_score = np.mean(scores)
        average_scores.append(average_score)

        if best_score > max_score:
            max_score = best_score
            breakthrough_generations.append(generation)
            console.print(f"[bold green]New best score: {max_score}[/bold green]")
            # Save the best individual
            best_individual = result_population[scores.index(best_score)]
            best_individual.save_model(f"{run_dir}/best_checkpoint_gen_{generation}_score_{best_score:.1f}.pkl")

        total_virtual_simulation_duration = simulation_duration * population_size
        console.print(f"[green]Best generation score: {max(scores)}[/green]")
        console.print(f"[green]Average generation score: {average_score}[/green]")

    total_virtual_simulation_duration = simulation_duration * population_size * num_generations
    console.print(f"[green]Total simulation time: {total_virtual_simulation_duration / 3600:.2f} virtual hours.[/green]")
    console.print(f"[bold green]Top score: {best_score}[/bold green]")

    final_model, final_score = sorted(zip(result_population, scores), key=lambda x: x[1], reverse=True)[0]
    final_model.save_model(f"{run_dir}/final_model_score_{final_score:.1f}.pkl")

    # Plot the scores
    plt.plot(range(1, num_generations + 1), max_scores, label='Max Score', markevery=breakthrough_generations)
    plt.plot(range(1, num_generations + 1), average_scores, label='Average Score')
    breakthrough_generations.remove(1)
    plt.plot(breakthrough_generations, [max_scores[i-1] for i in breakthrough_generations], 'rx', label='Breakthroughs')
    plt.plot(breakthrough_generations, [average_scores[i-1] for i in breakthrough_generations], 'rx')
    for gen in breakthrough_generations:
        plt.axvline(gen, color='r', linestyle='-.')
    plt.xlabel('Generation')
    plt.ylabel('Scores')
    plt.legend()
    plt.title('Evolutionary Model Scores Over Generations')
    plt.savefig(f"{run_dir}/score_over_generations.png")
    plt.show()

    # Select top N individuals
    top_n = sorted(zip(result_population, scores), key=lambda x: x[1], reverse=True)[:return_top_n]
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

        outputs = ann.forward(inputs).numpy(force=True)

        actions = [simulation.Action(outputs.argmax())]

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

    console = Console()

    new_population = []
    for idx, individual in enumerate(population):
        if len(new_population) + len(population) >= population_size:
            console.print(f"[bold red]Population size reached {len(new_population) + len(population)}, stopping breeding.[/bold red]")
            break
        for potential_id, potential_mate in enumerate(population):
            if potential_id <= idx:
                continue
            if individual.hidden_dimensions == potential_mate.hidden_dimensions:
                # Create a new individual by combining the weights of the two parents
                # console.print(f"[bold green]Breeding individual {idx} with individual {potential_id}.[/bold green]")
                new_individual = individual.breed(potential_mate)
                new_population.append(new_individual)
                break  # Stop breeding with this individual once a mate is found

    # Add the new individuals to the population
    console.print(f"[bold blue]Breeding complete, created {len(new_population)} new individuals.[/bold blue]")
    population.extend(new_population)

    # Ensure the population size is maintained
    if len(population) > population_size:
        console.print(f"[bold red]Population size exceeded {population_size}, trimming the population.[/bold red]")
        population = population[:population_size]
    elif len(population) < population_size:
        # If the population is smaller than the desired size, fill it with random individuals
        console.print(f"[bold yellow]Population size is less than {population_size}, creating new individuals.[/bold yellow]")
        while len(population) < population_size:
            population.append(ANN.create_individual(input_size, output_size))
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
