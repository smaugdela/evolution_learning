import random
import numpy as np
from pathlib import Path
import pickle
import pygame


def relu(x):
    """
    ReLU activation function.
    """
    return np.maximum(0, x)


class ANN:

    def __init__(
            self,
            input_size,
            output_size,
            activation_function=relu,
            add_random_hidden_neurons=False,
            max_hidden_neurons=10
        ):

        self.activation_function = activation_function
        self.output_indexes = np.arange(output_size)
        self.input_indexes = np.arange(output_size, output_size + input_size)

        if add_random_hidden_neurons:
            num_hidden = random.randint(1, max_hidden_neurons) if max_hidden_neurons > 0 else 0
            if num_hidden < 0: num_hidden = 0
            self.dimension = input_size + output_size + num_hidden
        else:
            self.dimension = input_size + output_size

        if self.dimension == 0:
            raise ValueError("Network dimension cannot be zero.")
        if input_size < 0 or output_size < 0:
            raise ValueError("Input and output sizes cannot be negative.")

        self.neurons_biases = np.random.rand(self.dimension) * 2 - 1
        self.synapses_weights = np.empty((self.dimension, self.dimension)) 
        self.neuron_rank = {} 

        max_retries = 20 
        success = False
        for attempt in range(max_retries):
            self._initialize_weights_and_ranks()
            self._apply_constraints_and_pruning(0.4)

            # Updated success condition to include hidden neuron connectivity
            if self._check_io_connectivity() and self._check_hidden_connectivity():
                success = True
                break 

            print(f"Attempt {attempt + 1}/{max_retries} failed. Retrying...")

        if not success:
            print(f"Warning: After {max_retries} attempts, network connectivity constraints (I/O and Hidden) "
                  "might not be fully met. The network might have isolated hidden neurons or unviable I/O. "
                  "Consider a different network configuration or more retries if this persists.")

        self.computed_neurons = set()

    def _get_neuron_types_and_indices(self):
        O_indices = list(self.output_indexes)
        I_indices = list(self.input_indexes)
        H_indices = [idx for idx in range(self.dimension) 
                       if idx not in O_indices and idx not in I_indices]
        return O_indices, I_indices, H_indices

    def _initialize_weights_and_ranks(self):
        """Initializes weights to random values and generates new ranks for an attempt."""
        self.synapses_weights = np.random.rand(self.dimension, self.dimension) * 2 - 1
        
        neuron_order = list(range(self.dimension))
        random.shuffle(neuron_order)
        self.neuron_rank = {neuron_idx: i for i, neuron_idx in enumerate(neuron_order)}

    def _generate_significant_random_weight(self):
        """Generates a random weight, avoiding zero and ensuring some magnitude."""
        # Generates value in [-1.0, -0.1] or [0.1, 1.0]
        val = (random.random() * 0.9) + 0.1 
        sign = random.choice([-1, 1])
        return sign * val

    def _apply_constraints_and_pruning(self, synapse_pruning_probability: float = 0.5):
        """
        Applies DAG, I/O roles, prunes, and enforces connectivity. 
        Modifies self.synapses_weights based on self.neuron_rank.
        """
        # Enforce basic input/output roles by zeroing out invalid connections
        for r_child_idx in range(self.dimension):
            for r_parent_idx in range(self.dimension):
                # Parent's rank must be less than child's rank
                if self.neuron_rank.get(r_parent_idx, float('inf')) >= self.neuron_rank.get(r_child_idx, float('-inf')):
                    self.synapses_weights[r_child_idx, r_parent_idx] = 0
                    continue
                
                # Input neurons cannot be children (have no parents)
                if r_child_idx in self.input_indexes:
                    self.synapses_weights[r_child_idx, r_parent_idx] = 0
                    continue

                # Output neurons cannot be parents (have no children)
                if r_parent_idx in self.output_indexes:
                    self.synapses_weights[r_child_idx, r_parent_idx] = 0
                    continue
        
        # Randomly prune some of the remaining (valid) connections
        for r_child_idx in range(self.dimension):
            for r_parent_idx in range(self.dimension):
                if self.synapses_weights[r_child_idx, r_parent_idx] == 0: 
                    continue
                if random.random() < synapse_pruning_probability:
                    self.synapses_weights[r_child_idx, r_parent_idx] = 0
        
        # Ensure connectivity constraints by adding connections if necessary
        O_indices, I_indices, H_indices = self._get_neuron_types_and_indices()

        # For each input neuron, ensure it has at least one child
        if I_indices and (H_indices or O_indices): # Only if there are inputs and potential children
            for i_idx in I_indices:
                if np.count_nonzero(self.synapses_weights[:, i_idx]) == 0: 
                    eligible_children = [
                        c_idx for c_idx in H_indices + O_indices 
                        if self.neuron_rank.get(i_idx, float('-inf')) < self.neuron_rank.get(c_idx, float('inf'))
                    ]
                    if eligible_children:
                        chosen_child = random.choice(eligible_children)
                        self.synapses_weights[chosen_child, i_idx] = self._generate_significant_random_weight()

        # For each output neuron, ensure it has at least one parent
        if O_indices and (H_indices or I_indices): # Only if there are outputs and potential parents
            for o_idx in O_indices:
                if np.count_nonzero(self.synapses_weights[o_idx, :]) == 0: 
                    eligible_parents = [
                        p_idx for p_idx in I_indices + H_indices
                        if self.neuron_rank.get(p_idx, float('-inf')) < self.neuron_rank.get(o_idx, float('inf'))
                    ]
                    if eligible_parents:
                        chosen_parent = random.choice(eligible_parents)
                        self.synapses_weights[o_idx, chosen_parent] = self._generate_significant_random_weight()

        # For hidden neurons: ensure they have at least one parent and one child if possible
        if H_indices:
            for h_idx in H_indices:
                # Ensure parent if possible
                if (I_indices or len(H_indices) > 1) and np.count_nonzero(self.synapses_weights[h_idx, :]) == 0: 
                    eligible_parents = [
                        p_idx for p_idx in I_indices + [idx for idx in H_indices if idx != h_idx]
                        if self.neuron_rank.get(p_idx, float('-inf')) < self.neuron_rank.get(h_idx, float('inf'))
                    ]
                    if eligible_parents:
                        chosen_parent = random.choice(eligible_parents)
                        self.synapses_weights[h_idx, chosen_parent] = self._generate_significant_random_weight()
                
                # Ensure child if possible
                if (O_indices or len(H_indices) > 1) and np.count_nonzero(self.synapses_weights[:, h_idx]) == 0:
                    eligible_children = [
                        c_idx for c_idx in O_indices + [idx for idx in H_indices if idx != h_idx]
                        if self.neuron_rank.get(h_idx, float('-inf')) < self.neuron_rank.get(c_idx, float('inf'))
                    ]
                    if eligible_children:
                        chosen_child = random.choice(eligible_children)
                        self.synapses_weights[chosen_child, h_idx] = self._generate_significant_random_weight()

    def _check_io_connectivity(self):
        """Checks if input neurons have children and output neurons have parents."""
        O_indices, I_indices, H_indices = self._get_neuron_types_and_indices()
        
        if not I_indices and not O_indices: return True # No I/O neurons to check

        # Check inputs only if there are potential children for them
        if I_indices and (O_indices or H_indices):
            for i_idx in I_indices:
                if np.count_nonzero(self.synapses_weights[:, i_idx]) == 0:
                    return False # Input neuron has no children
        
        # Check outputs only if there are potential parents for them
        if O_indices and (I_indices or H_indices):
            for o_idx in O_indices:
                if np.count_nonzero(self.synapses_weights[o_idx, :]) == 0:
                    return False # Output neuron has no parents
        return True

    def _check_hidden_connectivity(self):
        """
        Checks if hidden neurons (if any) have at least one parent and one child.
        """
        O_indices, I_indices, H_indices = self._get_neuron_types_and_indices()

        if not H_indices:
            return True # No hidden neurons, constraint trivially met.

        for h_idx in H_indices:
            # Check for parents for the current hidden neuron h_idx
            has_parent = np.count_nonzero(self.synapses_weights[h_idx, :]) > 0
            if not has_parent:
                self.synapses_weights[h_idx, random.choice(I_indices)] = self._generate_significant_random_weight()

            # Check for children for the current hidden neuron h_idx
            has_child = np.count_nonzero(self.synapses_weights[:, h_idx]) > 0
            if not has_child:
                self.synapses_weights[random.choice(O_indices), h_idx] = self._generate_significant_random_weight()
        return True

    @classmethod
    def load_model(cls, path: str):
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file {path_obj} does not exist.")

        with open(path_obj, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, cls):
            raise TypeError(f"File {path_obj} does not contain a valid {cls.__name__} model.")
        # # Ensure neuron_rank is present if older model didn't have it
        # if not hasattr(data, 'neuron_rank'):
        #     print("Warning: Loaded model does not have 'neuron_rank' attribute.")
        #     data.neuron_rank = {} # Add empty dict if missing, though it might not correspond to weights
        return data

    def save_model(self, path: str) -> Path:
        path_obj = Path(path)
        with open(path_obj, 'wb') as f:
            pickle.dump(self, f)
        return path_obj

    def compute_outputs(self, inputs: np.ndarray) -> np.ndarray:
        if len(inputs) != len(self.input_indexes):
            raise ValueError(f"Input size {len(inputs)} does not match the ANN input size {len(self.input_indexes)}.")

        self.neurons_values = np.empty(self.dimension)
        self.neurons_values[self.input_indexes] = inputs
        
        self.computed_neurons.clear()
        self.computed_neurons.update(self.input_indexes)
        
        recursion_check_set = set()

        output_values = np.empty(len(self.output_indexes))
        for i, index in enumerate(self.output_indexes):
            output_values[i] = self.compute_neuron(index, recursion_check_set)
        return output_values

    def compute_neuron(self, index: int, recursion_check_set: set) -> float:
        if index in self.computed_neurons:
            return self.neurons_values[index]
        
        if index in recursion_check_set:
            raise RecursionError(f"Cycle detected during computation involving neuron {index}. Graph may not be a DAG.")

        recursion_check_set.add(index)

        parents_indexes = np.where(self.synapses_weights[index, :] != 0)[0]

        # If an output neuron (or a hidden one leading to it)
        # ends up without parents, it can't compute a value beyond its bias.
        # The _check_io_connectivity should prevent this for output neurons.
        if not (index in self.input_indexes) and len(parents_indexes) == 0:
            if index in self.output_indexes:
                 print(f"Warning: Output neuron {index} has no parents during computation, will only use its bias.")

        current_value = self.neurons_biases[index]
        for parent_index in parents_indexes:
            assert index != parent_index, f"Neuron {index} has a self-loop to itself as parent."
            current_value += self.synapses_weights[index, parent_index] * self.compute_neuron(parent_index, recursion_check_set)

        output = self.activation_function(current_value)

        self.neurons_values[index] = output
        self.computed_neurons.add(index)
        recursion_check_set.remove(index) 

        return output

    def render_ann_on_surface(self, surface: pygame.Surface, neuron_radius=20, layer_spacing=150, show_weights=False, show_bias=False):
        """
        Renders a graphical representation of the ANN on a given Pygame surface.
        """
        surface.fill((255, 255, 255)) # White background

        O_indices, I_indices, H_indices = self._get_neuron_types_and_indices()

        neuron_positions = {}
        font = pygame.font.SysFont(None, 20 if neuron_radius > 15 else 15)
        weight_font = pygame.font.SysFont(None, 18)

        # Inputs
        x_input = 50
        if I_indices:
            y_step_input = (surface.get_height() - 2 * neuron_radius) / max(1, len(I_indices))
            for i, idx in enumerate(I_indices):
                y = neuron_radius + y_step_input * i + (y_step_input / 2 if len(I_indices) == 1 else 0)
                if len(I_indices) == 1 : y = surface.get_height() / 2
                neuron_positions[idx] = (x_input, y)

        # Outputs
        x_output = surface.get_width() - 50
        if O_indices:
            y_step_output = (surface.get_height() - 2 * neuron_radius) / max(1, len(O_indices))
            for i, idx in enumerate(O_indices):
                y = neuron_radius + y_step_output * i + (y_step_output / 2 if len(O_indices) == 1 else 0)
                if len(O_indices) == 1 : y = surface.get_height() / 2
                neuron_positions[idx] = (x_output, y)

        # Hidden
        if H_indices:
            num_hidden_layers = 1 # For simplicity, one layer of hidden neurons. Could be more sophisticated.
            hidden_x_coords = []
            if x_input + layer_spacing < x_output - layer_spacing:
                hidden_x_coords = [x_input + (i + 1) * ((x_output - x_input - 2*layer_spacing) / (num_hidden_layers +1)) for i in range(num_hidden_layers)]
            else: # Not enough space for dedicated hidden layer column
                hidden_x_coords = [(x_input + x_output) / 2]
            
            y_step_hidden = (surface.get_height() - 2 * neuron_radius) / max(1, len(H_indices))
            for i, idx in enumerate(H_indices):
                # Simplified placement for one column of hidden neurons
                x = hidden_x_coords[0] 
                y = neuron_radius + y_step_hidden * i + (y_step_hidden / 2 if len(H_indices) == 1 else 0)
                if len(H_indices) == 1 : y = surface.get_height() / 2
                neuron_positions[idx] = (x, y)
        
        # If no hidden neurons, ensure neuron_positions is populated for I/O if they exist alone
        if not H_indices:
            if not I_indices and O_indices: # Only outputs
                for idx in O_indices: neuron_positions[idx] = (surface.get_width()/2, neuron_positions[idx][1])
            if not O_indices and I_indices: # Only inputs
                for idx in I_indices: neuron_positions[idx] = (surface.get_width()/2, neuron_positions[idx][1])

        # Draw Synapses as arrows
        # synapses_weights[child, parent]
        for r_child_idx in range(self.dimension):
            for r_parent_idx in range(self.dimension):
                weight = self.synapses_weights[r_child_idx, r_parent_idx]
                if weight != 0:
                    pos_parent = neuron_positions.get(r_parent_idx)
                    pos_child = neuron_positions.get(r_child_idx)

                    if pos_parent is None or pos_child is None: continue

                    # Color based on weight sign, transparency on magnitude
                    alpha = min(255, max(50, int(abs(weight) * 150))) # More opaque for stronger weights
                    if weight > 0:
                        color = (0, 150, 0, alpha) # Greenish
                    else:
                        color = (150, 0, 0, alpha) # Reddish
                    
                    thickness = min(6, max(1, int(abs(weight) * 2.5)))
                    
                    # Create a temporary surface for the line to handle alpha
                    line_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
                    pygame.draw.line(line_surface, color, pos_parent, pos_child, thickness)
                    
                    # Arrowhead (simple triangle)
                    angle = np.arctan2(pos_child[1] - pos_parent[1], pos_child[0] - pos_parent[0])
                    arrow_length = 10
                    arrow_angle_offset = np.pi / 8

                    p1 = (pos_child[0] - arrow_length * np.cos(angle - arrow_angle_offset),
                        pos_child[1] - arrow_length * np.sin(angle - arrow_angle_offset))
                    p2 = (pos_child[0] - arrow_length * np.cos(angle + arrow_angle_offset),
                        pos_child[1] - arrow_length * np.sin(angle + arrow_angle_offset))
                    
                    # Adjust points to be on the circle's edge rather than center
                    dx, dy = pos_child[0] - pos_parent[0], pos_child[1] - pos_parent[1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist > 0:
                        nx, ny = dx/dist, dy/dist # Normalized vector
                        
                        # Move end point of line to edge of child neuron
                        end_point_on_edge = (pos_child[0] - nx * neuron_radius, 
                                            pos_child[1] - ny * neuron_radius)
                        pygame.draw.line(line_surface, color, pos_parent, end_point_on_edge, thickness)

                        # Recalculate arrowhead points based on new end_point_on_edge
                        p1 = (end_point_on_edge[0] - arrow_length * np.cos(angle - arrow_angle_offset),
                            end_point_on_edge[1] - arrow_length * np.sin(angle - arrow_angle_offset))
                        p2 = (end_point_on_edge[0] - arrow_length * np.cos(angle + arrow_angle_offset),
                            end_point_on_edge[1] - arrow_length * np.sin(angle + arrow_angle_offset))
                        pygame.draw.polygon(line_surface, color, [end_point_on_edge, p1, p2])

                    surface.blit(line_surface, (0,0))

                    if show_weights:
                        mid_x = (pos_parent[0] + pos_child[0]) / 2
                        mid_y = (pos_parent[1] + pos_child[1]) / 2
                        weight_text = weight_font.render(f"{weight:.1f}", True, (50,50,50))
                        surface.blit(weight_text, (mid_x, mid_y))

        # Draw Neurons (circles)
        for idx, pos_tuple in neuron_positions.items():
            pos = (int(pos_tuple[0]), int(pos_tuple[1])) # Ensure integer coordinates
            fill_color = (220, 220, 220) # Default grey
            border_color = (0,0,0)
            text_color = (0,0,0)

            if idx in I_indices: fill_color = (173, 216, 230) # Light Blue for input
            elif idx in O_indices: fill_color = (144, 238, 144) # Light Green for output
            elif idx in H_indices: fill_color = (255, 182, 193) # Light Pink for hidden
            
            pygame.draw.circle(surface, fill_color, pos, neuron_radius)
            pygame.draw.circle(surface, border_color, pos, neuron_radius, 2) # Border

            # Draw neuron index label
            idx_text = font.render(str(idx), True, text_color)
            text_rect = idx_text.get_rect(center=pos)
            surface.blit(idx_text, text_rect)

            if show_bias:
                bias_text = font.render(f"b={self.neurons_biases[idx]:.1f}", True, text_color)
                bias_rect = bias_text.get_rect(center=(pos[0], pos[1] + neuron_radius + 7))
                surface.blit(bias_text, bias_rect)


def run_visualization(ann_instance: ANN):
    pygame.init()
    screen_width = 800
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("ANN Visualization")

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    print("Re-initializing ANN for visualization...")
                    ann_instance = ANN(input_size=ann_instance.input_indexes.size,
                                       output_size=ann_instance.output_indexes.size,
                                       add_random_hidden_neurons=bool(len(ann_instance._get_neuron_types_and_indices()[2])),
                                       max_hidden_neurons=10)


        ann_instance.render_ann_on_surface(screen, neuron_radius=25, layer_spacing=200, show_weights=True)
        pygame.display.flip()
        clock.tick(30) # Limit FPS

    pygame.quit()


def main():
    print("Creating ANN...")
    ann = ANN(input_size=3, output_size=2, add_random_hidden_neurons=True, max_hidden_neurons=5)
    print(f"ANN Dimension: {ann.dimension}")
    print(f"Input Indexes: {ann.input_indexes}")
    print(f"Output Indexes: {ann.output_indexes}")
    O, I, H = ann._get_neuron_types_and_indices()
    print(f"Hidden Indexes: {H}")
    # print(f"Neuron Ranks: {ann.neuron_rank}")
    # print(f"Synapse Weights (sum): {np.sum(np.abs(ann.synapses_weights))}")

    inputs = np.array([0.1, 0.2, 0.3])
    print(f"\nComputing with inputs: {inputs}")
    try:
        outputs = ann.compute_outputs(inputs)
        print("Outputs:", outputs)
    except RecursionError as e:
        print(f"Recursion Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Test save/load
    model_path = Path("test_ann_model.pkl")
    print(f"\nSaving model to {model_path}...")
    ann.save_model(str(model_path))
    print("Loading model...")
    loaded_ann = ANN.load_model(str(model_path))
    print(f"Loaded ANN Dimension: {loaded_ann.dimension}")
    outputs_from_loaded = loaded_ann.compute_outputs(inputs)
    print("Outputs from loaded model:", outputs_from_loaded)
    np.testing.assert_array_almost_equal(outputs, outputs_from_loaded)
    print("Outputs match after loading.")
    model_path.unlink() # Clean up


if __name__ == "__main__":
    main()

    # Create an ANN instance for visualization
    vis_ann = ANN(input_size=3, output_size=2, add_random_hidden_neurons=True, max_hidden_neurons=4)
    print("\n--- Running Pygame Visualization ---")
    print("Press 'R' to re-initialize the network, ESC to quit.")
    run_visualization(vis_ann)
