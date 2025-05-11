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
            random_hidden_neurons=False,
            max_random_hidden_neurons=10,
            fixed_hidden_neurons=3,
        ):

        self.activation_function = activation_function
        self.output_indexes = np.arange(output_size)
        self.input_indexes = np.arange(output_size, output_size + input_size)

        if random_hidden_neurons:
            num_hidden = random.randint(1, max_random_hidden_neurons) if max_random_hidden_neurons > 0 else 0
            if num_hidden < 0: num_hidden = 0
            self.dimension = input_size + output_size + num_hidden
        else:
            self.dimension = input_size + output_size + fixed_hidden_neurons

        if self.dimension == 0:
            raise ValueError("Network dimension cannot be zero.")
        if input_size < 0 or output_size < 0:
            raise ValueError("Input and output sizes cannot be negative.")

        self.neurons_biases = np.random.rand(self.dimension) * 2 - 1
        self.synapses_weights = np.empty((self.dimension, self.dimension)) 
        self.neuron_rank = {} 

        ### LEGACY CODE ###
        # max_retries = 20 
        # success = False
        # for attempt in range(max_retries):
        #     self._initialize_weights_and_ranks()
        #     self._apply_constraints_and_pruning(0.4)

            # # Updated success condition to include hidden neuron connectivity
            # if self._check_io_connectivity() and self._check_hidden_connectivity():
            #     success = True
            #     break 

        #     print(f"Attempt {attempt + 1}/{max_retries} failed. Retrying...")

        # if not success:
        #     print(f"Warning: After {max_retries} attempts, network connectivity constraints (I/O and Hidden) "
        #           "might not be fully met. The network might have isolated hidden neurons or unviable I/O. "
        #           "Consider a different network configuration or more retries if this persists.")
        ### END LEGACY CODE ###

        ### NEW CODE ###
        self._initialize_weights_and_ranks()
        self._apply_constraints_and_pruning(0.5)
        self._check_io_connectivity()
        self._check_hidden_connectivity()
        ### END NEW CODE ###

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
        """
        Checks if input neurons have children and output neurons have parents.
        If not, adds a connection to a random valid child/parent.
        """

        O_indices, I_indices, H_indices = self._get_neuron_types_and_indices()

        if not I_indices and not O_indices: return True # No I/O neurons to check

        # Check inputs only if there are potential children for them
        if I_indices and (O_indices or H_indices):
            for i_idx in I_indices:
                if np.count_nonzero(self.synapses_weights[:, i_idx]) == 0:
                    # Input neuron has no children, connect to a random child
                    self.synapses_weights[random.choice(O_indices + H_indices), i_idx] = self._generate_significant_random_weight()

        # Check outputs only if there are potential parents for them
        if O_indices and (I_indices or H_indices):
            for o_idx in O_indices:
                if np.count_nonzero(self.synapses_weights[o_idx, :]) == 0:
                    # Output neuron has no parents
                    self.synapses_weights[o_idx, random.choice(I_indices + H_indices)] = self._generate_significant_random_weight()

    def _check_hidden_connectivity(self):
        """
        Checks if hidden neurons (if any) have at least one parent and one child.
        """
        O_indices, I_indices, H_indices = self._get_neuron_types_and_indices()

        if not H_indices:
            return  # No hidden neurons, constraint trivially met.

        for h_idx in H_indices:
            # Check for parents for the current hidden neuron h_idx
            has_parent = np.count_nonzero(self.synapses_weights[h_idx, :]) > 0
            if not has_parent:
                self.synapses_weights[h_idx, random.choice(I_indices)] = self._generate_significant_random_weight()

            # Check for children for the current hidden neuron h_idx
            has_child = np.count_nonzero(self.synapses_weights[:, h_idx]) > 0
            if not has_child:
                self.synapses_weights[random.choice(O_indices), h_idx] = self._generate_significant_random_weight()

    @classmethod
    def load_model(cls, path: str):
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file {path_obj} does not exist.")

        with open(path_obj, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, cls):
            raise TypeError(f"File {path_obj} does not contain a valid {cls.__name__} model.")
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
        hidden_width = x_output - x_input
        if H_indices:
            num_hidden_layers = 3 # For simplicity, one layer of hidden neurons. Could be more sophisticated.
            hidden_x_coords = []
            if x_input + layer_spacing < x_output - layer_spacing:
                hidden_x_coords = [
                    x_input + layer_spacing + i * (hidden_width - 2 * layer_spacing) / (num_hidden_layers - 1)
                    for i in range(num_hidden_layers)
                ]
            else: # Not enough space for dedicated hidden layer column
                hidden_x_coords = [(x_input + x_output) / 2]

            y_step_hidden = (surface.get_height() - 2 * neuron_radius) / max(1, len(H_indices))
            for i, idx in enumerate(H_indices):
                # Simplified placement for one column of hidden neurons
                x = hidden_x_coords[0] if len(hidden_x_coords) == 1 else hidden_x_coords[i % num_hidden_layers]
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

    ### MUTATIONS METHODS ###
    def add_hidden_neuron(self, split_connection_prob=0.3):
        """
        Adds a new hidden neuron to the network.
        With probability `split_connection_prob`, it tries to split an existing connection.
        Otherwise, it connects the new neuron to random valid parent/child neurons.
        """
        O_indices, I_indices, H_indices = self._get_neuron_types_and_indices()
        
        new_neuron_idx = self.dimension
        self.dimension += 1
        
        new_biases = np.zeros(self.dimension)
        new_biases[:-1] = self.neurons_biases
        new_biases[new_neuron_idx] = (random.random() * 2 - 1) * 0.01 # Near-zero bias initially
        self.neurons_biases = new_biases

        old_weights = self.synapses_weights
        self.synapses_weights = np.zeros((self.dimension, self.dimension))
        if old_weights.size > 0: # Handle case where network was previously empty
            self.synapses_weights[:-1, :-1] = old_weights
        
        # Assign a rank to the new neuron.
        # Try to insert it "between" existing ranks to facilitate splitting or connecting.
        max_rank = -1
        if self.neuron_rank:
            max_rank = max(self.neuron_rank.values()) if self.neuron_rank else -1
        
        # For splitting, rank needs to be between a parent and child.
        # For general add, rank can be `max_rank + 1` or strategically placed.
        # For now, let's assign a high rank initially. If splitting, this will be adjusted.
        self.neuron_rank[new_neuron_idx] = max_rank + 1

        connection_made = False
        if random.random() < split_connection_prob:
            # Try to split an existing connection: P -> C becomes P -> new_H -> C
            candidate_connections = []
            for r_child_idx in range(new_neuron_idx): # Iterate over old dimension
                for r_parent_idx in range(new_neuron_idx):
                    if self.synapses_weights[r_child_idx, r_parent_idx] != 0:
                        # Ensure parent is not an output and child is not an input
                        parent_is_output = r_parent_idx in self.output_indexes
                        child_is_input = r_child_idx in self.input_indexes
                        if not parent_is_output and not child_is_input:
                             candidate_connections.append((r_parent_idx, r_child_idx))
            
            if candidate_connections:
                p_idx, c_idx = random.choice(candidate_connections)
                original_weight = self.synapses_weights[c_idx, p_idx]
                
                # Adjust rank of new_neuron_idx to be between p_idx and c_idx
                # This is the tricky part to do without breaking other DAG constraints.
                # A simpler rank assignment for splitting:
                # If rank[p] < rank[c], try to set rank[new] = (rank[p] + rank[c]) / 2
                # This might create non-integer ranks or require re-normalization of all ranks.
                # For now, we'll use the simpler high rank and hope it works, or fallback.
                # A robust splitting would re-evaluate ranks more globally.
                
                # Set rank to be just after parent, if possible before child
                self.neuron_rank[new_neuron_idx] = self.neuron_rank.get(p_idx, -1) + 0.5 # Temp non-integer for sorting
                
                # Re-normalize ranks to be integers again
                sorted_neurons_by_temp_rank = sorted(self.neuron_rank.keys(), key=lambda k: self.neuron_rank[k])
                self.neuron_rank = {neuron_idx: i for i, neuron_idx in enumerate(sorted_neurons_by_temp_rank)}

                if self.neuron_rank[p_idx] < self.neuron_rank[new_neuron_idx] and \
                   self.neuron_rank[new_neuron_idx] < self.neuron_rank[c_idx]:
                    
                    self.synapses_weights[c_idx, p_idx] = 0 # Remove old connection
                    self.synapses_weights[new_neuron_idx, p_idx] = 1.0 # P -> new_H
                    self.synapses_weights[c_idx, new_neuron_idx] = original_weight # new_H -> C
                    self.neurons_biases[new_neuron_idx] = 0.0 # Often set to 0 when splitting
                    print(f"Added hidden neuron {new_neuron_idx} by splitting connection {p_idx}->{c_idx}.")
                    connection_made = True
                else:
                    # Failed to place rank correctly for splitting, revert rank change and fall back
                    neuron_order = list(range(self.dimension)) # Re-shuffle all ranks
                    random.shuffle(neuron_order)
                    self.neuron_rank = {neuron_idx: i for i, neuron_idx in enumerate(neuron_order)}


        if not connection_made: # Fallback or general add
            # Assign a generic high rank if splitting failed or wasn't attempted
            max_r = max(self.neuron_rank.values()) if self.neuron_rank else -1
            self.neuron_rank[new_neuron_idx] = max_r +1 # Ensure it's the highest rank
            
            # Connect to one random valid parent
            potential_parents = [
                p_idx for p_idx in I_indices + H_indices 
                if self.neuron_rank.get(p_idx, float('-inf')) < self.neuron_rank[new_neuron_idx]
            ]
            if potential_parents:
                chosen_parent = random.choice(potential_parents)
                self.synapses_weights[new_neuron_idx, chosen_parent] = self._generate_significant_random_weight()

            # Connect to one random valid child
            potential_children = [
                c_idx for c_idx in O_indices + H_indices 
                if self.neuron_rank[new_neuron_idx] < self.neuron_rank.get(c_idx, float('inf'))
            ]
            if potential_children:
                chosen_child = random.choice(potential_children)
                self.synapses_weights[chosen_child, new_neuron_idx] = self._generate_significant_random_weight()
            print(f"Added hidden neuron {new_neuron_idx} with general connections. New dimension: {self.dimension}")

        self.neurons_values = np.empty(self.dimension) 
        self.computed_neurons.clear()
        # After adding, connectivity might need re-validation for the whole network.

    def remove_hidden_neuron(self, neuron_to_remove_idx=None):
        """Removes a hidden neuron. If no index is provided, a random one is chosen."""
        O_indices, I_indices, H_indices = self._get_neuron_types_and_indices()

        if not H_indices:
            # print("No hidden neurons to remove.")
            return False # Indicate failure or no action

        if neuron_to_remove_idx is None:
            neuron_to_remove_idx = random.choice(H_indices)
        elif neuron_to_remove_idx not in H_indices:
            # print(f"Neuron {neuron_to_remove_idx} is not a hidden neuron or does not exist.")
            return False

        # print(f"Removing hidden neuron {neuron_to_remove_idx}...")
        
        new_dim = self.dimension - 1
        if new_dim == 0 : # Cannot remove if it leads to an empty network this way
            # print("Cannot remove neuron, would result in empty network.")
            return False

        new_biases = np.zeros(new_dim)
        new_weights = np.zeros((new_dim, new_dim))
        new_neuron_rank = {}
        
        new_input_indexes = []
        new_output_indexes = []

        current_new_idx = 0
        old_to_new_map = {}

        for old_idx in range(self.dimension):
            if old_idx == neuron_to_remove_idx:
                continue 
            
            old_to_new_map[old_idx] = current_new_idx
            new_biases[current_new_idx] = self.neurons_biases[old_idx]
            if old_idx in self.neuron_rank:
                 new_neuron_rank[current_new_idx] = self.neuron_rank[old_idx] 
            
            if old_idx in self.input_indexes: new_input_indexes.append(current_new_idx)
            if old_idx in self.output_indexes: new_output_indexes.append(current_new_idx)
            current_new_idx += 1
        
        for old_r_child_idx in range(self.dimension):
            if old_r_child_idx == neuron_to_remove_idx: continue
            new_r_child_idx = old_to_new_map[old_r_child_idx]
            for old_r_parent_idx in range(self.dimension):
                if old_r_parent_idx == neuron_to_remove_idx: continue
                new_r_parent_idx = old_to_new_map[old_r_parent_idx]
                new_weights[new_r_child_idx, new_r_parent_idx] = \
                    self.synapses_weights[old_r_child_idx, old_r_parent_idx]

        self.dimension = new_dim
        self.neurons_biases = new_biases
        self.synapses_weights = new_weights
        
        # Re-normalize ranks to be contiguous from 0 to N-1
        if new_neuron_rank:
            sorted_new_neurons_by_old_rank_val = sorted(new_neuron_rank.keys(), key=lambda k: new_neuron_rank[k])
            self.neuron_rank = {neuron_idx: i for i, neuron_idx in enumerate(sorted_new_neurons_by_old_rank_val)}
        else:
            self.neuron_rank = {}

        self.input_indexes = np.array(new_input_indexes)
        self.output_indexes = np.array(new_output_indexes)

        self.neurons_values = np.empty(self.dimension)
        self.computed_neurons.clear()
        # print(f"Removed hidden neuron. New dimension: {self.dimension}")

        # After removal, it's highly recommended to re-check all connectivity constraints
        # and potentially re-run parts of _apply_constraints_and_pruning or the full init retry logic
        # if this mutation is part of an evolutionary algorithm.
        # For now, we just perform the structural removal.
        if self.dimension > 0 and (not self._check_io_connectivity() or not self._check_hidden_connectivity()):
             # print("Warning: Connectivity may be compromised after neuron removal. Re-validation needed.")
             # One could try to "heal" the network here.
             # For now, this function returns True if structural removal was done.
             pass
        return True # Indicate success of removal operation

    def breed(self, other_ann: 'ANN'):
        """
        Breeds with another ANN instance by averaging biases and weights.
        Must be of the same dimension.
        """
        if not isinstance(other_ann, ANN):
            raise ValueError("Can only breed with another ANN instance.")
        if self.dimension != other_ann.dimension:
            raise ValueError("ANNs must have the same dimension to breed.")

        new_biases = (self.neurons_biases + other_ann.neurons_biases) / 2
        new_weights = (self.synapses_weights + other_ann.synapses_weights) / 2

        child = ANN(input_size=len(self.input_indexes), output_size=len(self.output_indexes))
        child.neurons_biases = new_biases
        child.synapses_weights = new_weights
        return child


### Test the ANN class and Pygame visualization ###
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
                                       random_hidden_neurons=bool(len(ann_instance._get_neuron_types_and_indices()[2])),
                                       max_random_hidden_neurons=10)


        ann_instance.render_ann_on_surface(screen, neuron_radius=25, layer_spacing=200, show_weights=True)
        pygame.display.flip()
        clock.tick(30) # Limit FPS

    pygame.quit()


def main():
    print("Creating ANN...")
    ann = ANN(input_size=3, output_size=2, random_hidden_neurons=True, max_random_hidden_neurons=5)
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
    vis_ann = ANN(input_size=3, output_size=2, random_hidden_neurons=True, max_random_hidden_neurons=4)
    print("\n--- Running Pygame Visualization ---")
    print("Press 'R' to re-initialize the network, ESC to quit.")
    run_visualization(vis_ann)
