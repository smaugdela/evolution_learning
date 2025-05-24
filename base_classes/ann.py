import random
import torch
from torch import nn
from torch.nn import Module
from typing import Iterable, Literal, Optional, List
import numpy as np
from pathlib import Path
import pickle
import pygame


class ANN(Module):

    activation_functions = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh()
    }


    class HiddenLayer(nn.Module):
        def __init__(self, input_size: int, output_size: int, activation_function: nn.Module):
            super().__init__()
            self.layer = nn.Sequential(
                nn.Linear(input_size, output_size),
                activation_function
            )

        def forward(self, x) -> torch.Tensor:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            return self.layer(x)


    def __init__(
            self,
            input_size: int,
            output_size: int,
            activation_function: Literal['relu', 'sigmoid', 'tanh'] = 'relu',
            hidden_dimensions: Iterable[int] = (4, 3, 4),
    ):
        super().__init__() # Call super constructor first

        assert input_size > 0, "Input size must be greater than 0."
        assert output_size > 0, "Output size must be greater than 0."
        assert len(list(hidden_dimensions)) > 0, "At least one hidden layer must be defined."
        assert all(dim > 0 for dim in hidden_dimensions), "All hidden layer dimensions must be greater than 0."

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dimensions = tuple(hidden_dimensions)
        self.activation_function_name = activation_function

        self.torch_activation_function = self.activation_functions.get(
            self.activation_function_name, nn.ReLU()
        )

        self.dimension = input_size + sum(self.hidden_dimensions) + output_size
        self.input_indexes = np.arange(input_size)
        self.output_indexes = np.arange(input_size, input_size + output_size)

        layers = []
        current_input_dim = input_size
        for hidden_size in self.hidden_dimensions:
            layers.append(self.HiddenLayer(current_input_dim, hidden_size, self.torch_activation_function))
            current_input_dim = hidden_size

        layers.append(nn.Linear(current_input_dim, output_size))
        layers.append(nn.Softmax(dim=-1))
        self.mlp = nn.Sequential(*layers)


    def forward(self, inputs) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        return self.mlp(inputs)


    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim == 1:
             inputs = inputs.reshape(1, -1) # Add batch dimension

        with torch.no_grad(): # Inference mode
            outputs_tensor = self.forward(inputs)

        return outputs_tensor.numpy(force=True).squeeze() # Squeeze to remove batch dim if it was 1


    @classmethod
    def create_individual(
        cls,
        input_size: int,
        output_size: int,
        max_hidden_layers: int = 5,
        max_neurons_per_layer: int = 5,
        activation_function: Literal["relu", "sigmoid", "tanh"] = "sigmoid"
    ) -> 'ANN':
        """
        Creates a new individual ANN with the specified input and output sizes.
        """
        n_layers = np.random.randint(1, max_hidden_layers + 1)  # Randomly number of layers
        hidden_dimensions = [np.random.randint(1, max_neurons_per_layer) for _ in range(n_layers)]  # Randomly choose number of neurons per layer
        return cls(input_size, output_size, activation_function, hidden_dimensions)


    @classmethod
    def load_model(cls, path: str):
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file {path_obj} does not exist.")

        with open(path_obj, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, cls):
            raise TypeError(f"File {path_obj} does not contain a valid {cls.__name__} model. Found type {type(data)}")
        return data

    def save_model(self, path: str) -> Path:
        path_obj = Path(path)
        with open(path_obj, 'wb') as f:
            pickle.dump(self, f)
        return path_obj

    def render_ann_on_surface(self, surface: pygame.Surface, show_weights: bool = False, show_bias: bool = False, inputs: Optional[np.ndarray] = None) -> Optional[torch.Tensor]:
        WIDTH, HEIGHT = surface.get_size()
        BACKGROUND_COLOR = (30, 30, 30)
        NEURON_COLOR = (100, 100, 200)
        INPUT_NEURON_COLOR = (100, 200, 100)
        OUTPUT_NEURON_COLOR = (200, 100, 100)
        LINE_COLOR_POSITIVE = (0, 255, 0, 150)  # Added alpha for transparency
        LINE_COLOR_NEGATIVE = (255, 0, 0, 150)  # Added alpha
        LINE_COLOR_NEUTRAL = (150, 150, 150, 100) # Added alpha
        TEXT_COLOR = (220, 220, 220)
        FONT_SIZE = 15
        pygame.font.init() # Ensure font module is initialized
        font = pygame.font.SysFont(None, FONT_SIZE)

        NEURON_RADIUS_DEFAULT = 12
        NEURON_RADIUS_MIN = 6
        NEURON_RADIUS_MAX = 20
        WEIGHT_THICKNESS_MIN = 1
        WEIGHT_THICKNESS_MAX = 4
        WEIGHT_MAGNITUDE_THRESHOLD = 1e-3

        surface.fill(BACKGROUND_COLOR)

        layer_sizes = [self.input_size] + list(self.hidden_dimensions) + [self.output_size]
        num_layers = len(layer_sizes)

        neuron_positions: List[List[tuple[float, float]]] = []
        layer_x_coords = np.linspace(WIDTH * 0.08, WIDTH * 0.92, num_layers)
        
        # max_neurons_in_any_layer = max(layer_sizes) if layer_sizes else 1
        
        for i, size in enumerate(layer_sizes):
            layer_neurons_pos = []
            # Calculate vertical spacing to ensure neurons fit and are centered
            total_height_for_neurons = (size -1) * (NEURON_RADIUS_MAX * 2.5) if size > 1 else NEURON_RADIUS_MAX * 2
            y_start_offset = (HEIGHT - total_height_for_neurons) / 2
            y_start_offset = max(y_start_offset, NEURON_RADIUS_MAX * 1.5) # Ensure margin from top

            for j in range(size):
                x = layer_x_coords[i]
                if size == 1:
                    y = HEIGHT / 2
                else:
                    # Distribute neurons within the allocated vertical space
                    y = y_start_offset + j * (total_height_for_neurons / (size-1) if size > 1 else 0)
                    y = min(y, HEIGHT - NEURON_RADIUS_MAX * 1.5) # Ensure margin from bottom
                layer_neurons_pos.append((x, y))
            neuron_positions.append(layer_neurons_pos)

        activations_per_layer: List[np.ndarray] = []
        if inputs is not None:
            current_tensor_input = torch.tensor(inputs, dtype=torch.float32)
            if current_tensor_input.ndim == 1:
                 current_tensor_input = current_tensor_input.unsqueeze(0) # Batch dim

            activations_per_layer.append(current_tensor_input.squeeze().numpy(force=True)) # Input activations

            temp_input = current_tensor_input
            for layer_module in self.mlp:
                if isinstance(layer_module, (ANN.HiddenLayer, nn.Linear)):
                    temp_input = layer_module(temp_input)
                    activations_per_layer.append(temp_input.detach().squeeze().numpy(force=True))
                elif isinstance(layer_module, nn.Softmax): # Softmax doesn't change shape, could also store its output
                    # temp_input = layer_module(temp_input)
                    # activations_per_layer.append(temp_input.detach().squeeze().numpy(force=True))
                    pass # We already stored pre-softmax linear outputs

        actual_linear_layers = []
        for module_in_mlp in self.mlp:
            if isinstance(module_in_mlp, ANN.HiddenLayer):
                actual_linear_layers.append(module_in_mlp.layer[0])  # nn.Linear inside HiddenLayer
            elif isinstance(module_in_mlp, nn.Linear):
                actual_linear_layers.append(module_in_mlp)

        for layer_idx, torch_linear_layer in enumerate(actual_linear_layers):
            weights = torch_linear_layer.weight.detach().numpy(force=True)
            
            # Normalize weights for thickness calculation for this layer
            abs_weights = np.abs(weights)
            min_abs_w, max_abs_w = abs_weights.min(), abs_weights.max()
            if max_abs_w <= min_abs_w: max_abs_w = min_abs_w + 1e-6 # Avoid division by zero


            for r_idx in range(weights.shape[1]):  # Neuron in 'from' layer (input to this linear layer)
                for c_idx in range(weights.shape[0]):  # Neuron in 'to' layer (output of this linear layer)
                    start_pos = neuron_positions[layer_idx][r_idx]
                    end_pos = neuron_positions[layer_idx + 1][c_idx]
                    weight_val = weights[c_idx, r_idx]

                    line_color = LINE_COLOR_NEUTRAL
                    line_thickness = 1

                    if show_weights:
                        if weight_val > 0:
                            line_color = LINE_COLOR_POSITIVE
                        elif weight_val < 0:
                            line_color = LINE_COLOR_NEGATIVE
                        
                        norm_abs_w_val = (np.abs(weight_val) - min_abs_w) / (max_abs_w - min_abs_w)
                        line_thickness = int(WEIGHT_THICKNESS_MIN + norm_abs_w_val * (WEIGHT_THICKNESS_MAX - WEIGHT_THICKNESS_MIN))
                        line_thickness = max(WEIGHT_THICKNESS_MIN, min(line_thickness, WEIGHT_THICKNESS_MAX))

                        if np.abs(weight_val) < WEIGHT_MAGNITUDE_THRESHOLD and line_thickness <= WEIGHT_THICKNESS_MIN:
                            continue
                    
                    try:
                        pygame.draw.line(surface, line_color, start_pos, end_pos, line_thickness)
                    except TypeError: # sometimes color alpha causes issues if not int
                        pygame.draw.line(surface, (line_color[0],line_color[1],line_color[2]), start_pos, end_pos, line_thickness)


        for layer_idx, current_layer_neuron_positions in enumerate(neuron_positions):
            for neuron_idx, pos in enumerate(current_layer_neuron_positions):
                neuron_radius = NEURON_RADIUS_DEFAULT
                current_neuron_color = NEURON_COLOR
                if layer_idx == 0: current_neuron_color = INPUT_NEURON_COLOR
                elif layer_idx == num_layers - 1: current_neuron_color = OUTPUT_NEURON_COLOR

                if inputs is not None and layer_idx < len(activations_per_layer):
                    layer_activs = activations_per_layer[layer_idx]
                    if not layer_activs.shape:
                        layer_activs = np.array([layer_activs])
                    if neuron_idx < len(layer_activs):
                        activation_value = layer_activs[neuron_idx]

                        max_abs_layer_activation = np.max(np.abs(layer_activs)) if len(layer_activs) > 0 else 1.0
                        if max_abs_layer_activation == 0: max_abs_layer_activation = 1.0
                        
                        scaled_activation_mag = np.abs(activation_value) / max_abs_layer_activation
                        neuron_radius = int(NEURON_RADIUS_MIN + scaled_activation_mag * (NEURON_RADIUS_MAX - NEURON_RADIUS_MIN))

                        blend_factor = min(abs(activation_value * 0.5), 0.7) # How much to blend
                        if activation_value > 0.05: # Small threshold
                            current_neuron_color = tuple(int(max(0,min(255, c1*(1-blend_factor) + c2*blend_factor))) for c1,c2 in zip(current_neuron_color, LINE_COLOR_POSITIVE))
                        elif activation_value < -0.05:
                            current_neuron_color = tuple(int(max(0,min(255, c1*(1-blend_factor) + c2*blend_factor))) for c1,c2 in zip(current_neuron_color, LINE_COLOR_NEGATIVE))
                
                neuron_radius = max(NEURON_RADIUS_MIN, min(neuron_radius, NEURON_RADIUS_MAX))
                pygame.draw.circle(surface, current_neuron_color, (int(pos[0]), int(pos[1])), neuron_radius)
                pygame.draw.circle(surface, (180,180,180), (int(pos[0]), int(pos[1])), neuron_radius, 1) # Border

                if show_bias and layer_idx > 0 and (layer_idx-1) < len(actual_linear_layers) :
                    torch_linear_layer_with_bias = actual_linear_layers[layer_idx-1]
                    if torch_linear_layer_with_bias.bias is not None and neuron_idx < len(torch_linear_layer_with_bias.bias):
                        bias_val = torch_linear_layer_with_bias.bias.detach().numpy(force=True)[neuron_idx]
                        bias_text_surf = font.render(f"{bias_val:.1f}", True, TEXT_COLOR)
                        text_rect = bias_text_surf.get_rect(center=(pos[0], pos[1] + neuron_radius + FONT_SIZE*0.6))
                        surface.blit(bias_text_surf, text_rect)

        # Legend
        y_offset = 10
        if show_weights:
            surface.blit(font.render("Weights: Green (+), Red (-)", True, TEXT_COLOR), (10, y_offset))
            y_offset += FONT_SIZE + 2
        if inputs is not None:
            surface.blit(font.render("Activation: Size/Color Intensity", True, TEXT_COLOR), (10, y_offset))
            y_offset += FONT_SIZE + 2
        if show_bias:
            surface.blit(font.render("Bias values shown below neurons", True, TEXT_COLOR), (10, y_offset))

        if inputs is not None:
            return torch.tensor(activations_per_layer[-1], dtype=torch.float32)  # Return last layer activations as output
        return None  # No inputs provided, nothing to return


    ### MUTATIONS METHODS ###
    def mutate(self, mutation_strength=0.1):
        """
        Randomly selects one parameter tensor (weights or biases of a layer)
        and adds small random noise to all its elements.
        """
        all_params_list = [p for p in self.parameters() if p.requires_grad]
        if not all_params_list:
            return

        param_to_mutate = random.choice(all_params_list)
        
        with torch.no_grad():
            mutation = torch.randn_like(param_to_mutate) * mutation_strength
            param_to_mutate.add_(mutation)


    def breed(self, other_ann: 'ANN') -> 'ANN':
        """
        Creates a new ANN by averaging the weights and biases of this ANN and another ANN.
        The new ANN will have the same architecture as this ANN.
        """
        s_hidden_dims = tuple(self.hidden_dimensions)
        o_hidden_dims = tuple(other_ann.hidden_dimensions)

        if not (self.input_size == other_ann.input_size and
                self.output_size == other_ann.output_size and
                s_hidden_dims == o_hidden_dims and
                self.activation_function_name == other_ann.activation_function_name):
            raise ValueError(
                "Both ANNs must have the same input size, output size, hidden dimensions, "
                "and activation function name for breeding."
            )

        new_ann = ANN(input_size=self.input_size, output_size=self.output_size,
                      activation_function=self.activation_function_name,
                      hidden_dimensions=self.hidden_dimensions)

        with torch.no_grad():
            for p_new, p_self, p_other in zip(new_ann.parameters(), self.parameters(), other_ann.parameters()):
                p_new.data.copy_((p_self.data + p_other.data) / 2.0)

        return new_ann


### Test the ANN class and Pygame visualization ###
def run_visualization(ann_instance: ANN, initial_inputs: Optional[np.ndarray] = None):
    pygame.init()
    screen_width = 900  # Increased width for better spacing
    screen_height = 700 # Increased height
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("ANN Visualization")

    running = True
    clock = pygame.time.Clock()
    
    show_weights_flag = True
    show_bias_flag = False
    current_inputs = initial_inputs.copy() if initial_inputs is not None else None


    print("\n--- Visualization Controls ---")
    print("ESC: Quit")
    print("W: Toggle Weights")
    print("B: Toggle Biases")
    print("I: Provide new inputs (console)")
    print("C: Clear/Randomize inputs")
    print("M: Mutate current ANN")
    print("----------------------------\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_w:
                    show_weights_flag = not show_weights_flag
                    print(f"Show weights: {show_weights_flag}")
                if event.key == pygame.K_b:
                    show_bias_flag = not show_bias_flag
                    print(f"Show biases: {show_bias_flag}")
                if event.key == pygame.K_m:
                    ann_instance.mutate()
                    print("ANN Mutated. Re-evaluating with current inputs if any.")
                if event.key == pygame.K_c:
                    if current_inputs is not None:
                         # Randomize inputs within a typical range, e.g., -1 to 1 or 0 to 1
                        current_inputs = np.random.rand(ann_instance.input_size).astype(np.float32) * 2 - 1
                        print(f"Inputs randomized to: {current_inputs}")
                    else:
                        print("No initial inputs to clear/randomize. Press 'I' to set inputs.")
                if event.key == pygame.K_i:
                    try:
                        input_str = input(f"Enter {ann_instance.input_size} comma-separated input values (e.g., 0.1,0.5,0.2): ")
                        current_inputs = np.array([float(x.strip()) for x in input_str.split(',') if x.strip()], dtype=np.float32)
                        if len(current_inputs) != ann_instance.input_size:
                            print(f"Error: Expected {ann_instance.input_size} inputs, got {len(current_inputs)}")
                            current_inputs = None if initial_inputs is None else initial_inputs.copy() # Revert or clear
                        else:
                             print(f"Inputs set to: {current_inputs}")
                    except ValueError:
                        print("Invalid input format.")
                        current_inputs = None if initial_inputs is None else initial_inputs.copy()


        screen.fill((30,30,30)) # Clear screen before drawing
        ann_instance.render_ann_on_surface(screen, show_weights=show_weights_flag, show_bias=show_bias_flag, inputs=current_inputs)
        pygame.display.flip()
        clock.tick(15) # Limit FPS

    pygame.quit()


def main():
    print("Creating ANN...")

    # Example 1
    ann = ANN(input_size=3, output_size=2, activation_function='relu', hidden_dimensions=[4, 3, 4])
    initial_test_inputs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    
    # Example 2: Larger network
    # ann = ANN(input_size=5, output_size=3, activation_function='sigmoid', hidden_dimensions=[8, 6, 8])
    # initial_test_inputs = np.random.rand(5).astype(np.float32)

    print(f"\nANN Architecture: Input({ann.input_size}) -> Hidden{ann.hidden_dimensions} -> Output({ann.output_size})")
    print(f"Activation function: {ann.activation_function_name}")

    if initial_test_inputs is not None:
        print(f"\nComputing with initial inputs: {initial_test_inputs}")
        try:
            outputs = ann(initial_test_inputs.copy()) # Pass a copy
            print("Outputs:", outputs)
        except Exception as e:
            print(f"An error occurred during initial computation: {e}")
            outputs = None
    else:
        outputs = None

    # Test save/load
    model_path = Path("test_ann_model.pkl")
    print(f"\nSaving model to {model_path}...")
    ann.save_model(str(model_path))
    print("Loading model...")
    try:
        loaded_ann = ANN.load_model(str(model_path))
        if initial_test_inputs is not None and outputs is not None:
            outputs_from_loaded = loaded_ann(initial_test_inputs.copy())
            print("Outputs from loaded model:", outputs_from_loaded)
            np.testing.assert_array_almost_equal(outputs, outputs_from_loaded, decimal=5)
            print("Outputs match after loading.")
        else:
            print("Skipping output comparison as initial inputs/outputs were not available.")
    except Exception as e:
        print(f"Error during model load/test: {e}")
    finally:
        if model_path.exists():
            model_path.unlink() # Clean up

    # Run visualization with the created ANN
    run_visualization(ann, initial_inputs=initial_test_inputs)
    # run_visualization(ann)


if __name__ == "__main__":
    main()
