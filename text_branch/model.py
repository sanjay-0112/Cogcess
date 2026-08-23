import torch
import torch.nn as nn


class CogcessTextModel(nn.Module):
    """
    Neural network for the Cogcess Text Branch.

    Input:
        11 numerical text/readability features

    Output:
        Predicted readability grade
    """

    def __init__(self, input_size=11):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":

    model = CogcessTextModel()

    print("===== COGCESS TEXT MODEL =====")
    print(model)

    # Test with one fake input containing 11 features
    sample_input = torch.randn(1, 11)

    output = model(sample_input)

    print("\nInput shape:", sample_input.shape)
    print("Output shape:", output.shape)
    print("Test prediction:", output.item())