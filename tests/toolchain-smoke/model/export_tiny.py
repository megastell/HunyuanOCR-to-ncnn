from pathlib import Path

import numpy as np
import pnnx
import torch
from torch import nn


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def main() -> None:
    torch.set_grad_enabled(False)
    torch.set_num_threads(9)

    model = TinyNet().eval()

    with torch.no_grad():
        model.fc1.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )

        model.fc1.bias.copy_(
            torch.tensor(
                [0.5, 1.0, -0.5],
                dtype=torch.float32,
            )
        )

        model.fc2.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, -1.0],
                    [-0.5, 1.0, 2.0],
                ],
                dtype=torch.float32,
            )
        )

        model.fc2.bias.copy_(
            torch.tensor(
                [0.25, -0.75],
                dtype=torch.float32,
            )
        )

    input_tensor = torch.tensor(
        [[1.0, -2.0, 0.5, 3.0]],
        dtype=torch.float32,
    )

    with torch.no_grad():
        output = model(input_tensor)

    output_values = output.cpu().numpy().reshape(-1)

    print("PyTorch input:", input_tensor.tolist())
    print("PyTorch output:", output_values.tolist())

    expected_path = Path(__file__).parent / "expected.txt"

    np.savetxt(
        expected_path,
        output_values,
        fmt="%.9f",
    )

    output_path = Path(__file__).parent / "tiny.pt"

    pnnx.export(
        model,
        str(output_path),
        (input_tensor,),
    )

    print("参考输出:", expected_path)
    print("模型导出完成:", output_path)


if __name__ == "__main__":
    main()
