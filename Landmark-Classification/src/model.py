import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.model1 = nn.Sequential(
          # 224*224
          nn.Conv2d(3, 16, 5, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2, 2),
          # 109*109
            
          nn.Conv2d(16, 16, 3, padding=0),
          nn.ReLU(),
          #nn.MaxPool2d(2, 2),
          # 107*107
            
          nn.Conv2d(16, 32, 3, padding=0),
          nn.ReLU(),
          #nn.MaxPool2d(2, 2),
          # 105*105

          nn.Conv2d(32, 32, 3, padding=0),
          nn.ReLU(),
          #nn.MaxPool2d(2, 2),
          # 103*103

          nn.Conv2d(32, 64, 3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2, 2),
          # 50*50
            
          nn.Conv2d(64, 64, 3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2, 2),
          # 24*24
            
          nn.Conv2d(64, 64, 3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2, 2),
          # 12*12
            
          nn.Conv2d(64, 64, 3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2, 2),
          # 6*6

          # Flatten feature maps
          nn.Flatten(),

          # Fully connected layers
          # input image 224x224
          # output from the last conv layer is (64, 7, 7)
          nn.Linear(2304, 512),
          nn.ReLU(),
          nn.Linear(512, 128),
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(128, num_classes)
        )
        
        self.model2 = nn.Sequential(
          nn.Conv2d(3, 16, 3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2, 2),

          nn.Conv2d(16, 32, 3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2, 2),

          nn.Conv2d(32, 64, 3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2, 2),
            
          nn.Conv2d(64, 64, 3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2, 2),
          # 14*14

          nn.Conv2d(64, 64, 3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2, 2),
          # 7*7

          # Flatten feature maps
          nn.Flatten(),

          # Fully connected layers. This assumes
          # that the input image was 32x32
          # output from the last conv layer is (64, 4, 4)
          nn.Linear(3136, 500),
          nn.Dropout(dropout),
          nn.ReLU(),
          nn.Linear(500, num_classes)
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model2(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
