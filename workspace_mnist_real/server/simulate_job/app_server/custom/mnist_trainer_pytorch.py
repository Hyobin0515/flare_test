"""
Real MNIST Federated Learning Trainer with PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal


class MNISTNet(nn.Module):
    """Simple CNN for MNIST"""

    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class MNISTTrainer(Executor):
    """Real MNIST trainer for federated learning"""

    def __init__(self, epochs=1, lr=0.01, batch_size=64, data_path="./data"):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.data_path = data_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MNISTNet().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

        self.train_loader = None
        self.test_loader = None

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Execute training or validation task"""

        try:
            if task_name == "train":
                return self._train(shareable, fl_ctx, abort_signal)
            elif task_name == "validate":
                return self._validate(shareable, fl_ctx)
            else:
                self.log_error(fl_ctx, f"Unknown task: {task_name}")
                return make_reply(ReturnCode.TASK_UNKNOWN)

        except Exception as e:
            self.log_error(fl_ctx, f"Exception in task execution: {e}")
            import traceback
            self.log_error(fl_ctx, traceback.format_exc())
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _load_data(self, fl_ctx: FLContext):
        """Load MNIST dataset"""

        if self.train_loader is not None:
            return

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load full MNIST dataset
        train_dataset = datasets.MNIST(
            self.data_path,
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.MNIST(
            self.data_path,
            train=False,
            transform=transform
        )

        # Get client name to partition data
        client_name = fl_ctx.get_identity_name()
        self.log_info(fl_ctx, f"Client: {client_name}")

        # Partition data based on client (simple split for demo)
        # site-1 gets first half, site-2 gets second half
        if "site-1" in client_name:
            indices = list(range(0, 30000))  # First 30k samples
        elif "site-2" in client_name:
            indices = list(range(30000, 60000))  # Last 30k samples
        else:
            indices = list(range(len(train_dataset)))  # All data for other clients

        train_subset = Subset(train_dataset, indices)

        self.train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=1000,
            shuffle=False
        )

        self.log_info(fl_ctx, f"Loaded {len(train_subset)} training samples")

    def _train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Perform training"""

        # Load data
        self._load_data(fl_ctx)

        # Get global model weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        if global_weights:
            self._load_weights(global_weights)
            self.log_info(fl_ctx, "Loaded global model weights")

        self.log_info(fl_ctx, f"Starting training for {self.epochs} epoch(s)")

        self.model.train()

        for epoch in range(self.epochs):
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            total_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                if batch_idx % 100 == 0:
                    self.log_info(fl_ctx,
                        f"Epoch {epoch+1}/{self.epochs}, Batch {batch_idx}/{len(self.train_loader)}, "
                        f"Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(self.train_loader)
            accuracy = 100.0 * correct / total

            self.log_info(fl_ctx,
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Avg Loss: {avg_loss:.4f}, "
                f"Train Accuracy: {accuracy:.2f}%")

        # Get updated model weights
        weights = self._get_weights()

        # Create DXO for model weights
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights)
        dxo.set_meta_prop("NUM_STEPS_CURRENT_ROUND", len(self.train_loader) * self.epochs)

        self.log_info(fl_ctx, "Training completed")
        return dxo.to_shareable()

    def _validate(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Perform validation"""

        # Load data
        self._load_data(fl_ctx)

        # Get model weights
        dxo = from_shareable(shareable)
        weights = dxo.data

        if weights:
            self._load_weights(weights)

        self.model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader)
        accuracy = 100.0 * correct / len(self.test_loader.dataset)

        self.log_info(fl_ctx,
            f"Validation - Avg Loss: {test_loss:.4f}, "
            f"Accuracy: {accuracy:.2f}%")

        # Create DXO for metrics
        dxo = DXO(
            data_kind=DataKind.METRICS,
            data={'val_loss': test_loss, 'val_accuracy': accuracy}
        )

        return dxo.to_shareable()

    def _get_weights(self):
        """Get model weights as dict"""
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

    def _load_weights(self, weights):
        """Load weights into model"""
        import numpy as np
        state_dict = {}
        for k, v in weights.items():
            # Convert to numpy array if not already
            if not isinstance(v, np.ndarray):
                v = np.array(v)
            state_dict[k] = torch.from_numpy(v)
        self.model.load_state_dict(state_dict)