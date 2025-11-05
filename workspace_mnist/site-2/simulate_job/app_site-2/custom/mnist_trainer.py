"""
Simple MNIST Federated Learning Trainer (NumPy-based)
This is a simplified example demonstrating federated learning concepts
"""

import numpy as np
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal


class SimpleMNISTTrainer(Executor):
    """Simple MNIST trainer for demonstration"""

    def __init__(self, epochs=1, lr=0.01):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.model_params = None

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Execute training task"""

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
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Perform training"""

        # Get global model
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        self.log_info(fl_ctx, f"Starting training for {self.epochs} epochs")
        self.log_info(fl_ctx, f"Received model keys: {list(global_weights.keys()) if global_weights else 'None'}")

        # Initialize or update model
        if global_weights:
            # Use the same key structure as the global model
            if 'numpy_key' in global_weights:
                # Working with simple numpy array structure
                self.model_params = {'numpy_key': global_weights['numpy_key'].copy()}
            else:
                # Custom weights structure
                self.model_params = {k: v.copy() for k, v in global_weights.items()}
        elif self.model_params is None:
            # Initialize with simple numpy array (3x3 for demo)
            self.model_params = {'numpy_key': np.random.randn(3, 3).astype(np.float32) * 0.01}

        # Simulate training (in real scenario, you would load actual data)
        for epoch in range(self.epochs):
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            # Simulate local training update - add small random updates
            for key in self.model_params:
                self.model_params[key] += np.random.randn(*self.model_params[key].shape).astype(np.float32) * self.lr * 0.01

            self.log_info(fl_ctx, f"Epoch {epoch + 1}/{self.epochs} completed")

        # Create DXO for model weights
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=self.model_params)
        dxo.set_meta_prop("num_steps_current_round", self.epochs)

        self.log_info(fl_ctx, "Training completed")
        return dxo.to_shareable()

    def _validate(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Perform validation"""

        # Get model weights
        dxo = from_shareable(shareable)
        model_weights = dxo.data

        if model_weights:
            self.model_params = model_weights

        # Simulate validation metrics
        accuracy = np.random.uniform(0.85, 0.95)
        loss = np.random.uniform(0.1, 0.3)

        self.log_info(fl_ctx, f"Validation - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

        # Create DXO for metrics
        dxo = DXO(
            data_kind=DataKind.METRICS,
            data={'accuracy': accuracy, 'loss': loss}
        )

        return dxo.to_shareable()