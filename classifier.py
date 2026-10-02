import io

import torch
from PIL import Image
from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights
)


class ImageClassifier:
    """
    Image classification service using
    the pretrained MobileNetV2 model.
    """

    def __init__(self):

        print("Loading MobileNetV2 model...")

        # Load pretrained model weights
        self.weights = (
            MobileNet_V2_Weights.DEFAULT
        )

        self.model = mobilenet_v2(
            weights=self.weights
        )

        # Set model to evaluation mode
        self.model.eval()

        # Get preprocessing pipeline
        self.preprocess = (
            self.weights.transforms()
        )

        # Get ImageNet class labels
        self.categories = (
            self.weights.meta["categories"]
        )

        print(
            "MobileNetV2 model loaded successfully."
        )

    # ========================================================
    # IMAGE PREPROCESSING
    # ========================================================

    def preprocess_image(
        self,
        image_data: bytes
    ):

        image = Image.open(
            io.BytesIO(image_data)
        ).convert("RGB")

        image_tensor = self.preprocess(
            image
        )

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(
            0
        )

        return image_tensor

    # ========================================================
    # IMAGE PREDICTION
    # ========================================================

    def predict(
        self,
        image_data: bytes
    ):

        image_tensor = (
            self.preprocess_image(
                image_data
            )
        )

        # Disable gradient calculation
        with torch.no_grad():

            outputs = self.model(
                image_tensor
            )

        # Convert outputs into probabilities
        probabilities = torch.nn.functional.softmax(
            outputs,
            dim=1
        )

        # Get top 3 predictions
        top_probabilities, top_indices = (
            torch.topk(
                probabilities,
                3
            )
        )

        results = []

        for probability, index in zip(
            top_probabilities[0],
            top_indices[0]
        ):

            confidence = float(
                probability.item()
            )

            category = self.categories[
                index.item()
            ]

            results.append(
                {
                    "label": category,
                    "confidence": round(
                        confidence,
                        4
                    ),
                    "confidence_percentage": round(
                        confidence * 100,
                        2
                    )
                }
            )

        return {
            "top_prediction": results[0],
            "alternatives": results[1:]
        }