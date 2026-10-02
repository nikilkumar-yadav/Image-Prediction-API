import json
import random
import shutil
from pathlib import Path

from torchvision.datasets import CIFAR100
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "dataset"

RAW_DIR = DATASET_DIR / "raw"

TRAIN_DIR = DATASET_DIR / "train"

VALIDATION_DIR = DATASET_DIR / "validation"

TEST_DIR = DATASET_DIR / "test"

SEED = 42

VALIDATION_RATIO = 0.20


# ============================================================
# SELECTED CLASSES
# ============================================================

SELECTED_CLASSES = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bee",
    "bicycle",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "clock",
    "cloud",
    "keyboard",
    "mouse",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "forest",
    "fox",
    "girl",
    "hamster",
]


# ============================================================
# CREATE DIRECTORY
# ============================================================

def create_directory(path: Path):

    path.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# DOWNLOAD DATASET
# ============================================================

def download_dataset():

    print()
    print("=" * 60)
    print("Downloading CIFAR-100 dataset")
    print("=" * 60)
    print()

    create_directory(RAW_DIR)

    train_dataset = CIFAR100(
        root=str(RAW_DIR),
        train=True,
        download=True
    )

    test_dataset = CIFAR100(
        root=str(RAW_DIR),
        train=False,
        download=True
    )

    print()
    print("CIFAR-100 downloaded successfully.")

    return train_dataset, test_dataset


# ============================================================
# CREATE CLASS FOLDERS
# ============================================================

def create_class_directories():

    print()
    print("Creating dataset directories...")
    print()

    for base_directory in [
        TRAIN_DIR,
        VALIDATION_DIR,
        TEST_DIR
    ]:

        create_directory(base_directory)

        for class_name in SELECTED_CLASSES:

            create_directory(
                base_directory / class_name
            )

    print("Directories created successfully.")


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(
    image,
    output_path: Path
):

    if not isinstance(image, Image.Image):

        image = Image.fromarray(image)

    image.save(
        output_path,
        format="PNG"
    )


# ============================================================
# PREPARE TRAINING DATA
# ============================================================

def prepare_training_data(dataset):

    print()
    print("Preparing training and validation data...")
    print()

    random.seed(SEED)

    class_indices = {}

    for index, class_name in enumerate(
        dataset.classes
    ):

        if class_name in SELECTED_CLASSES:

            class_indices[class_name] = []

    for index, label in enumerate(
        dataset.targets
    ):

        class_name = dataset.classes[label]

        if class_name in class_indices:

            class_indices[class_name].append(
                index
            )

    total_training_images = 0
    total_validation_images = 0

    for class_name in SELECTED_CLASSES:

        indices = class_indices[class_name]

        random.shuffle(indices)

        validation_count = int(
            len(indices) * VALIDATION_RATIO
        )

        validation_indices = (
            indices[:validation_count]
        )

        training_indices = (
            indices[validation_count:]
        )

        # Save training images
        for image_number, index in enumerate(
            training_indices
        ):

            image = dataset.data[index]

            output_path = (
                TRAIN_DIR
                / class_name
                / f"{image_number:05d}.png"
            )

            save_image(
                image,
                output_path
            )

            total_training_images += 1

        # Save validation images
        for image_number, index in enumerate(
            validation_indices
        ):

            image = dataset.data[index]

            output_path = (
                VALIDATION_DIR
                / class_name
                / f"{image_number:05d}.png"
            )

            save_image(
                image,
                output_path
            )

            total_validation_images += 1

        print(
            f"{class_name:20} "
            f"train={len(training_indices):3} "
            f"validation={len(validation_indices):3}"
        )

    print()
    print(
        f"Total training images   : "
        f"{total_training_images}"
    )

    print(
        f"Total validation images : "
        f"{total_validation_images}"
    )


# ============================================================
# PREPARE TEST DATA
# ============================================================

def prepare_test_data(dataset):

    print()
    print("Preparing test data...")
    print()

    class_counters = {
        class_name: 0
        for class_name in SELECTED_CLASSES
    }

    total_test_images = 0

    for index, label in enumerate(
        dataset.targets
    ):

        class_name = dataset.classes[label]

        if class_name not in class_counters:

            continue

        image = dataset.data[index]

        image_number = class_counters[
            class_name
        ]

        output_path = (
            TEST_DIR
            / class_name
            / f"{image_number:05d}.png"
        )

        save_image(
            image,
            output_path
        )

        class_counters[class_name] += 1

        total_test_images += 1

    print(
        f"Total test images: "
        f"{total_test_images}"
    )


# ============================================================
# SAVE DATASET INFORMATION
# ============================================================

def save_dataset_info():

    dataset_info = {

        "dataset_name": "CIFAR-100",

        "selected_classes": SELECTED_CLASSES,

        "number_of_classes": len(
            SELECTED_CLASSES
        ),

        "random_seed": SEED,

        "validation_ratio": VALIDATION_RATIO,

        "image_format": "PNG",

        "description":
            "30-class image classification "
            "dataset created from CIFAR-100.",

        "directories": {
            "train": "dataset/train",
            "validation": "dataset/validation",
            "test": "dataset/test"
        }

    }

    output_file = (
        DATASET_DIR / "dataset_info.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dataset_info,
            file,
            indent=4
        )

    print()
    print(
        f"Dataset information saved to:"
    )

    print(
        output_file
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("VISIONPREDICT AI - DATASET BUILDER")
    print("=" * 60)

    print()
    print(
        f"Selected classes: "
        f"{len(SELECTED_CLASSES)}"
    )

    train_dataset, test_dataset = (
        download_dataset()
    )

    create_class_directories()

    prepare_training_data(
        train_dataset
    )

    prepare_test_data(
        test_dataset
    )

    save_dataset_info()

    print()
    print("=" * 60)
    print("DATASET CREATION COMPLETED")
    print("=" * 60)

    print()
    print(
        f"Dataset location:"
    )

    print(
        DATASET_DIR
    )

    print()
    print(
        "Classes:"
    )

    for number, class_name in enumerate(
        SELECTED_CLASSES,
        start=1
    ):

        print(
            f"{number:02d}. {class_name}"
        )

    print()


if __name__ == "__main__":

    main()