// ============================================================
// VISIONPREDICT AI
// Image Upload & Prediction Interface
// ============================================================


// ============================================================
// DOM ELEMENTS
// ============================================================

const uploadArea = document.getElementById("uploadArea");

const chooseButton = document.getElementById("chooseButton");

const imageInput = document.getElementById("imageInput");

const previewContainer =
    document.getElementById("previewContainer");

const imagePreview =
    document.getElementById("imagePreview");

const fileName =
    document.getElementById("fileName");

const removeButton =
    document.getElementById("removeButton");

const predictButton =
    document.getElementById("predictButton");

const buttonText =
    document.getElementById("buttonText");

const loadingSpinner =
    document.getElementById("loadingSpinner");

const resultCard =
    document.getElementById("resultCard");

const predictionLabel =
    document.getElementById("predictionLabel");

const confidenceValue =
    document.getElementById("confidenceValue");

const confidenceProgress =
    document.getElementById("confidenceProgress");

const alternativeList =
    document.getElementById("alternativeList");

const errorMessage =
    document.getElementById("errorMessage");


// ============================================================
// APPLICATION STATE
// ============================================================

let selectedFile = null;


// ============================================================
// OPEN FILE SELECTOR
// ============================================================

chooseButton.addEventListener(
    "click",
    () => {

        imageInput.click();

    }
);


// ============================================================
// FILE INPUT CHANGE
// ============================================================

imageInput.addEventListener(
    "change",
    (event) => {

        const file =
            event.target.files[0];

        if (file) {

            handleFile(file);

        }

    }
);


// ============================================================
// HANDLE SELECTED FILE
// ============================================================

function handleFile(file) {

    clearError();

    // Validate file type
    const allowedTypes = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ];

    if (!allowedTypes.includes(file.type)) {

        showError(
            "Please upload a JPG, PNG, or WEBP image."
        );

        resetFileInput();

        return;
    }


    // Validate file size
    const maxSize =
        10 * 1024 * 1024;

    if (file.size > maxSize) {

        showError(
            "Image size must be smaller than 10 MB."
        );

        resetFileInput();

        return;
    }


    selectedFile = file;


    // Create image preview
    const reader =
        new FileReader();


    reader.onload = function (event) {

        imagePreview.src =
            event.target.result;

        previewContainer.hidden =
            false;

        uploadArea.hidden =
            true;

        predictButton.disabled =
            false;

        resultCard.hidden =
            true;

        fileName.textContent =
            file.name;

    };


    reader.readAsDataURL(file);
}


// ============================================================
// DRAG & DROP
// ============================================================

uploadArea.addEventListener(
    "dragover",
    (event) => {

        event.preventDefault();

        uploadArea.classList.add(
            "drag-over"
        );

    }
);


uploadArea.addEventListener(
    "dragleave",
    () => {

        uploadArea.classList.remove(
            "drag-over"
        );

    }
);


uploadArea.addEventListener(
    "drop",
    (event) => {

        event.preventDefault();

        uploadArea.classList.remove(
            "drag-over"
        );


        const file =
            event.dataTransfer.files[0];


        if (file) {

            handleFile(file);

        }

    }
);


// ============================================================
// REMOVE IMAGE
// ============================================================

removeButton.addEventListener(
    "click",
    () => {

        resetApplication();

    }
);


// ============================================================
// PREDICT IMAGE
// ============================================================

predictButton.addEventListener(
    "click",
    async () => {

        if (!selectedFile) {

            showError(
                "Please select an image first."
            );

            return;
        }


        clearError();

        setLoading(true);


        try {

            const formData =
                new FormData();


            formData.append(
                "file",
                selectedFile
            );


            const response =
                await fetch(
                    "/api/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Prediction failed."
                );

            }


            displayPrediction(
                data
            );

        }

        catch (error) {

            console.error(
                "Prediction error:",
                error
            );


            showError(
                error.message ||
                "Something went wrong while predicting the image."
            );

        }

        finally {

            setLoading(false);

        }

    }
);


// ============================================================
// DISPLAY PREDICTION
// ============================================================

function displayPrediction(data) {

    if (
        !data.prediction ||
        !data.prediction.top_prediction
    ) {

        showError(
            "The API returned an invalid prediction."
        );

        return;
    }


    const topPrediction =
        data.prediction.top_prediction;


    const alternatives =
        data.prediction.alternatives ||
        [];


    // Prediction label
    predictionLabel.textContent =
        formatLabel(
            topPrediction.label
        );


    // Confidence
    const confidence =
        Number(
            topPrediction.confidence_percentage
        );


    confidenceValue.textContent =
        `${confidence.toFixed(2)}%`;


    // Animate confidence bar
    confidenceProgress.style.width =
        "0%";


    resultCard.hidden =
        false;


    setTimeout(
        () => {

            confidenceProgress.style.width =
                `${confidence}%`;

        },
        50
    );


    // Clear old alternatives
    alternativeList.innerHTML =
        "";


    // Add alternatives
    alternatives.forEach(
        (item) => {

            const alternative =
                document.createElement(
                    "div"
                );


            alternative.className =
                "alternative-item";


            const name =
                document.createElement(
                    "span"
                );


            name.className =
                "alternative-name";


            name.textContent =
                formatLabel(
                    item.label
                );


            const confidenceText =
                document.createElement(
                    "span"
                );


            confidenceText.className =
                "alternative-confidence";


            confidenceText.textContent =
                `${Number(
                    item.confidence_percentage
                ).toFixed(2)}%`;


            alternative.appendChild(
                name
            );


            alternative.appendChild(
                confidenceText
            );


            alternativeList.appendChild(
                alternative
            );

        }
    );


    // Scroll to result
    resultCard.scrollIntoView(
        {
            behavior: "smooth",
            block: "center"
        }
    );
}


// ============================================================
// FORMAT LABEL
// ============================================================

function formatLabel(label) {

    if (!label) {

        return "Unknown";

    }


    return label
        .replaceAll("_", " ")
        .replace(/\b\w/g, char =>
            char.toUpperCase()
        );

}


// ============================================================
// LOADING STATE
// ============================================================

function setLoading(isLoading) {

    if (isLoading) {

        predictButton.disabled =
            true;

        buttonText.textContent =
            "Analyzing Image...";

        loadingSpinner.hidden =
            false;

    }

    else {

        predictButton.disabled =
            false;

        buttonText.textContent =
            "Predict Image";

        loadingSpinner.hidden =
            true;

    }

}


// ============================================================
// SHOW ERROR
// ============================================================

function showError(message) {

    errorMessage.textContent =
        message;

    errorMessage.hidden =
        false;


    errorMessage.scrollIntoView(
        {
            behavior: "smooth",
            block: "center"
        }
    );

}


// ============================================================
// CLEAR ERROR
// ============================================================

function clearError() {

    errorMessage.textContent =
        "";

    errorMessage.hidden =
        true;

}


// ============================================================
// RESET FILE INPUT
// ============================================================

function resetFileInput() {

    imageInput.value =
        "";

}


// ============================================================
// RESET APPLICATION
// ============================================================

function resetApplication() {

    selectedFile =
        null;


    resetFileInput();


    imagePreview.src =
        "";


    fileName.textContent =
        "";


    previewContainer.hidden =
        true;


    uploadArea.hidden =
        false;


    predictButton.disabled =
        true;


    resultCard.hidden =
        true;


    confidenceProgress.style.width =
        "0%";


    alternativeList.innerHTML =
        "";


    clearError();

}