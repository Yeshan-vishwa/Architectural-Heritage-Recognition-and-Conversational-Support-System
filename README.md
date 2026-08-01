# Architectural Heritage Recognition and Conversational Support System

An AI-based system that recognises architectural heritage elements from uploaded images and provides heritage-related information through a conversational question-answering interface.

The project combines:

- Computer vision image classification
- Transfer learning with EfficientNetB0
- Retrieval-Augmented Generation
- LangChain document retrieval
- Vector embeddings
- A Streamlit-based user interface

## Project Overview

Architectural heritage is important for understanding and preserving historical buildings and cultural identity. However, identifying architectural elements can require specialist knowledge.

This project provides an accessible AI system that allows users to:

1. Upload an image of an architectural heritage element.
2. Receive a predicted architectural class.
3. View the model confidence score.
4. Ask questions about architectural heritage.
5. Receive answers based on selected heritage documents.

The system combines a trained computer vision model with a document-based conversational assistant.

## Main Features

### Architectural Image Classification

The computer vision component:

- Accepts an uploaded architectural image.
- Preprocesses the image for the trained model.
- Uses an EfficientNetB0-based neural network.
- Predicts the architectural heritage class.
- Displays a confidence score for the prediction.

### Heritage Question-Answering System

The conversational component:

- Loads architectural heritage documents.
- Divides the documents into smaller text sections.
- Converts the text into vector embeddings.
- Stores the embeddings in an in-memory vector store.
- Retrieves information relevant to the user's question.
- Generates an answer using the retrieved information.

### Integrated User Interface

The system is designed to provide both features through a Streamlit interface:

- Image upload
- Image preview
- Heritage element prediction
- Confidence score
- Question input
- Document-based responses

## System Architecture

```text
User
  │
  ▼
Streamlit Web Interface
  │
  ├── Uploaded Image
  │       ▼
  │   Image Preprocessing
  │       ▼
  │   EfficientNetB0 Model
  │       ▼
  │   Predicted Class and Confidence
  │       │
  │       └──────────────────────────┐
  │                                  ▼
  └── User Question ─────────► LangChain Retrieval
                                     │
Heritage Documents                   │
        ▼                            │
nomic-embed-text                     │
        ▼                            │
InMemoryVectorStore ─────────────────┘
        │
        ▼
Retrieved Context
        │
        ▼
Language Model
        │
        ▼
Document-Based Answer
```

## Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB0
- Streamlit
- LangChain
- Ollama
- `nomic-embed-text`
- InMemoryVectorStore
- NumPy
- Pillow
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Dataset

The computer vision model was developed using the **Architectural Heritage Elements Image64 Dataset**.

Dataset source:

[Architectural Heritage Elements Image64 Dataset on Kaggle](https://www.kaggle.com/datasets/ikobzev/architectural-heritage-elements-image64-dataset)

The dataset contains images of different architectural heritage elements that can be used for multi-class image classification.

Please refer to the Kaggle dataset page for its licence, ownership information and usage conditions.

## Repository Structure

```text
Architectural-Heritage-Recognition-and-Conversational-Support-System/
│
├── heritage_documents_detailed/
│   └── Heritage information used by the retrieval system
│
├── best_heritage_model.keras
│   └── Saved trained computer vision model
│
├── heritage_classification.ipynb
│   └── Dataset preparation, model training and evaluation
│
├── part_c_integrated_system.ipynb
│   └── Integrated image-classification and conversational system
│
└── README.md
    └── Project documentation
```

## Notebook Descriptions

### `heritage_classification.ipynb`

This notebook contains the computer vision development process, including:

- Dataset loading
- Image preprocessing
- Training, validation and testing preparation
- Data augmentation
- EfficientNetB0 transfer learning
- Model training
- Performance evaluation
- Prediction testing
- Model saving

### `part_c_integrated_system.ipynb`

This notebook contains the integrated system, including:

- Loading the saved Keras model
- Image preprocessing
- Architectural heritage prediction
- Confidence-score display
- Heritage document loading
- Text splitting
- Embedding generation
- Vector storage
- Similarity-based retrieval
- Conversational response generation
- Streamlit interface development

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Yeshan-vishwa/Architectural-Heritage-Recognition-and-Conversational-Support-System.git
```

```bash
cd Architectural-Heritage-Recognition-and-Conversational-Support-System
```

### 2. Create a Virtual Environment

Using Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Using macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the Main Dependencies

```bash
pip install tensorflow streamlit langchain langchain-community langchain-ollama numpy pillow matplotlib scikit-learn jupyter
```

Depending on the installed LangChain version, additional LangChain packages may be required.

## Ollama Setup

The conversational component uses Ollama for local embedding and language-model support.

### 1. Install Ollama

Download and install Ollama from:

[https://ollama.com](https://ollama.com)

### 2. Download the Embedding Model

```bash
ollama pull nomic-embed-text
```

### 3. Download the Language Model

Download the language model referenced in the integrated notebook. For example:

```bash
ollama pull llama3.2
```

The model name in the notebook must match the model installed through Ollama.

### 4. Start Ollama

Make sure the Ollama service is running before using the conversational component.

```bash
ollama serve
```

## How to Run the Project

### Run the Computer Vision Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
heritage_classification.ipynb
```

Run the cells in order to review the dataset preparation, model training and evaluation process.

Training the model again may require significant time and computing resources.

### Run the Integrated System Notebook

Open:

```text
part_c_integrated_system.ipynb
```

Before running the notebook, confirm that:

- `best_heritage_model.keras` is available in the correct path.
- The `heritage_documents_detailed` folder is available.
- Ollama is installed and running.
- The required embedding model is installed.
- The required language model is installed.
- Any local file paths in the notebook have been updated.

Run the notebook cells in order.

If the Streamlit application is exported as a Python file, it can be started using:

```bash
streamlit run app.py
```

Replace `app.py` with the actual filename used for the Streamlit application.

## Computer Vision Workflow

The image-classification process follows these steps:

1. The user uploads an image.
2. The image is converted into a compatible colour format.
3. The image is resized to the model's required input size.
4. The image is converted into a numerical array.
5. The trained Keras model produces class probabilities.
6. The class with the highest probability is selected.
7. The predicted class and confidence score are displayed.

## Conversational Retrieval Workflow

The conversational component follows these steps:

1. Heritage documents are loaded from the document directory.
2. The text is divided into smaller chunks.
3. Each chunk is converted into a numerical embedding.
4. The embeddings are stored in an `InMemoryVectorStore`.
5. The user's question is converted into an embedding.
6. Similar document chunks are retrieved.
7. The retrieved text is supplied to the language model as context.
8. The system generates a response based on the available heritage documents.

## Model Evaluation

The computer vision model is evaluated using measures such as:

- Training accuracy
- Validation accuracy
- Test accuracy
- Training loss
- Validation loss
- Confusion matrix
- Classification report
- Precision
- Recall
- F1-score

The confidence value shown by the application represents the model's probability for the selected class. It should not be interpreted as a guarantee that the prediction is correct.

## Ethical Considerations

### Image Privacy

Users should avoid uploading private or sensitive images. Uploaded images should only be used for the requested prediction and should not be stored without a clear reason and user consent.

### Model Bias

The image-classification model may perform better for architectural styles that are well represented in the training dataset. It may be less accurate for:

- Rare architectural elements
- Regional styles not included in the dataset
- Damaged buildings
- Poor-quality images
- Images with unusual lighting or camera angles

### Hallucination Risk

The language model may generate incorrect or unsupported information. Retrieval reduces this risk but does not completely remove it.

Users should verify important historical information using trusted academic, heritage or government sources.

### Transparency

The system provides a confidence score to help users understand the uncertainty of image predictions. Conversational answers should remain connected to the retrieved heritage documents whenever possible.

## Limitations

- Prediction quality depends on image quality.
- The model is limited to the classes included in the training dataset.
- Visually similar architectural elements may be confused.
- The confidence score may not always reflect real-world correctness.
- The vector store is held in memory and may need to be rebuilt after restarting the system.
- The conversational assistant is limited by the supplied heritage documents.
- The language model may still produce inaccurate information.
- Local model execution may require sufficient memory and processing power.
- Some paths in the notebooks may need to be changed for a different computer.

## Future Improvements

Possible future improvements include:

- Increasing the size and diversity of the training dataset
- Supporting additional architectural heritage classes
- Improving prediction accuracy through model fine-tuning
- Adding model-explanation visualisations such as Grad-CAM
- Adding source citations to conversational answers
- Using a persistent vector database
- Improving confidence calibration
- Adding multilingual support
- Adding user feedback for incorrect predictions
- Deploying the complete system as a public web application
- Improving accessibility and mobile-device support
- Adding stronger validation for uploaded files
- Conducting testing with heritage specialists

## Intended Use

This system was developed for educational and research purposes.

It may be useful for:

- Students
- Researchers
- Tourists
- Heritage enthusiasts
- Museums
- Cultural organisations
- Heritage education projects

The system should be treated as a supporting tool rather than a replacement for professional architectural or historical assessment.

## Disclaimer

Predictions and generated answers may contain errors. The system should not be used as the only source for conservation decisions, legal decisions, historical authentication or professional heritage assessment.

For important decisions, consult qualified heritage professionals and reliable official sources.

## Author

**Yeshan Vishwa**

GitHub: [Yeshan-vishwa](https://github.com/Yeshan-vishwa)

## Acknowledgements

- The creator of the Architectural Heritage Elements Image64 Dataset
- TensorFlow and Keras
- The EfficientNet research and development community
- Streamlit
- LangChain
- Ollama
- Open-source machine-learning and heritage-information communities
