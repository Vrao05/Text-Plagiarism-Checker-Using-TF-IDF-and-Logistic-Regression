# Text-Plagiarism-Checker-Using-TF-IDF-and-Logistic-Regression
# Plagiarism Detection System

This repository contains a Python-based Flask web application for detecting plagiarism between text files using TF-IDF vectorization and logistic regression. The project is designed to compare multiple text files, identify similarities, and flag plagiarized content.

## Features

- **Upload Multiple Files**: Allows users to upload multiple text files at once.
- **TF-IDF Vectorization**: Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) to represent text data in a numerical form.
- **Cosine Similarity**: Measures the cosine similarity between document pairs to detect potential plagiarism.
- **Logistic Regression Model**: Trains a logistic regression model to classify whether pairs of documents are plagiarized or not.
- **Plagiarism Report**: Provides a summary of the files that are detected as plagiarized along with their corresponding similar file and similarity percentage.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Vrao05/Text-Plagiarism-Checker-Using-TF-IDF-and-Logistic-Regression.git
    cd plagiarism-detection
    ```

2. **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up Ngrok (optional):**
    - This project uses Ngrok to expose the Flask app to the internet.
    - Sign up for an Ngrok account and follow their installation instructions.
    - You can start Ngrok with the command:
        ```bash
        ngrok http 5000
        ```

4. **Run the Flask app:**
    ```bash
    python app.py
    ```

5. **Access the web app:**
    - Visit `http://localhost:5000` to use the plagiarism detection tool.
    - If using Ngrok, use the provided public URL.

## Usage

1. **Upload Files**: Go to the main page of the web app and upload the text files you want to check for plagiarism.
2. **View Results**: After the files are processed, you will be redirected to a results page showing which files are plagiarized and their similarity percentages.
3. **Review Files**: Plagiarized files will be annotated with a message in the text, indicating that they are detected as plagiarized.

## Project Structure

- `app.py`: Main Flask application.
- `PlagiarismDetector.py`: The core class for handling file uploads, vectorization, model training, and prediction.
- `templates/`: Contains HTML templates for the web application.
- `static/`: Contains static files like CSS, JS, etc.
- `requirements.txt`: List of required Python packages.

## Flowchart

Below is a flowchart outlining the process flow of the plagiarism detection system.

```mermaid
graph TD;
    A[Start] --> B[Upload Files];
    B --> C[TF-IDF Vectorization];
    C --> D[Calculate Cosine Similarity];
    D --> E[Train Logistic Regression Model];
    E --> F[Predict Plagiarism];
    F --> G[Generate Report];
    G --> H[Display Results];
    H --> I[End];
