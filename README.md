# CIFAR-10 Image Classification Web App

This is a Flask-based web application that uses a trained Deep Learning model to classify images into one of the 10 CIFAR-10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

## Features

- **Drag and Drop Interface**: Easily upload images for classification.
- **Top 3 Predictions**: Displays the highest confidence predictions.
- **Responsive UI**: A premium, modern frontend experience.
- **Robust Error Handling**: Handles corrupted images, empty uploads, and oversized files safely.
- **Production Ready**: Configured for WSGI deployment using gunicorn.

## Setup Instructions

### 1. Create a virtual environment and install dependencies

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment Variables
Copy `.env.example` to `.env` and configure your settings.
```bash
cp .env.example .env
```

### 3. Run the application locally
```bash
python run.py
```
Open `http://127.0.0.1:5000/` in your browser.

## Running Tests
Run the test suite using pytest:
```bash
pytest
```

## Production Deployment
This application includes a `Procfile` and uses `gunicorn` for deployment on platforms like Heroku or Render.

```bash
gunicorn run:app
```
