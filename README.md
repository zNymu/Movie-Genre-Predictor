# Movie Genre Predictor

This application uses machine learning to predict movie genres based on plot summaries. It demonstrates text classification using TF-IDF features and multi-label classification.

![Screenshot of the application](screenshots/app_screenshot.png)

## Features

- Multi-label genre prediction from textual movie plots
- TF-IDF vectorization for text feature extraction
- Logistic Regression with OneVsRest classification
- Interactive GUI with confidence threshold adjustment
- Color-coded prediction confidence visualization

## Setup Instructions

### Requirements
- Python 3.7+
- Required packages: pandas, scikit-learn, numpy, tkinter (usually included with Python)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/movie-genre-predictor.git
cd movie-genre-predictor
pip install pandas scikit-learn numpy
```

2. Ensure you have the dataset file:
   - The training script expects a file at `data/wiki_movie_plots_deduped.csv`
   - This file should contain at least 'Plot' and 'Genre' columns

### Running the Application

#### Step 1: Train the Model
First, you need to train the model with your dataset:

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train a multi-label classification model
- Save the trained model, vectorizer, and genre encoder to the `models/` directory

#### Step 2: Run the Application
After the model is trained, you can run the application:

```bash
python genre_predictor_app.py
```

Or for the enhanced version:

```bash
python enhanced_genre_predictor_app.py
```

## Application Features

### `genre_predictor_app.py`
- Modern, stylish UI
- Loading screen while the model initializes
- Adjustable confidence threshold slider
- Visual confidence indicators (color-coded)
- Example placeholder text
- Responsive design

## How It Works

1. The application loads a pre-trained machine learning model
2. When you enter a movie plot and click "Predict Genres":
   - The text is processed using TF-IDF vectorization
   - The model predicts genre probabilities
   - Genres above the confidence threshold are displayed
3. Results show each predicted genre with its confidence score

## Troubleshooting

- **"Models not found" error**: Make sure you run `train_model.py` before launching the app
- **Unicode errors**: If you see encoding errors, try running in a terminal with UTF-8 support
- **Performance issues**: For very large datasets, the training process may take time

## Customization

You can adjust several parameters:
- In `train_model.py`: Change model parameters, feature selection, or preprocessing steps
- In `predict_genre.py`: Modify the prediction threshold or output format
- In the app file: Customize the UI appearance and behavior

## Files Included

- `train_model.py`: Script for training the genre prediction model
- `predict_genre.py`: Core prediction functionality
- `genre_predictor_app.py`: The GUI application
