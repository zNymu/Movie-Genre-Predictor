import pickle
import numpy as np


def load_models():
    """Load the saved models and return them"""
    try:
        model = pickle.load(open('models/model.pkl', 'rb'))
        tfidf = pickle.load(open('models/tfidf.pkl', 'rb'))
        mlb = pickle.load(open('models/mlb.pkl', 'rb'))
        print("Models loaded successfully!")
        return model, tfidf, mlb
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None


def predict_genre(plot_summary, model, tfidf, mlb, threshold=0.2):
    """
    Predicts movie genres based on a plot summary.

    Args:
        plot_summary (str): The movie plot summary
        model: Trained classifier model
        tfidf: TF-IDF vectorizer
        mlb: MultiLabelBinarizer
        threshold (float): Confidence threshold (0-1)

    Returns:
        list: Predicted genres with confidence scores
    """
    if not plot_summary.strip():
        return ["Input is empty"]

    X = tfidf.transform([plot_summary])

    try:
        decision_scores = model.decision_function(X)[0]

        results = []
        for i, score in enumerate(decision_scores):
            prob = 1 / (1 + np.exp(-score))
            if prob >= threshold:
                genre = mlb.classes_[i]
                results.append((genre, float(prob)))

    except Exception as e:
        print(f"Decision function failed: {e}")

        try:
            y_pred = model.predict(X)[0]
            results = []

            for i, is_present in enumerate(y_pred):
                if is_present:
                    genre = mlb.classes_[i]
                    try:
                        proba_values = model.predict_proba(X)[i][0]
                        if len(proba_values) > 1:
                            confidence = proba_values[1]
                        else:
                            confidence = proba_values[0]
                    except:
                        confidence = 0.8

                    if confidence >= threshold:
                        results.append((genre, float(confidence)))
        except Exception as e2:
            print(f"Prediction method failed: {e2}")
            return ["Prediction failed"]

    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        return ["No genres met the confidence threshold"]

    return results


def main():
    """Main function to run the prediction interface"""
    print("\nMovie Genre Predictor")
    print("Enter a movie plot summary to predict its genres (or 'q' to quit)\n")

    model, tfidf, mlb = load_models()
    if not all([model, tfidf, mlb]):
        print("Failed to load models. Make sure you've run train_model.py first!")
        return

    while True:
        user_input = input("Plot summary: ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            break

        if not user_input.strip():
            print("Please enter a plot summary!")
            continue

        predictions = predict_genre(user_input, model, tfidf, mlb, threshold=0.15)

        if isinstance(predictions[0], str):
            print(predictions[0])
        else:
            print("\nPredicted genres:")
            for genre, score in predictions:
                print(f"- {genre}: {score:.2f}")
        print("\n---")

    print("Thanks for using the Movie Genre Predictor!")


if __name__ == "__main__":
    main()