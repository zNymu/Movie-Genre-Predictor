import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import os
from predictor import load_models, predict_genre
import threading

class EnhancedMovieGenrePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Movie Genre Predictor")
        self.root.geometry("700x600")
        self.root.minsize(600, 500)
        self.configure_styles()
        self.create_loading_screen()
        self.load_thread = threading.Thread(target=self.load_models_thread)
        self.load_thread.daemon = True
        self.load_thread.start()

    def configure_styles(self):
        self.bg_color = "#f5f5f5"
        self.accent_color = "#3498db"
        self.success_color = "#2ecc71"
        self.warning_color = "#e74c3c"
        self.root.configure(bg=self.bg_color)

        style = ttk.Style()
        style.configure("TFrame", background=self.bg_color)
        style.configure("TLabel", background=self.bg_color, font=("Arial", 10))
        style.configure("Header.TLabel", font=("Arial", 16, "bold"))
        style.configure("Subheader.TLabel", font=("Arial", 12))
        style.configure("TButton", font=("Arial", 10), background=self.accent_color)
        style.map("TButton", background=[("active", self.accent_color)])

        style.configure(
            "Accent.TButton",
            background=self.accent_color,
            foreground="black",
            padding=10
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#2980b9")],
            foreground=[("active", "black")]
        )

    def create_loading_screen(self):
        self.loading_frame = ttk.Frame(self.root)
        self.loading_frame.pack(fill=tk.BOTH, expand=True)

        loading_label = ttk.Label(
            self.loading_frame,
            text="Loading AI Model...",
            style="Header.TLabel"
        )
        loading_label.pack(pady=(200, 20))

        self.progress = ttk.Progressbar(
            self.loading_frame,
            orient="horizontal",
            length=400,
            mode="indeterminate"
        )
        self.progress.pack(pady=20)
        self.progress.start(10)

        loading_info = ttk.Label(
            self.loading_frame,
            text="Please wait while we load the genre prediction model.",
            wraplength=400
        )
        loading_info.pack(pady=10)

    def load_models_thread(self):
        """Load models in background thread"""
        self.model, self.tfidf, self.mlb = load_models()
        self.root.after(0, self.finish_loading)

    def finish_loading(self):
        """Called when model loading is complete"""
        if not all([self.model, self.tfidf, self.mlb]):
            messagebox.showerror(
                "Error",
                "Failed to load prediction models.\nPlease make sure you've run train_model.py first!"
            )
            self.root.destroy()
            return

        self.progress.stop()
        self.loading_frame.destroy()

        self.create_widgets()

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root, padding=15)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))

        header_label = ttk.Label(
            header_frame,
            text="AI Movie Genre Predictor",
            style="Header.TLabel"
        )
        header_label.pack(anchor=tk.W)

        subheader_label = ttk.Label(
            header_frame,
            text="Enter a movie plot and the AI will predict its genres",
            style="Subheader.TLabel"
        )
        subheader_label.pack(anchor=tk.W, pady=(5, 0))

        input_frame = ttk.Frame(self.main_frame)
        input_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        input_label = ttk.Label(input_frame, text="Movie Plot Summary:")
        input_label.pack(anchor=tk.W, pady=(0, 5))

        self.plot_input = scrolledtext.ScrolledText(
            input_frame,
            wrap=tk.WORD,
            height=10,
            font=("Arial", 11),
            padx=8,
            pady=8
        )
        self.plot_input.pack(fill=tk.BOTH, expand=True)

        example_text = "Example: A young farm boy discovers his destiny when he meets a wise old mentor who teaches him about a mystical energy that connects all living things."
        self.plot_input.insert(tk.END, example_text)
        self.plot_input.bind("<FocusIn>", self.clear_placeholder)
        self.plot_has_placeholder = True

        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill=tk.X, pady=15)

        self.predict_button = ttk.Button(
            button_frame,
            text="Predict Genres",
            command=self.predict,
            style="Accent.TButton"
        )
        self.predict_button.pack(side=tk.LEFT, padx=(0, 10))

        self.clear_button = ttk.Button(
            button_frame,
            text="Clear",
            command=self.clear_fields
        )
        self.clear_button.pack(side=tk.LEFT)

        threshold_frame = ttk.Frame(button_frame)
        threshold_frame.pack(side=tk.RIGHT)

        threshold_label = ttk.Label(threshold_frame, text="Confidence Threshold:")
        threshold_label.pack(side=tk.LEFT, padx=(0, 5))

        self.threshold_var = tk.DoubleVar(value=0.15)
        self.threshold_scale = ttk.Scale(
            threshold_frame,
            from_=0.05,
            to=0.50,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            length=100
        )
        self.threshold_scale.pack(side=tk.LEFT, padx=(0, 5))

        self.threshold_value_label = ttk.Label(threshold_frame, text="0.15")
        self.threshold_value_label.pack(side=tk.LEFT)

        self.threshold_var.trace_add("write", self.update_threshold_label)

        results_frame = ttk.Frame(self.main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        results_label = ttk.Label(results_frame, text="Predicted Genres:")
        results_label.pack(anchor=tk.W, pady=(0, 5))

        self.results_area = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            height=8,
            font=("Arial", 11),
            padx=8,
            pady=8,
            state=tk.DISABLED
        )
        self.results_area.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready to analyze movie plots")

        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(5, 2)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_threshold_label(self, *args):
        """Update the threshold value label when the slider changes"""
        value = self.threshold_var.get()
        self.threshold_value_label.config(text=f"{value:.2f}")

    def clear_placeholder(self, event):
        """Clear placeholder text when the user clicks in the text box"""
        if self.plot_has_placeholder:
            self.plot_input.delete("1.0", tk.END)
            self.plot_has_placeholder = False

    def predict(self):
        if self.plot_has_placeholder:
            self.plot_input.delete("1.0", tk.END)
            self.plot_has_placeholder = False
            return

        plot_summary = self.plot_input.get("1.0", tk.END).strip()

        if not plot_summary:
            messagebox.showinfo("Input Required", "Please enter a plot summary to predict genres.")
            return

        threshold = self.threshold_var.get()

        self.status_var.set("Analyzing plot...")
        self.predict_button.config(state=tk.DISABLED)
        self.root.update_idletasks()

        try:
            def prediction_thread():
                predictions = predict_genre(plot_summary, self.model, self.tfidf, self.mlb, threshold=threshold)
                self.root.after(0, lambda: self.update_results(predictions))

            thread = threading.Thread(target=prediction_thread)
            thread.daemon = True
            thread.start()

        except Exception as e:
            self.status_var.set("Error in prediction")
            self.predict_button.config(state=tk.NORMAL)
            messagebox.showerror("Prediction Error", f"An error occurred: {str(e)}")

    def update_results(self, predictions):
        """Update the results area with predictions"""
        self.predict_button.config(state=tk.NORMAL)
        self.results_area.config(state=tk.NORMAL)
        self.results_area.delete("1.0", tk.END)

        if isinstance(predictions[0], str):
            self.results_area.insert(tk.END, predictions[0])
        else:
            self.results_area.tag_config("high", foreground="green")
            self.results_area.tag_config("medium", foreground="orange")
            self.results_area.tag_config("low", foreground="red")
            indicator = "●"

            for i, (genre, score) in enumerate(predictions):
                if score < 0.3:
                    tag = "low"
                elif score < 0.6:
                    tag = "medium"
                else:
                    tag = "high"
                self.results_area.insert(tk.END, f"{indicator} {genre}: {score:.2f}\n", tag)

        self.results_area.config(state=tk.DISABLED)

        if isinstance(predictions[0], str):
            self.status_var.set(predictions[0])
        else:
            genre_count = len(predictions)
            genre_text = "genre" if genre_count == 1 else "genres"
            self.status_var.set(f"Analysis complete - Found {genre_count} {genre_text}")

    def clear_fields(self):
        self.plot_input.delete("1.0", tk.END)
        self.plot_has_placeholder = False

        self.results_area.config(state=tk.NORMAL)
        self.results_area.delete("1.0", tk.END)
        self.results_area.config(state=tk.DISABLED)

        self.status_var.set("Ready to analyze movie plots")


def main():
    if not os.path.exists('models/model.pkl'):
        print("Models not found. Please run train_model.py first to train the model.")
        messagebox.showerror(
            "Models Not Found",
            "Models not found.\nPlease run train_model.py first to train the model."
        )
        return

    root = tk.Tk()
    app = EnhancedMovieGenrePredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()