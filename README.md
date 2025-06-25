# 📝 Handwritten Text Recognition with OCR and Grammar Correction

This project is an end-to-end **Handwritten Text Recognition (HTR)** application built with **Streamlit**. It uses **YOLO-based object detection** models for word and character recognition, combined with **custom rule-based grammar correction** and **intelligent spell-checking**. The app allows users to upload images containing handwritten content and converts them into accurately transcribed and corrected digital text.

---

## 🔧 Features

- **Word Segmentation** using YOLO models
- **Character Recognition** with character mapping
- **OCR Post-processing** with rule-based corrections for common handwritten errors
- **Advanced Spell Checking** using `pyspellchecker` with support for custom dictionaries
- **Side-by-Side Visual Comparison** of original and corrected outputs
- **User-Friendly Interface** built with Streamlit
- **Downloadable Results** for both raw and corrected text
- **Debug Mode & Custom Dictionary Input**

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python (OpenCV, NumPy, regex, SpellChecker)
- **Models**: YOLOv8 (for both word and character detection)
- **Libraries**: OpenCV, Matplotlib, PIL, PySpellChecker, Ultralytics

---

## 🚀 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/JuhiSharma0510/Handwritten_Text_Recognition.git
   cd Handwritten_Text_Recognition
   
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
   
4. **Add your trained YOLO models**
    ```bash
    Place word_model.pt and char_model.pt in the project root.
    Or use the built-in mock models for demo purposes.

6. **Run the app**
    ```bash
    streamlit run app.py
   
📂 Project Structure

.
├── app.py                # Main Streamlit application

├── word_model.pt         # YOLO model for word segmentation

├── char_model.pt         # YOLO model for character recognition

├── requirements.txt      # Python dependencies

└── README.md             # Project documentation

📌 Use Cases
Digitizing handwritten notes or documents

Educational tools for handwriting transcription

Pre-processing step for NLP on handwritten datasets

OCR enhancement with linguistic accuracy

🙋‍♀️ Author

Juhi Sharma

📄 License

This project is licensed under the MIT License.
