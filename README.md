# Nutrition Receipt Analyzer

An end-to-end machine learning pipeline that extracts nutritional information from grocery receipts using OCR, food matching, and provides healthier swap suggestions.

## 🎯 Features

- **OCR Text Extraction**: Extract text from receipt images using Tesseract
- **Intelligent Food Matching**: Match receipt items to USDA FoodData Central database using:
  - TF-IDF similarity (baseline)
  - Sentence embeddings (semantic matching)
  - Logistic regression re-ranker (supervised learning)
- **Macro Estimation**: Calculate calories, protein, carbs, fat, and fiber
- **Healthier Swaps**: Suggest better alternatives with LLM-generated friendly explanations
- **Comprehensive Evaluation**: OCR accuracy, matching metrics, confidence analysis

## 📋 Requirements

### System Requirements

- Python 3.8+
- Tesseract OCR (must be installed separately)

### Python Dependencies

```
pytesseract
pandas
numpy
scikit-learn
sentence-transformers
torch
transformers
pillow
matplotlib
seaborn
jupyter
```

## 🚀 Installation

### 1. Install Tesseract OCR

**Windows:**

```bash
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
# Install to: C:\Program Files\Tesseract-OCR\
```

**macOS:**

```bash
brew install tesseract
```

**Linux:**

```bash
sudo apt-get install tesseract-ocr
```

### 2. Install Python Dependencies

```bash
# Clone the repository
cd capstone-project

# Install dependencies
pip install -r requirements.txt
```

### 3. Download USDA Data

1. Visit [USDA FoodData Central](https://fdc.nal.usda.gov/download-datasets.html)
2. Download:
   - Foundation Foods CSV
   - SR Legacy CSV
3. Extract to `data/usda_fdc/`

### 4. Add Receipt Images

Place your receipt images (JPG, PNG) in `data/receipts/images/`

## 💻 Usage

### Jupyter Notebook (Main Analysis)

```bash
jupyter notebook nutrition_receipt_analyzer.ipynb
```

Run all cells in order. The notebook is organized into sections:

1. Setup and Data Loading
2. OCR and Text Extraction
3. Text Normalization and Parsing
4. Food Matching Pipeline
5. Macro Estimation
6. Healthier Swap Suggestions
7. Evaluation Suite
8. End-to-End Integration

### Python API

```python
# Analyze a single receipt
result = analyze_receipt('data/receipts/images/0.jpg')

if result['success']:
    print(f"Found {result['num_items']} items")
    print(f"Total calories: {result['macros_summary']['calories']:.0f} kcal")
    print(f"Healthier swaps: {result['num_swaps']}")
```

### Flask Web App

```bash
# Quick test (demo mode with mock data)
python app.py
# Visit http://localhost:5000

# Full mode (requires setup - see FLASK_APP_GUIDE.md)
# 1. Export notebook functions to pipeline_api.py
# 2. Run: python app.py
```

**See `FLASK_APP_GUIDE.md` for complete setup instructions!**

## 📊 Project Structure

```
capstone-project/
├── nutrition_receipt_analyzer.ipynb  # Main notebook
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── data/
│   ├── usda_fdc/                    # USDA FoodData Central CSVs
│   ├── receipts/                    # Receipt images and annotations
│   ├── ground_truth.json            # Annotated items for training
│   └── models/                      # Cached embedding models
├── results/                         # Evaluation reports
└── tasks/                           # Project task lists and PRD
```

## 🎓 Technical Details

### Pipeline Overview

1. **OCR**: Tesseract extracts text from receipt images
2. **Parsing**: Regex patterns extract item names, quantities, prices
3. **Food Matching**:
   - TF-IDF vectorization for text similarity
   - Sentence-BERT embeddings for semantic matching
   - Logistic regression re-ranker combines both scores
4. **Macro Calculation**: Match quantities to USDA per-100g nutrients
5. **Swap Generation**: Rule-based system suggests healthier alternatives
6. **LLM Rephrasing**: Qwen2.5-1.5B generates friendly explanations

### Key Technologies

- **OCR**: pytesseract, PIL
- **ML**: scikit-learn (TF-IDF, Logistic Regression)
- **Deep Learning**: sentence-transformers, transformers (Qwen2.5)
- **Data**: pandas, numpy
- **Viz**: matplotlib, seaborn

## 📈 Performance Metrics

- **OCR Detection Rate**: 65.2% (varies by receipt quality)
- **Food Matching Top-1 Accuracy**: 31.2% (16 annotated items)
- **Average Match Confidence**: 0.310 (with trained re-ranker)
- **Processing Time**: ~3.7s per receipt (CPU)

## 🔧 Troubleshooting

### Tesseract not found

```python
# Windows - manually set path in notebook:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Out of memory (LLM)

```python
# Disable LLM rephrasing to save memory:
llm_available = False
```

### Poor OCR results

- Ensure receipt image is clear and well-lit
- Try image preprocessing (contrast, denoising)
- Check Tesseract installation

## 🚧 Limitations

- OCR accuracy depends on receipt quality
- Brand names not in USDA database
- Unit conversion assumes standard serving sizes
- Re-ranker needs more training data (currently 16 items)

## 🔮 Future Improvements

1. **More Training Data**: Annotate 50-100 items for better re-ranker performance
2. **Image Preprocessing**: Enhance images before OCR (contrast, rotation, denoising)
3. **Brand Mapping**: Create database of store brands → generic USDA foods
4. **Better Unit Extraction**: Use ML model instead of regex
5. **Web App**: Deploy Flask app for easy access
6. **Mobile App**: React Native app for on-the-go scanning

## 👤 Author

**Elham**  
Capstone Project - Nutrition Receipt Analyzer

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- USDA FoodData Central for nutrition database
- Tesseract OCR team
- Sentence-Transformers project
- Qwen team for the LLM

---
