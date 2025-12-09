# 🌐 Flask Web App - Quick Start Guide

This guide will help you launch the Nutrition Receipt Analyzer web application.

## 🎯 Prerequisites

1. **Jupyter Notebook completed**: You must run `nutrition_receipt_analyzer.ipynb` first to set up all functions
2. **Python 3.8+** installed
3. **Dependencies** installed: `pip install -r requirements.txt`
4. **Tesseract OCR** installed on your system

## 🚀 Quick Start

### Option 1: Simple Approach (Recommended)

**Step 1:** Run the complete notebook first
```bash
# Open and run all cells
jupyter notebook nutrition_receipt_analyzer.ipynb
```

**Step 2:** Convert notebook to Python script
```bash
# This creates nutrition_receipt_analyzer.py with all functions
jupyter nbconvert --to script nutrition_receipt_analyzer.ipynb
```

**Step 3:** Start the Flask server
```bash
python app.py
```

**Step 4:** Open your browser
```
http://localhost:5000
```

### Option 2: Development Mode

If you're actively developing, keep the notebook open and use it as the backend:

```bash
# Run Flask in one terminal
python app.py

# Keep Jupyter running in another terminal  
jupyter notebook
```

## 📱 Using the Web App

1. **Upload**: Drag & drop a receipt image or click to browse
2. **Wait**: Processing takes 3-10 seconds depending on your CPU
3. **View Results**:
   - Macronutrient summary (calories, protein, carbs, fat)
   - Detailed items table with confidence scores
   - Healthier swap suggestions with AI-generated text
   - Visualizations (pie chart of macro breakdown)

## 🎨 Features

✅ **Drag & Drop Interface**: Easy receipt upload  
✅ **Real-time Analysis**: See progress while processing  
✅ **Beautiful Dashboard**: Clean, modern UI with Bootstrap  
✅ **Macro Visualizations**: Pie charts and stat cards  
✅ **Confidence Indicators**: Color-coded badges show matching quality  
✅ **Swap Suggestions**: AI-generated friendly recommendations  
✅ **Responsive Design**: Works on desktop, tablet, and mobile  

## 🔧 Troubleshooting

### "Analysis pipeline not initialized"
**Problem**: Notebook hasn't been run or converted

**Solution**:
```bash
# Run these commands in order:
jupyter notebook nutrition_receipt_analyzer.ipynb  # Run all cells
jupyter nbconvert --to script nutrition_receipt_analyzer.ipynb
python app.py
```

### Flask won't start
**Problem**: Port 5000 already in use

**Solution**: Change port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

### Upload fails
**Problem**: File too large or wrong format

**Solution**: 
- Max file size: 16MB
- Supported formats: JPG, PNG, JPEG
- Ensure image is clear and readable

### Slow processing
**Problem**: OCR and LLM are CPU-intensive

**Expected**: 3-5 seconds per receipt on modern CPUs

**Tips**:
- Disable LLM for faster processing (set `llm_available = False` in notebook)
- Use smaller images (resize to max 2000px width)
- Close other applications

## 📂 Project Structure

```
capstone-project/
├── app.py                    # Flask application
├── notebook_bridge.py        # Bridge between Flask and notebook functions
├── nutrition_receipt_analyzer.ipynb  # Main notebook (run first!)
├── nutrition_receipt_analyzer.py     # Auto-generated from notebook
├── templates/
│   └── index.html           # Web interface
├── uploads/                 # Uploaded receipt images (auto-created)
└── requirements.txt         # Python dependencies
```

## 🎓 Architecture

```
User Browser
    ↓ (upload receipt)
Flask App (app.py)
    ↓ (calls)
Notebook Bridge (notebook_bridge.py)
    ↓ (imports)
Analysis Pipeline (nutrition_receipt_analyzer.py)
    ↓ (uses)
- Tesseract OCR
- USDA Database
- Sentence-BERT
- Logistic Regression Re-ranker
- Qwen LLM (optional)
    ↓ (returns)
JSON Results → Beautiful Dashboard
```

## 🌟 API Endpoints

### `GET /`
Main page with upload interface

### `POST /upload`
Upload and analyze receipt

**Request**: Multipart form data with `receipt` file

**Response**:
```json
{
  "success": true,
  "receipt_name": "receipt.jpg",
  "num_items": 15,
  "items": [...],
  "macros": {
    "calories": 1500,
    "protein_g": 60,
    "carbs_g": 200,
    "fat_g": 50
  },
  "swaps": [...],
  "processing_time": 3.45
}
```

### `GET /health`
Health check endpoint

## 🚀 Deployment (Optional)

For production deployment:

1. **Disable debug mode** in `app.py`:
   ```python
   app.run(debug=False)
   ```

2. **Use production server** (e.g., Gunicorn):
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Add reverse proxy** (e.g., Nginx) for HTTPS

4. **Set environment variables**:
   ```bash
   export FLASK_ENV=production
   export SECRET_KEY=your-secret-key
   ```

## 💡 Tips for Best Results

1. **Clear images**: Ensure receipt is well-lit and not blurry
2. **Flat receipts**: Avoid wrinkled or folded receipts
3. **Standard stores**: Major chains have more consistent formatting
4. **Recent receipts**: Thermal receipts fade over time

## 📊 Performance

- **Processing time**: 3-10 seconds per receipt
- **Accuracy**: 65% OCR detection, 31% food matching (with current training data)
- **Memory**: ~2GB with LLM, ~500MB without
- **Concurrent users**: Supports 1-5 users on local machine

## 🎉 Success!

If you see the upload interface at `http://localhost:5000`, you're ready to go!

Upload a receipt and watch the magic happen! 🍎✨

---

**Questions or issues?** Check the main `README.md` or review the notebook documentation.

