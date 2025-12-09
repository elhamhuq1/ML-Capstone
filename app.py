"""
Nutrition Receipt Analyzer - Flask Web Application

A web interface for analyzing grocery receipts and providing
nutritional insights and healthier swap suggestions.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime

# Import our analysis pipeline
# Note: This requires running the notebook first to set up all functions
import sys
sys.path.append('.')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_demo_results(filepath, filename):
    """Generate demo results for UI testing (when pipeline not available)"""
    return {
        'success': True,
        'demo_mode': True,
        'receipt_name': filename,
        'num_items': 3,
        'items_with_macros': [
            {
                'original_name': 'BANANAS',
                'matched_food': 'Bananas, raw',
                'confidence': 0.85,
                'calories': 89,
                'protein_g': 1.1,
                'carbs_g': 22.8,
                'fat_g': 0.3,
                'fiber_g': 2.6
            },
            {
                'original_name': 'WHOLE MILK',
                'matched_food': 'Milk, whole, 3.25% milkfat',
                'confidence': 0.92,
                'calories': 61,
                'protein_g': 3.2,
                'carbs_g': 4.8,
                'fat_g': 3.3,
                'fiber_g': 0.0
            },
            {
                'original_name': 'WHITE BREAD',
                'matched_food': 'Bread, white, commercially prepared',
                'confidence': 0.78,
                'calories': 265,
                'protein_g': 9.0,
                'carbs_g': 49.0,
                'fat_g': 3.2,
                'fiber_g': 2.7
            }
        ],
        'macros_summary': {
            'calories': 415,
            'protein_g': 13.3,
            'carbs_g': 76.6,
            'fat_g': 6.8,
            'fiber_g': 5.3
        },
        'swap_suggestions': [
            {
                'item': 'WHOLE MILK',
                'formatted': '''🔄 **Healthier Swap Suggestion**

Consider switching to skim milk or 1% milk! You'll get the same protein and calcium with significantly less saturated fat, helping support your heart health while keeping your bones strong.

📊 **Nutritional Details:**
• Original: Milk, whole, 3.25% milkfat
• Alternative: Milk, nonfat, fluid

**Key Changes (per 100g):**
• Calories: -27 kcal
• Fat: -3.2g
• Fiber: ~same'''
            }
        ],
        'num_swaps': 1,
        'processing_time': 2.5
    }

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_receipt():
    """Handle receipt upload and analysis"""
    # Check if file was uploaded
    if 'receipt' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['receipt']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save file
        filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Use the real pipeline
            try:
                from pipeline import analyze_receipt
                result = analyze_receipt(filepath, verbose=False)
            except Exception as e:
                # Fallback to demo mode if pipeline fails
                print(f"Pipeline error: {e}")
                result = generate_demo_results(filepath, filename)
            
            if result['success']:
                # Format result for frontend
                response = {
                    'success': True,
                    'receipt_name': result['receipt_name'],
                    'image_url': f'/uploads/{filename}',
                    'num_items': result.get('num_items', 0),
                    'items': result.get('items_with_macros', []),
                    'macros': result.get('macros_summary', {}),
                    'swaps': result.get('swap_suggestions', []),
                    'processing_time': round(result.get('processing_time', 0), 2),
                    'demo_mode': result.get('demo_mode', False)
                }
                return jsonify(response)
            else:
                return jsonify({'error': result.get('error', 'Analysis failed')}), 500
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Nutrition Receipt Analyzer API is running'})

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Nutrition Receipt Analyzer - Web App")
    print("=" * 70)
    
    # Try to pre-load the pipeline
    try:
        print("\nPre-loading analysis pipeline...")
        from pipeline import analyze_receipt
        print("[OK] Pipeline loaded successfully!")
        print("\nServer will use LIVE ANALYSIS")
    except Exception as e:
        print(f"\n[WARNING] Pipeline load failed: {e}")
        print("Server will run in DEMO MODE")
    
    print("\nStarting Flask server at: http://localhost:5000")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

