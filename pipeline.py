"""
Nutrition Receipt Analyzer - Complete Pipeline

This module loads all data and models, and provides the analyze_receipt()
function for the Flask web application.
"""

import os
import re
import time
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

print("=" * 70)
print("Initializing Nutrition Receipt Analyzer Pipeline...")
print("=" * 70)

# ============================================================================
# CONFIGURE TESSERACT (Windows)
# ============================================================================

if os.name == 'nt':
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(
            os.environ.get('USERNAME', ''))
    ]
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"[OK] Tesseract found at: {path}")
            break

# ============================================================================
# LOAD USDA FOODDATA CENTRAL
# ============================================================================

print("\n[1/4] Loading USDA FoodData Central database...")

def load_usda_data():
    """Load and preprocess USDA FoodData Central data"""
    usda_dir = 'data/usda_fdc'
    
    # Find food.csv files
    food_csv_paths = glob.glob(os.path.join(usda_dir, '**', 'food.csv'), recursive=True)
    
    if not food_csv_paths:
        raise FileNotFoundError(f"No food.csv found in {usda_dir}")
    
    # Load and combine all food.csv files
    food_dfs = []
    for path in food_csv_paths:
        df = pd.read_csv(path, low_memory=False)
        food_dfs.append(df)
    
    foods = pd.concat(food_dfs, ignore_index=True)
    
    # Find food_nutrient.csv files
    nutrient_csv_paths = glob.glob(os.path.join(usda_dir, '**', 'food_nutrient.csv'), recursive=True)
    
    nutrients_dfs = []
    for path in nutrient_csv_paths:
        df = pd.read_csv(path, low_memory=False)
        nutrients_dfs.append(df)
    
    nutrients = pd.concat(nutrients_dfs, ignore_index=True)
    
    # Filter for relevant food types
    relevant_types = ['sr_legacy_food', 'foundation_food']
    exclude_types = ['agricultural_acquisition', 'sub_sample_food']
    
    foods_filtered = foods[
        (foods['data_type'].isin(relevant_types)) |
        ((foods['data_type'] == 'foundation_food') & 
         (~foods['food_category_id'].isin([8, 9])))
    ].copy()
    
    # Get macro nutrients (per 100g)
    nutrient_mapping = {
        1008: 'calories',      # Energy (kcal)
        1003: 'protein_g',     # Protein
        1005: 'carbs_g',       # Carbohydrate
        1004: 'fat_g',         # Total lipid (fat)
        1079: 'fiber_g'        # Fiber, total dietary
    }
    
    # Pivot nutrients
    nutrients_filtered = nutrients[nutrients['nutrient_id'].isin(nutrient_mapping.keys())].copy()
    nutrients_pivot = nutrients_filtered.pivot_table(
        index='fdc_id',
        columns='nutrient_id',
        values='amount',
        aggfunc='first'
    ).reset_index()
    
    nutrients_pivot.columns = ['fdc_id'] + [
        nutrient_mapping.get(col, f'nutrient_{col}')
        for col in nutrients_pivot.columns if col != 'fdc_id'
    ]
    
    # Merge
    usda_foods = foods_filtered.merge(nutrients_pivot, on='fdc_id', how='left')
    usda_foods = usda_foods.fillna(0)
    
    # Normalize descriptions
    usda_foods['description_normalized'] = usda_foods['description'].apply(
        lambda x: normalize_text(str(x))
    )
    
    print(f"[OK] Loaded {len(usda_foods):,} food items")
    
    return usda_foods

def normalize_text(text):
    """Normalize text for matching"""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('*', '').replace('~', '').replace('^', '')
    return text

# Load data
usda_foods = load_usda_data()
usda_descriptions = usda_foods['description_normalized'].tolist()

# ============================================================================
# LOAD TF-IDF VECTORIZER
# ============================================================================

print("\n[2/4] Building TF-IDF index...")

tfidf_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 4),
    min_df=1,
    max_features=5000
)

usda_tfidf_matrix = tfidf_vectorizer.fit_transform(usda_descriptions)

print(f"[OK] TF-IDF index built ({usda_tfidf_matrix.shape[1]:,} features)")

# ============================================================================
# LOAD SENTENCE EMBEDDING MODEL
# ============================================================================

print("\n[3/4] Loading sentence embedding model...")

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("   Encoding USDA descriptions (this may take a minute)...")
usda_embeddings = embedding_model.encode(
    usda_descriptions,
    show_progress_bar=False,
    batch_size=256
)

print(f"[OK] Embeddings ready ({usda_embeddings.shape[0]:,} items)")

# ============================================================================
# OCR FUNCTIONS
# ============================================================================

def load_receipt_image(image_path):
    """Load a receipt image"""
    try:
        img = Image.open(image_path)
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def extract_text_from_receipt(image):
    """Extract text using Tesseract OCR"""
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

# ============================================================================
# TEXT PARSING
# ============================================================================

# Regex patterns
PRICE_PATTERN = re.compile(r'\$?\s*(\d+\.\d{2})')
QUANTITY_PATTERN = re.compile(r'(?<!\d)(\d+\.?\d*)\s*(lb|lbs|oz|kg|g|gal|each|ea|pack|ct)', re.IGNORECASE)

STORE_CODE_PATTERNS = [
    re.compile(r'\b\d{12,14}\b'),          # UPC codes
    re.compile(r'\b\d{8}\b'),              # Short codes
    re.compile(r'\b[A-Z]{1,3}\d+[A-Z]{0,2}\b'),  # Mixed codes
    re.compile(r'\s+[A-Z]{1,2}\s*$'),      # Trailing letters
]

COMMON_ABBREVIATIONS = {
    'gv': 'great value',
    'kf': '',
    'pkg': 'package',
    'org': 'organic',
}

def extract_price(text):
    """Extract price from text"""
    match = PRICE_PATTERN.search(text)
    return float(match.group(1)) if match else None

def remove_store_codes(text):
    """Remove store-specific codes and expand abbreviations"""
    text = normalize_text(text)
    for pattern in STORE_CODE_PATTERNS:
        text = pattern.sub('', text)
    
    words = text.split()
    cleaned_words = [COMMON_ABBREVIATIONS.get(word, word) for word in words]
    text = ' '.join(word for word in cleaned_words if word)
    
    text = re.sub(r'[%@/]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def parse_receipt_line(line):
    """Parse a single receipt line - UPDATED VERSION"""
    if not line or len(line.strip()) < 3:
        return None
    
    line = line.strip()
    
    # Skip non-item lines
    skip_keywords = ['total', 'subtotal', 'tax', 'change', 'cash', 'credit', 
                     'debit', 'balance', 'visa', 'mastercard', 'thank you',
                     'receipt', 'store', 'customer', 'cashier', 'discount',
                     'save', 'savings', 'coupon', 'card', 'member', 'rewards',
                     'special', 'net @', 'net@', 'loyalty']
    
    line_lower = line.lower()
    if any(keyword in line_lower for keyword in skip_keywords):
        return None
    
    # Skip weight lines (e.g., "0.778kq NET @ $5.99")
    if re.search(r'^\d+\.\d+\s*(kg|kq|k9|lb|lbs|oz)\s+(net|@)', line_lower):
        return None
    
    # Initialize result
    result = {
        'original_line': line,
        'item_name': None,
        'quantity': 1.0,
        'unit': 'ea',
        'total_price': None,
        'raw_text': line
    }
    
    # Extract price
    price = extract_price(line)
    if not price:
        return None
    
    # Extract item name (everything before price) - FIXED FOR DOLLAR SIGNS
    item_match = re.match(r'^(.+?)\s+\$?\s*\d+\.\d{2}', line)
    if item_match:
        result['item_name'] = item_match.group(1).strip()
        result['total_price'] = price
        return result
    
    return None

# ============================================================================
# FOOD MATCHING
# ============================================================================

def get_tfidf_matches(item_name, top_k=10):
    """Get top matches using TF-IDF"""
    item_vec = tfidf_vectorizer.transform([normalize_text(item_name)])
    scores = cosine_similarity(item_vec, usda_tfidf_matrix).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'fdc_id': usda_foods.iloc[idx]['fdc_id'],
            'description': usda_foods.iloc[idx]['description'],
            'tfidf_score': scores[idx]
        })
    return results

def get_embedding_matches(item_name, top_k=10):
    """Get top matches using sentence embeddings"""
    item_embedding = embedding_model.encode([normalize_text(item_name)])
    scores = cosine_similarity(item_embedding, usda_embeddings).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'fdc_id': usda_foods.iloc[idx]['fdc_id'],
            'description': usda_foods.iloc[idx]['description'],
            'embedding_score': scores[idx]
        })
    return results

def match_item_to_usda(item_name, top_k=5):
    """
    Match receipt item to USDA food (combined approach)
    Returns top-k matches with confidence scores
    """
    # Get candidates from both methods
    tfidf_results = get_tfidf_matches(item_name, top_k=10)
    embedding_results = get_embedding_matches(item_name, top_k=10)
    
    # Combine by averaging scores
    combined = {}
    for result in tfidf_results:
        fdc_id = result['fdc_id']
        combined[fdc_id] = {
            'fdc_id': fdc_id,
            'description': result['description'],
            'tfidf_score': result['tfidf_score'],
            'embedding_score': 0
        }
    
    for result in embedding_results:
        fdc_id = result['fdc_id']
        if fdc_id in combined:
            combined[fdc_id]['embedding_score'] = result['embedding_score']
        else:
            combined[fdc_id] = {
                'fdc_id': fdc_id,
                'description': result['description'],
                'tfidf_score': 0,
                'embedding_score': result['embedding_score']
            }
    
    # Calculate average score
    for fdc_id in combined:
        combined[fdc_id]['avg_score'] = (
            combined[fdc_id]['tfidf_score'] + 
            combined[fdc_id]['embedding_score']
        ) / 2
        combined[fdc_id]['confidence'] = combined[fdc_id]['avg_score']
    
    # Sort by average score
    sorted_matches = sorted(
        combined.values(),
        key=lambda x: x['avg_score'],
        reverse=True
    )
    
    return sorted_matches[:top_k]

# ============================================================================
# MACRO CALCULATION
# ============================================================================

UNIT_CONVERSIONS = {
    'g': 1.0,
    'kg': 1000.0,
    'lb': 453.592,
    'lbs': 453.592,
    'oz': 28.3495,
    'gal': 3785.0,  # Approximate for milk
    'ea': 100.0,    # Default estimate
    'each': 100.0,
}

def convert_to_grams(quantity, unit):
    """Convert quantity to grams"""
    unit_lower = unit.lower()
    factor = UNIT_CONVERSIONS.get(unit_lower, 100.0)
    return quantity * factor, 'medium' if unit_lower in UNIT_CONVERSIONS else 'low'

def get_nutrients_for_item(fdc_id):
    """Get per-100g nutrients for a USDA food item"""
    row = usda_foods[usda_foods['fdc_id'] == fdc_id]
    
    if row.empty:
        return None
    
    row = row.iloc[0]
    
    return {
        'calories': row.get('calories', 0),
        'protein_g': row.get('protein_g', 0),
        'carbs_g': row.get('carbs_g', 0),
        'fat_g': row.get('fat_g', 0),
        'fiber_g': row.get('fiber_g', 0)
    }

def calculate_item_macros(fdc_id, quantity, unit):
    """Calculate total macros for purchased quantity"""
    nutrients_per_100g = get_nutrients_for_item(fdc_id)
    
    if not nutrients_per_100g:
        return None
    
    grams, confidence = convert_to_grams(quantity, unit)
    scale_factor = grams / 100.0
    
    return {
        'calories': nutrients_per_100g['calories'] * scale_factor,
        'protein_g': nutrients_per_100g['protein_g'] * scale_factor,
        'carbs_g': nutrients_per_100g['carbs_g'] * scale_factor,
        'fat_g': nutrients_per_100g['fat_g'] * scale_factor,
        'fiber_g': nutrients_per_100g['fiber_g'] * scale_factor,
        'grams': grams,
        'unit_confidence': confidence
    }

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_receipt(image_path, verbose=False):
    """
    Complete end-to-end receipt analysis
    
    Returns:
        dict with keys: success, items, macros_summary, processing_time, etc.
    """
    start_time = time.time()
    
    result = {
        'success': False,
        'receipt_name': os.path.basename(image_path),
        'error': None
    }
    
    try:
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"ANALYZING: {os.path.basename(image_path)}")
            print('=' * 70)
        
        # 1. Load image
        if verbose:
            print("[1/5] Loading image...")
        image = load_receipt_image(image_path)
        if not image:
            raise Exception("Failed to load image")
        
        # 2. OCR
        if verbose:
            print("[2/5] Extracting text with OCR...")
        ocr_text = extract_text_from_receipt(image)
        
        # 3. Parse items
        if verbose:
            print("[3/5] Parsing receipt items...")
        
        parsed_items = []
        for line in ocr_text.split('\n'):
            parsed = parse_receipt_line(line)
            if parsed and parsed['item_name']:
                cleaned_name = remove_store_codes(parsed['item_name'])
                if cleaned_name and len(cleaned_name) > 2:
                    parsed['item_name_cleaned'] = cleaned_name
                    parsed_items.append(parsed)
        
        if verbose:
            print(f"   Found {len(parsed_items)} items")
        
        result['num_items'] = len(parsed_items)
        
        # 4. Match to USDA
        if verbose:
            print("[4/5] Matching items to USDA database...")
        
        items_with_matches = []
        for item in parsed_items:
            matches = match_item_to_usda(item['item_name_cleaned'], top_k=1)
            if matches:
                best_match = matches[0]
                items_with_matches.append({
                    'original_name': item['item_name'],
                    'cleaned_name': item['item_name_cleaned'],
                    'fdc_id': int(best_match['fdc_id']),  # Convert numpy.int64 to int
                    'matched_food': str(best_match['description']),
                    'confidence': round(float(best_match['confidence']), 3),
                    'quantity': float(item['quantity']),
                    'unit': str(item['unit']),
                    'price': float(item['total_price']) if item['total_price'] else None
                })
        
        # 5. Calculate macros
        if verbose:
            print("[5/5] Calculating macronutrients...")
        
        items_with_macros = []
        total_macros = {
            'calories': 0,
            'protein_g': 0,
            'carbs_g': 0,
            'fat_g': 0,
            'fiber_g': 0
        }
        
        for item in items_with_matches:
            macros = calculate_item_macros(
                item['fdc_id'],
                item['quantity'],
                item['unit']
            )
            
            if macros:
                item.update({
                    'calories': round(float(macros['calories']), 1),
                    'protein_g': round(float(macros['protein_g']), 1),
                    'carbs_g': round(float(macros['carbs_g']), 1),
                    'fat_g': round(float(macros['fat_g']), 1),
                    'fiber_g': round(float(macros['fiber_g']), 1)
                })
                
                for key in total_macros:
                    total_macros[key] += macros[key]
                
                items_with_macros.append(item)
        
        # Round totals and convert to native Python floats
        for key in total_macros:
            total_macros[key] = round(float(total_macros[key]), 1)
        
        if verbose:
            print(f"\n   Total: {total_macros['calories']:.0f} kcal")
        
        # Package results
        result['success'] = True
        result['items_with_macros'] = items_with_macros
        result['macros_summary'] = total_macros
        result['swap_suggestions'] = []  # Simplified for Flask app
        result['num_swaps'] = 0
        result['processing_time'] = time.time() - start_time
        
        if verbose:
            print(f"\n[OK] Analysis complete in {result['processing_time']:.2f}s")
            print('=' * 70)
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        if verbose:
            print(f"\n[ERROR] {e}")
    
    return result

# ============================================================================
# INITIALIZATION COMPLETE
# ============================================================================

print("\n[4/4] Pipeline ready!")
print("=" * 70)
print("[OK] All components loaded successfully")
print(f"[OK] USDA database: {len(usda_foods):,} foods")
print(f"[OK] Embedding model: {embedding_model.get_sentence_embedding_dimension()}-dim")
print("=" * 70)
print("\nTo analyze a receipt, call: analyze_receipt('path/to/image.jpg')\n")
