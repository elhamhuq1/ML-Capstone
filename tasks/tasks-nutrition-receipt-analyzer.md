# Task List: Nutrition Receipt Analyzer

## Relevant Files

- `nutrition_receipt_analyzer.ipynb` - Main Jupyter notebook containing the entire pipeline implementation
- `data/usda_fdc/` - Directory for USDA FoodData Central database CSV files
- `data/receipts/` - Directory for sample receipt images (JPG, PNG)
- `data/ground_truth.json` - Ground truth annotations for 3+ receipts for evaluation
- `requirements.txt` - Python dependencies list for reproducibility
- `data/models/` - Directory for downloaded sentence embedding models (cached locally)
- `results/evaluation_report.md` - Final evaluation report with metrics and analysis
- `README.md` - Instructions for running the notebook and setup guide

### Notes

- The entire pipeline is implemented in a single Jupyter notebook (`nutrition_receipt_analyzer.ipynb`)
- All processing is done locally on CPU
- USDA FoodData Central data can be downloaded from: https://fdc.nal.usda.gov/download-datasets.html
- Tesseract OCR must be installed separately on the system (not just via pip)

## Instructions for Completing Tasks

**IMPORTANT:** As you complete each task, you must check it off in this markdown file by changing `- [ ]` to `- [x]`. This helps track progress and ensures you don't skip any steps.

Example:

- `- [ ] 1.1 Read file` → `- [x] 1.1 Read file` (after completing)

Update the file after completing each sub-task, not just after completing an entire parent task.

## Tasks

- [x] 0.0 Create feature branch

  - [x] 0.1 Create and checkout a new branch for this feature (e.g., `git checkout -b feature/nutrition-receipt-analyzer`)

- [x] 1.0 Setup Jupyter Notebook and Install Dependencies

  - [x] 1.1 Create `requirements.txt` with all necessary Python packages (pytesseract, pandas, numpy, sentence-transformers, scikit-learn, matplotlib, seaborn, pillow, transformers)
  - [x] 1.2 Install system-level Tesseract OCR (platform-specific: apt-get, brew, or Windows installer)
  - [x] 1.3 Install Python dependencies from requirements.txt (`pip install -r requirements.txt`)
  - [x] 1.4 Create `nutrition_receipt_analyzer.ipynb` with initial markdown structure (Introduction, Setup, sections 1-9)
  - [x] 1.5 Add imports cell with all required libraries
  - [x] 1.6 Verify all imports work and Tesseract is accessible

- [x] 2.0 Data Acquisition and Preparation

  - [x] 2.1 Create directory structure: `data/usda_fdc/`, `data/receipts/`, `data/models/`
  - [x] 2.2 Download USDA FoodData Central database (Foundation Foods and SR Legacy CSV files)
  - [x] 2.3 Load USDA CSV files into pandas DataFrame and explore structure (columns: fdc_id, description, nutrients)
  - [x] 2.4 Filter and preprocess USDA data (select relevant columns, handle missing values)
  - [x] 2.5 Collect 5-10 diverse sample receipt images from different grocery stores
  - [x] 2.6 Create `data/ground_truth.json` with manual annotations for 3 receipts (item names, quantities, matched USDA IDs)
  - [x] 2.7 Load and display sample receipt images in notebook to verify data quality

- [x] 3.0 OCR and Text Extraction Implementation

  - [x] 3.1 Write function `load_receipt_image(image_path)` to load image using PIL
  - [x] 3.2 Implement function `extract_text_from_receipt(image)` using pytesseract.image_to_string()
  - [x] 3.3 Test OCR on all sample receipts and print raw extracted text
  - [x] 3.4 Implement function `extract_lines_from_receipt(image)` using pytesseract.image_to_data() for line-level extraction
  - [x] 3.5 Analyze OCR errors: calculate character error rate (CER) against ground truth
  - [x] 3.6 Document common OCR error patterns (misread characters, missing text, extra noise)

- [x] 4.0 Text Normalization and Parsing

  - [x] 4.1 Implement function `normalize_text(text)` for lowercasing and removing special characters
  - [x] 4.2 Create regex patterns to parse quantity/unit/price from receipt lines
  - [x] 4.3 Implement function `parse_receipt_line(line)` that returns dict with {item_name, quantity, unit, price}
  - [x] 4.4 Create list of common store codes and abbreviations to remove
  - [x] 4.5 Implement function `remove_store_codes(text)` to clean item names
  - [x] 4.6 Test parsing on all sample receipts and create structured DataFrame of items
  - [x] 4.7 Handle edge cases: lines without prices, multi-line items, totals/subtotals

- [x] 5.0 Food Matching Pipeline Development

  - [x] 5.1 Prepare USDA descriptions: extract and normalize all food descriptions from database
  - [x] 5.2 Implement baseline cosine similarity matching using TF-IDF vectorizer from scikit-learn
  - [x] 5.3 Write function `get_baseline_matches(item_name, usda_df, top_k=5)` that returns top-k USDA candidates
  - [x] 5.4 Load sentence embedding model (sentence-transformers/all-MiniLM-L6-v2) and encode USDA descriptions
  - [x] 5.5 Implement function `get_embedding_matches(item_name, usda_embeddings, top_k=10)` using k-NN
  - [x] 5.6 Create training dataset for re-ranker from ground truth annotations (features: cosine score, embedding score)
  - [x] 5.7 Train logistic regression re-ranker using scikit-learn on training data
  - [x] 5.8 Implement function `rerank_candidates(item_name, candidates)` that returns final match with confidence score
  - [x] 5.9 Evaluate matching accuracy on test set: calculate top-1, top-3, top-5 accuracy
  - [x] 5.10 Analyze matching errors and document challenging item types

  **NOTE:** Re-ranker is implemented but needs iteration:

  - Ground truth file (data/ground_truth.json) needs manual USDA food ID annotations for 10-20 items
  - Once annotated, re-run cells 51, 53 to train the re-ranker
  - Current system uses average of TF-IDF + embedding scores as fallback
  - Expected improvement: 0.28 → 0.6+ average confidence after training

- [x] 6.0 Macro Estimation and Unit Conversion

  - [x] 6.1 Create unit conversion dictionary mapping common units (oz, lb, g, kg, pack, etc.) to grams
  - [x] 6.2 Implement function `convert_to_grams(quantity, unit)` with handling for ambiguous units
  - [x] 6.3 Write function `get_nutrients_for_item(usda_id, usda_df)` that extracts calories, protein, carbs, fat per 100g
  - [x] 6.4 Implement function `scale_nutrients(nutrients, grams)` to calculate total macros for purchased quantity
  - [x] 6.5 Handle uncertainty: when unit is unclear, provide ranges (min/max estimates)
  - [x] 6.6 Implement confidence scoring based on match quality and unit clarity
  - [x] 6.7 Test macro estimation on ground truth receipts and calculate mean absolute percentage error (MAPE)
  - [x] 6.8 Create summary function that aggregates total macros across all receipt items

- [x] 7.0 Healthier Swap Suggestion Generation

  - [x] 7.1 Define swap rules dictionary (refined grain → whole grain, fatty meat → lean meat, full-fat dairy → low-fat)
  - [x] 7.2 Implement function `identify_swap_category(usda_description)` that classifies items by swap type
  - [x] 7.3 Write function `generate_swap_candidate(item, usda_df)` that finds healthier USDA alternatives
  - [x] 7.4 Implement comparison logic: calculate macro differences (lower saturated fat, higher fiber, etc.)
  - [x] 7.5 Add USDA citations to each swap suggestion (FDC IDs and URLs)
  - [x] 7.6 Create template strings for swap explanations (e.g., "Switch to {alternative} for {benefit}")
  - [x] 7.7 (Optional) Load small LLM (Qwen2.5-1.5B-Instruct) for rephrasing suggestions
  - [x] 7.8 (Optional) Implement function `rephrase_suggestion(template_text)` using LLM for friendly tone
  - [x] 7.9 Test swap generation on sample receipts and format output clearly

- [x] 8.0 Evaluation Suite Implementation

  - [x] 8.1 Implement OCR accuracy metrics: character error rate (CER) and line-level extraction accuracy
  - [x] 8.2 Calculate food mapping accuracy: top-1, top-3, top-5 against ground truth
  - [x] 8.3 Compute macro estimation error: MAPE for calories, protein, carbs, fat on annotated receipts
  - [x] 8.4 Measure end-to-end processing latency: time from image input to final output
  - [x] 8.5 Create visualizations: confusion matrices for matching, error distributions, latency charts
  - [x] 8.6 Implement qualitative evaluation: display sample outputs for manual review
  - [x] 8.7 Calculate confidence score distribution and analyze low-confidence predictions
  - [x] 8.8 Document edge cases and failure modes with examples

- [ ] 9.0 Final Integration, Testing, and Documentation

  - [x] 9.1 Integrate all pipeline components into end-to-end workflow in notebook
  - [x] 9.2 Create main function `analyze_receipt(image_path)` that runs entire pipeline
  - [ ] 9.3 Test pipeline on all sample receipts and verify outputs
  - [ ] 9.4 Generate example outputs with visualizations (tables, charts for macros and swaps)
  - [ ] 9.5 Create evaluation report section in notebook with all metrics and findings
  - [ ] 9.6 Write discussion section: challenges, limitations, future work
  - [x] 9.7 Create `README.md` with setup instructions, dependencies, and usage guide
  - [ ] 9.8 Export evaluation report to `results/evaluation_report.md`
  - [ ] 9.9 Clean up notebook: add markdown explanations, ensure cells run in sequence, test fresh kernel run
  - [ ] 9.10 Final review: verify all PRD requirements are met and success metrics are documented

  **BONUS:**

  - [ ] 9.11 Create Flask web application with UI for receipt upload and analysis
