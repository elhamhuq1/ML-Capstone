# Product Requirements Document: Nutrition Receipt Analyzer

## 1. Introduction/Overview

The Nutrition Receipt Analyzer is a research tool that extracts actionable nutrition insights from grocery receipt photos without requiring manual food logging. The system reads receipt images, identifies purchased items, maps them to USDA nutritional data, estimates macronutrients (calories, protein, carbs, fat), and suggests healthier alternatives with clear citations.

**Problem Statement:** Current nutrition tracking tools require tedious manual logging, leading to underreporting and user fatigue. Grocery receipts already contain a complete list of purchased items but are underutilized for nutrition analysis.

**Solution:** A fully local, CPU-friendly pipeline implemented as a single Jupyter notebook that processes receipt photos end-to-end, providing per-item macro estimates and evidence-based swap suggestions.

**Target Users:** 
- Health-conscious individuals tracking their own nutrition
- General consumers curious about the nutritional content of their grocery purchases

## 2. Goals

1. **Accurate Text Extraction:** Achieve >90% accuracy in extracting item names and quantities from receipt photos
2. **Reliable Food Mapping:** Achieve >80% accuracy in mapping noisy receipt text to correct USDA FoodData Central entries
3. **Useful Swap Suggestions:** Generate clear, evidence-based healthier alternatives that users find helpful and easy to understand
4. **Local Processing:** Ensure the entire pipeline runs locally on CPU without requiring cloud services or GPU
5. **Transparency:** Provide clear citations to USDA entries and confidence scores for all estimates
6. **Research Contribution:** Demonstrate a zero-cost pipeline with comprehensive evaluation covering extraction, mapping, macro-accuracy, and suggestion usefulness

## 3. User Stories

1. **As a health-conscious shopper**, I want to photograph my grocery receipt after shopping so that I can quickly understand the nutritional content of my purchases without manually logging each item.

2. **As a general consumer**, I want to see which items in my cart are less healthy so that I can make better choices next time without feeling overwhelmed.

3. **As a nutrition-curious user**, I want to see healthier alternatives for items I bought so that I can learn about better options with clear explanations and evidence.

4. **As a skeptical user**, I want to see confidence scores and data sources for nutritional estimates so that I can trust the tool's accuracy.

5. **As a researcher**, I want to evaluate the pipeline's performance across multiple dimensions (OCR accuracy, mapping accuracy, macro estimation error, latency) so that I can validate the methodology.

## 4. Functional Requirements

### 4.1 Receipt Image Processing
1. The system must accept receipt photos taken on phone or laptop cameras in common formats (JPG, PNG)
2. The system must use Tesseract OCR (via pytesseract) to extract text from receipt images
3. The system must handle messy, real-world receipts with varying quality, lighting, and formatting
4. The system must parse extracted text to identify item names, quantities, units, and prices

### 4.2 Text Normalization
5. The system must normalize receipt text by lowercasing, removing store codes, and cleaning special characters
6. The system must parse quantity/unit/price information from receipt lines
7. The system must handle common receipt abbreviations and store-specific formatting

### 4.3 Food Matching
8. The system must load and query the USDA FoodData Central database locally
9. The system must implement baseline cosine similarity matching between receipt items and USDA descriptions
10. The system must use sentence embeddings (e.g., sentence-transformers/all-MiniLM-L6-v2 or intfloat/e5-small-v2) for robust text similarity
11. The system must retrieve top-k candidate USDA matches using k-NN
12. The system must implement a logistic regression re-ranker with features (cosine score, embedding score) to select the final USDA match
13. The system must provide confidence scores for matches based on similarity metrics

### 4.4 Macro Estimation
14. The system must convert receipt quantities (lbs, oz, packs, etc.) to grams
15. The system must scale USDA nutrient values to match the quantity purchased
16. The system must calculate per-item estimates for calories, protein, carbohydrates, and fat
17. The system must display ranges or confidence intervals when match quality is uncertain
18. The system must handle missing or incomplete unit information gracefully

### 4.5 Healthier Swap Suggestions
19. The system must generate rule-based healthier alternatives (e.g., whole-grain for refined grains, lean protein for fatty cuts)
20. The system must provide USDA citations for all swap suggestions
21. The system must optionally rephrase suggestions using a small local LLM (e.g., Qwen2.5-1.5B-Instruct) for clarity and friendliness without changing facts
22. The system must explain why each swap is healthier (e.g., "lower saturated fat," "higher fiber")

### 4.6 Output and Reporting
23. The system must display results in a clear, readable format within the Jupyter notebook
24. The system must show per-item breakdowns with: original item, matched USDA entry, macros, confidence score
25. The system must summarize total macros across all items
26. The system must list suggested swaps with comparisons (original vs. alternative macros)

### 4.7 Evaluation Suite
27. The system must include evaluation metrics for OCR accuracy (character error rate, item extraction accuracy)
28. The system must measure food mapping accuracy (top-1, top-3, top-5 accuracy)
29. The system must evaluate macro estimation error against ground truth where available
30. The system must measure end-to-end processing latency
31. The system must include sample receipts with ground truth annotations for testing

### 4.8 Technical Implementation
32. The system must be implemented as a single end-to-end Jupyter notebook with all code in sequence
33. The system must run entirely on CPU without GPU requirements
34. The system must use only open-source, freely available libraries and datasets
35. The system must process a typical receipt (10-20 items) in under 60 seconds on standard laptop hardware

## 5. Non-Goals (Out of Scope)

The following features are explicitly **not** included in this project:

1. **Multi-user support or user accounts:** No database of users, login systems, or user profiles
2. **Mobile app development:** The tool is implemented as a Jupyter notebook only, not as a mobile or web application
3. **Real-time nutritional tracking over time:** No longitudinal tracking, dashboards, or progress monitoring
4. **Dietary restriction personalization:** No customization for allergies, religious restrictions, dietary preferences (vegan, keto, etc.)
5. **Recipe generation:** No full recipe creation, only references to RecipeNLG for phrasing examples
6. **Barcode scanning:** Only receipt text is processed, not product barcodes
7. **Price optimization or budgeting:** No financial analysis beyond parsing receipt prices
8. **Restaurant receipts:** Only grocery store receipts are supported
9. **Integration with fitness trackers or health apps:** No API integrations or data exports
10. **Cloud deployment:** The system remains local and is not deployed as a web service

## 6. Design Considerations

### 6.1 Jupyter Notebook Structure
- The notebook should be organized into clear sections with markdown headers:
  1. Introduction and Setup
  2. Data Loading (USDA database, sample receipts)
  3. OCR and Text Extraction
  4. Text Normalization and Parsing
  5. Food Matching (Baseline + Embeddings + Re-ranker)
  6. Macro Estimation
  7. Swap Suggestion Generation
  8. Evaluation Suite
  9. Results and Discussion

### 6.2 User Experience
- All outputs should be clearly formatted with tables, visualizations, and readable text
- Confidence scores should be presented with visual indicators (e.g., color coding, percentage bars)
- Citations should be inline and clickable (USDA FDC IDs with URLs)
- Error messages should be informative and guide users on how to improve results

### 6.3 Sample Data
- Include 5-10 diverse sample receipt images from different stores
- Provide ground truth annotations for at least 3 receipts for validation
- Include edge cases: poor lighting, crumpled receipts, unusual formats

## 7. Technical Considerations

### 7.1 Libraries and Dependencies
- **OCR:** pytesseract (Python wrapper for Tesseract)
- **Text Processing:** pandas, numpy, re (regex)
- **Embeddings:** sentence-transformers (Hugging Face)
- **Machine Learning:** scikit-learn (logistic regression, k-NN)
- **Optional LLM:** transformers (Hugging Face) for Qwen2.5-1.5B-Instruct
- **Data:** USDA FoodData Central database (CSV download)
- **Visualization:** matplotlib, seaborn (for evaluation plots)

### 7.2 Data Sources
- **USDA FoodData Central:** Public-domain nutritional database with comprehensive macro and micronutrient data
- **RecipeNLG:** Optional corpus for phrasing examples for friendly suggestions

### 7.3 Performance Constraints
- CPU-only processing (no CUDA required)
- Target latency: <60 seconds for end-to-end processing of a typical receipt
- Memory footprint: Should run on laptops with 8GB RAM

### 7.4 Fallback Strategies
- If OCR noise is too high, document the option to use Document AI APIs as an alternative
- If cosine similarity is insufficient, prioritize embedding-based matching
- If re-ranker doesn't improve results, document and fall back to embedding scores alone

### 7.5 Privacy and Data Handling
- All processing is local; no data is sent to external servers
- Receipt images are not stored permanently unless explicitly saved by the user
- No personally identifiable information (PII) is extracted or logged

## 8. Success Metrics

### 8.1 Quantitative Metrics
1. **OCR Accuracy:** >90% correct extraction of item names and quantities (measured by character error rate and line-level accuracy against ground truth)
2. **Food Mapping Accuracy:** >80% top-1 accuracy in matching receipt items to correct USDA entries (measured on annotated test set)
3. **Top-3 Mapping Accuracy:** >90% of correct USDA matches appear in top-3 candidates
4. **Macro Estimation Error:** Mean absolute percentage error (MAPE) <20% for calories, protein, carbs, fat (where ground truth is available)
5. **Processing Latency:** <60 seconds per receipt on standard laptop CPU

### 8.2 Qualitative Metrics
6. **Swap Usefulness:** Collect feedback from 5-10 test users on whether swap suggestions are:
   - Easy to understand
   - Actionable
   - Evidence-based and trustworthy
   - Friendly in tone (if LLM rephrasing is used)
7. **Clarity of Output:** Users can easily identify which USDA entry was matched to each receipt item and understand confidence scores

### 8.3 Research Contributions
8. Demonstrate a working zero-cost pipeline from receipt photo to actionable nutrition insights
9. Provide a comprehensive evaluation suite that can be reproduced by others
10. Document challenges and limitations clearly for future research

## 9. Open Questions

1. **OCR Fallback:** If Tesseract accuracy is below 90%, should Document AI be integrated as a premium option, or should preprocessing techniques be prioritized?

2. **Unit Conversion Ambiguity:** How should the system handle receipt items with unclear units (e.g., "2 packs of chicken" without weight)? Should default assumptions be made, or should these be flagged as low-confidence?

3. **Swap Suggestion Threshold:** What confidence score threshold should trigger a swap suggestion? Should low-confidence items be excluded from suggestions?

4. **LLM Rephrasing Trade-off:** Does the LLM rephrasing step add meaningful value in terms of user satisfaction, or does it add unnecessary latency? This should be evaluated with A/B testing.

5. **Ground Truth Acquisition:** How will ground truth annotations be created for evaluation? Will this involve manual labeling by domain experts, or will a smaller validation set be used?

6. **Store-Specific Formatting:** Should the system be trained/tuned for specific grocery store receipt formats, or should it remain store-agnostic?

7. **Error Handling:** How should the system respond to completely illegible receipts or non-receipt images? Should there be a quality check step before processing?

8. **Batch Processing:** Should the notebook support processing multiple receipts in a batch, or is single-receipt processing sufficient for the research scope?

9. **Visualization of Results:** What visualizations (bar charts, pie charts, tables) are most useful for presenting macro breakdowns and comparisons?

10. **Extension to Document AI:** If the project timeline allows, should Document AI be integrated as an optional enhancement, or should this be left as future work?

---

## Deliverables

1. **Jupyter Notebook:** A single, well-documented notebook implementing the entire pipeline from receipt photo to macro estimates and swap suggestions
2. **Evaluation Report:** A separate document (or final section in the notebook) with:
   - Performance metrics (OCR accuracy, mapping accuracy, macro estimation error, latency)
   - Qualitative analysis of swap suggestions
   - Discussion of challenges, limitations, and future work
   - Visualizations of results
3. **Sample Data:** 5-10 receipt images with at least 3 annotated ground truth examples
4. **README or Documentation:** Instructions for running the notebook, installing dependencies, and downloading the USDA database

---

## Timeline Reference

This PRD supports the following project timeline:
- **Mid Sept:** Receipt collection and baseline OCR testing
- **Late Sept:** Preprocessing pipeline development
- **Early Oct:** USDA database setup and baseline matching
- **Early-Mid Oct:** Embedding-based matching and k-NN retrieval
- **Mid Oct:** Re-ranker implementation and evaluation
- **Late Oct:** Macro estimation and unit conversion
- **Early Nov:** Swap suggestion generation
- **Mid Nov:** Optional LLM rephrasing
- **Late Nov:** Jupyter notebook integration and stress testing
- **End Nov - Early Dec:** Comprehensive evaluation and reporting

---

**Document Version:** 1.0  
**Last Updated:** December 5, 2025  
**Status:** Final - Ready for Implementation

