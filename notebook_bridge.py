"""
Bridge between Flask app and Jupyter notebook analysis pipeline.

This file should be run AFTER the notebook has been executed
to import all the necessary functions and data.
"""

# Instructions for setup:
# 1. Run all cells in nutrition_receipt_analyzer.ipynb
# 2. Run: jupyter nbconvert --to script nutrition_receipt_analyzer.ipynb
# 3. This will create nutrition_receipt_analyzer.py
# 4. Import functions from that file

def analyze_receipt_api(image_path):
    """
    Wrapper for the analyze_receipt function from the notebook.
    Returns results formatted for the API.
    """
    try:
        # Import from the converted notebook
        from nutrition_receipt_analyzer import analyze_receipt
        
        # Run analysis
        result = analyze_receipt(image_path, verbose=False)
        
        return result
        
    except ImportError as e:
        print("Error: Could not import from notebook.")
        print("Please run: jupyter nbconvert --to script nutrition_receipt_analyzer.ipynb")
        raise e
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise e

# For development/testing
if __name__ == "__main__":
    print("Testing notebook bridge...")
    print("This requires the notebook to be converted first.")
    print("Run: jupyter nbconvert --to script nutrition_receipt_analyzer.ipynb")

