# app/app.py
from flask import Flask, request, jsonify, send_from_directory
import json
import os
import uuid
from datetime import datetime

# Use relative imports within the 'app' package
from .llm_service import LLMService, DEFAULT_MODEL # Import default model ID
from .text_processing import extract_text_from_file, clean_text

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DATA_FOLDER = os.path.join(BASE_DIR, 'data')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

INTERACTIONS_FILE = os.path.join(DATA_FOLDER, 'interactions.json')
MATERIALS_FILE = os.path.join(DATA_FOLDER, 'materials.json')

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='')

# --- Initialize LLM Service ---
# Service now handles multiple models internally
llm_service = LLMService()

# --- Helper functions (load_json, save_json - keep as before) ---
def load_json(file_path):
    # ... (keep existing implementation)
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content: # Handle empty file
                    return []
                return json.loads(content)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}. Returning empty list.")
            return []
        except Exception as e:
            print(f"Error loading JSON from {file_path}: {e}")
            return []
    return []

def save_json(file_path, data):
    # ... (keep existing implementation)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, default=str)
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")


# --- Routes ---

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '')
    functionality = data.get('functionality', 'concept_explanation')
    material_id = data.get('materialId')
    model_id = data.get('modelId', DEFAULT_MODEL) # Get selected model ID, fallback to default

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    context = None
    if material_id:
        materials = load_json(MATERIALS_FILE)
        found_material = next((m for m in materials if m.get('id') == material_id), None)
        if found_material:
            context = found_material.get('extracted_text', '')
            print(f"Using context from material: {found_material.get('filename')}")
        else:
            print(f"Warning: Material ID {material_id} not found.")

    # --- Generate Response using selected model via LLM Service ---
    try:
        # Pass model_id to the service
        response_text = llm_service.generate_response(
            model_id=model_id,
            functionality=functionality,
            query=query,
            context=context
            # max_length can be adjusted if needed, but service might handle internally for APIs
        )
    except Exception as e:
        print(f"Error during LLM generation call in app.py: {e}")
        # Check if llm_service itself failed during init maybe?
        if not llm_service:
             return jsonify({"error": "LLM Service failed to initialize."}), 503
        return jsonify({"error": f"Failed to generate response using model {model_id}."}), 500

    # --- Store Interaction ---
    interaction_id = str(uuid.uuid4())
    interaction = {
        "id": interaction_id,
        "timestamp": datetime.now().isoformat(),
        "model_used": model_id, # Log which model was used
        "functionality": functionality,
        "query": query,
        "context_material_id": material_id,
        "response": response_text,
        "rating": None
    }
    interactions = load_json(INTERACTIONS_FILE)
    interactions.append(interaction)
    save_json(INTERACTIONS_FILE, interactions)

    return jsonify({
        "response": response_text,
        "interactionId": interaction_id
    })

# --- /api/upload and /api/feedback routes (keep as before) ---
@app.route('/api/upload', methods=['POST'])
def upload_material():
    # ... (keep existing implementation)
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400

    if file:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        try:
            file.save(filepath)
            print(f"File saved to: {filepath}")
            extracted_text = extract_text_from_file(filepath)
            if "Unsupported file format" in extracted_text:
                 os.remove(filepath)
                 return jsonify({"success": False, "error": extracted_text}), 415

            cleaned_text = clean_text(extracted_text)
            material_id = str(uuid.uuid4())
            material_data = {
                "id": material_id, "filename": filename,
                "original_filepath": filepath,
                "upload_time": datetime.now().isoformat(),
                "extracted_text": cleaned_text
            }
            materials = load_json(MATERIALS_FILE)
            materials.append(material_data)
            save_json(MATERIALS_FILE, materials)
            return jsonify({"success": True, "message": f"File '{filename}' uploaded.", "materialId": material_id})
        except Exception as e:
            print(f"Error processing upload: {e}")
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError as ose:
                    print(f"Error removing file: {ose}")
            return jsonify({"success": False, "error": "Server error during file processing"}), 500
    return jsonify({"success": False, "error": "Unknown error during upload"}), 500


@app.route('/api/feedback', methods=['POST'])
def feedback():
    # ... (keep existing implementation)
    data = request.json
    interaction_id = data.get('interactionId')
    rating = data.get('rating')
    if not interaction_id or rating is None: return jsonify({"success": False, "error": "Missing interactionId or rating"}), 400
    try: rating = int(rating); assert rating in [1, 5]
    except (ValueError, AssertionError): return jsonify({"success": False, "error": "Invalid rating value"}), 400

    interactions = load_json(INTERACTIONS_FILE)
    interaction_updated = False
    for interaction in interactions:
        if interaction.get('id') == interaction_id:
            interaction['rating'] = rating; interaction['feedback_time'] = datetime.now().isoformat()
            interaction_updated = True; break
    if interaction_updated:
        save_json(INTERACTIONS_FILE, interactions)
        print(f"Feedback recorded for interaction {interaction_id}: Rating {rating}")
        return jsonify({"success": True, "message": "Feedback received"})
    else:
        print(f"Feedback received for unknown interaction ID: {interaction_id}")
        return jsonify({"success": False, "error": "Interaction ID not found"}), 404
