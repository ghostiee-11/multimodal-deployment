import os
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import traceback # For more detailed error logging
from flask_cors import CORS # Import CORS

# --- Custom Module Imports ---
# (These should be the same as your previous correct server.py)

# Iteration 3 (main_assistant_w_new_one.py)
try:
    from main_assistant_w_new_one import (
        initial_setup as assistant_initial_setup_v3,
        get_assistant_response as get_assistant_response_v3,
        MAX_CONVERSATION_HISTORY_TURNS as MAX_CONVERSATION_HISTORY_TURNS_V3
    )
    print("Server: Successfully imported V3 assistant modules from main_assistant_w_new_one.py.")
except ImportError as e:
    print(f"Server WARNING: Could not import V3 assistant from main_assistant_w_new_one.py: {e}. V3 will not be available.")
    def assistant_initial_setup_v3(): print("V3 setup skipped due to import error."); return False
    def get_assistant_response_v3(query, history): return "Iteration V3 is currently unavailable due to an import error."
    MAX_CONVERSATION_HISTORY_TURNS_V3 = 3

# Iteration 2 (main_assistant_w_new.py)
try:
    from main_assistant_w_new import (
        initial_setup_v2,
        get_assistant_response as get_assistant_response_v2,
        MAX_CONVERSATION_HISTORY_TURNS_V2
    )
    print("Server: Successfully imported V2 assistant modules from main_assistant_w_new.py.")
except ImportError as e:
    print(f"Server WARNING: Could not import V2 assistant from main_assistant_w_new.py: {e}. V2 will not be available.")
    def initial_setup_v2(): print("V2 setup skipped due to import error."); return False
    def get_assistant_response_v2(query, history): return "Iteration V2 is currently unavailable due to an import error."
    MAX_CONVERSATION_HISTORY_TURNS_V2 = 3

# Iteration 1 (main_assistant_w.py)
try:
    from main_assistant_w import (
        initial_setup_v1,
        get_assistant_response as get_assistant_response_v1,
        MAX_CONVERSATION_HISTORY_TURNS_V1
    )
    print("Server: Successfully imported V1 assistant modules from main_assistant_w.py.")
except ImportError as e:
    print(f"Server WARNING: Could not import V1 assistant from main_assistant_w.py: {e}. V1 will not be available.")
    def initial_setup_v1(): print("V1 setup skipped due to import error."); return False
    def get_assistant_response_v1(query, history): return "Iteration V1 is currently unavailable due to an import error."
    MAX_CONVERSATION_HISTORY_TURNS_V1 = 3


app = Flask(__name__)
CORS(app) # Enable CORS for all routes on the app. Good for development.

# --- Global State ---
MAX_CONVERSATION_HISTORY_STORE_LIMIT = 7
conversation_history_store = []

iteration_ready_flags = {
    "v1": False,
    "v2": False,
    "v3": False,
}

# --- Initialization ---
print("Server: Starting application setup...")
if callable(assistant_initial_setup_v3):
    try:
        if assistant_initial_setup_v3():
            iteration_ready_flags["v3"] = True
            print("Server: V3 assistant resources initialized successfully.")
        else:
            print("Server: V3 assistant resource initialization failed or was skipped by its setup function.")
    except Exception as e:
        print(f"Server ERROR initializing V3 (main_assistant_w_new_one.py): {e}")
        traceback.print_exc()
else:
    print("Server: V3 initial_setup function not found or not callable.")


if callable(initial_setup_v2):
    try:
        if initial_setup_v2():
            iteration_ready_flags["v2"] = True
            print("Server: V2 assistant resources initialized successfully.")
        else:
            print("Server: V2 assistant resource initialization failed or was skipped by its setup function.")
    except Exception as e:
        print(f"Server ERROR initializing V2 (main_assistant_w_new.py): {e}")
        traceback.print_exc()
else:
    print("Server: V2 initial_setup function not found or not callable.")


if callable(initial_setup_v1):
    try:
        if initial_setup_v1():
            iteration_ready_flags["v1"] = True
            print("Server: V1 assistant resources initialized successfully.")
        else:
            print("Server: V1 assistant resource initialization failed or was skipped by its setup function.")
    except Exception as e:
        print(f"Server ERROR initializing V1 (main_assistant_w.py): {e}")
        traceback.print_exc()
else:
    print("Server: V1 initial_setup function not found or not callable.")

print("Server: Application setup phase complete.")
print(f"Server: Iteration readiness status: {iteration_ready_flags}")


# --- API Endpoints ---
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/style.css')
def style():
    return send_from_directory('.', 'style.css')

@app.route('/script.js')
def script():
    return send_from_directory('.', 'script.js')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    global conversation_history_store
    try:
        data = request.get_json()
        user_query = data.get('query')
        client_history = data.get('history', [])
        iteration_version = data.get('iteration', 'v3')

        if not user_query:
            return jsonify({"error": "Query cannot be empty"}), 400

        current_turn_history_tuples = client_history

        print(f"Server: Received query for iteration '{iteration_version}': '{user_query}'")

        assistant_answer = "Sorry, the selected iteration is not currently available or an error occurred during processing."

        if iteration_version == 'v3':
            if not iteration_ready_flags["v3"]:
                return jsonify({"error": "Iteration V3 backend is not ready or failed to initialize properly."}), 503
            assistant_answer = get_assistant_response_v3(user_query, current_turn_history_tuples)
        elif iteration_version == 'v2':
            if not iteration_ready_flags["v2"]:
                return jsonify({"error": "Iteration V2 backend is not ready or failed to initialize properly."}), 503
            assistant_answer = get_assistant_response_v2(user_query, current_turn_history_tuples)
        elif iteration_version == 'v1':
            if not iteration_ready_flags["v1"]:
                return jsonify({"error": "Iteration V1 backend is not ready or failed to initialize properly."}), 503
            assistant_answer = get_assistant_response_v1(user_query, current_turn_history_tuples)
        else:
            return jsonify({"error": f"Unknown iteration version '{iteration_version}'."}), 400

        updated_history_for_client = current_turn_history_tuples + [[user_query, assistant_answer]]
        if len(updated_history_for_client) > MAX_CONVERSATION_HISTORY_STORE_LIMIT:
            updated_history_for_client = updated_history_for_client[-MAX_CONVERSATION_HISTORY_STORE_LIMIT:]
        
        conversation_history_store = updated_history_for_client

        return jsonify({"answer": assistant_answer, "updated_history": updated_history_for_client})

    except Exception as e:
        print(f"Server Error in /api/chat for query '{user_query if 'user_query' in locals() else 'unknown'}': {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred processing your request.", "details": str(e)}), 500

if __name__ == '__main__':
    load_dotenv()
    # CHANGED PORT TO 5016
    port = int(os.environ.get("PORT", 5016))
    print(f"Flask server attempting to run on http://0.0.0.0:{port}")
    print("Ensure you are accessing the website via this address and port after starting this server.")
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)