# app/llm_service.py
import os
from dotenv import load_dotenv
from transformers import pipeline as hf_pipeline # Rename to avoid conflict
import google.generativeai as genai
from groq import Groq, RateLimitError
import time

# Load API keys from .env file
load_dotenv()

# --- Constants ---
DEFAULT_MODEL = "local_flan_t5_small"
MODEL_CONFIG = {
    "local_flan_t5_small": {"type": "local", "name": "google/flan-t5-small"},
    "gemini-1.5-flash": {"type": "google", "name": "gemini-1.5-flash"},
    "llama3-8b-8192": {"type": "groq", "name": "llama3-8b-8192"},
    # Add more models here if needed, e.g., from Groq or other APIs
    # "mixtral-8x7b-32768": {"type": "groq", "name": "mixtral-8x7b-32768"},
}

# --- API Client Configuration ---
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Google AI SDK Configured.")
    else:
        print("Warning: GOOGLE_API_KEY not found in .env file. Google models disabled.")
except Exception as e:
    print(f"Error configuring Google AI SDK: {e}")
    GOOGLE_API_KEY = None # Disable if config fails

try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("Groq Client Initialized.")
    else:
        print("Warning: GROQ_API_KEY not found in .env file. Groq models disabled.")
except Exception as e:
    print(f"Error initializing Groq Client: {e}")
    GROQ_API_KEY = None # Disable if config fails

# --- LLM Service Class ---
class LLMService:
    def __init__(self):
        self.local_generator = None
        self.google_model_client = None
        self.groq_client = None

        # Initialize the default local model
        self._initialize_local_model(MODEL_CONFIG[DEFAULT_MODEL]["name"])

        # Prepare API model clients if keys are available
        if GOOGLE_API_KEY:
            # Using a default model name, actual model used depends on the call
            self.google_model_client = genai.GenerativeModel('gemini-1.5-flash')
        if GROQ_API_KEY:
            self.groq_client = groq_client # Use the client initialized outside

    def _initialize_local_model(self, model_name):
        """Initializes the local Hugging Face pipeline."""
        try:
            print(f"Loading local model: {model_name}...")
            self.local_generator = hf_pipeline(
                "text2text-generation",
                model=model_name,
                # device_map="auto"
            )
            print("Local model loaded successfully!")
        except Exception as e:
            import traceback
            print(f"!!! Error initializing local model {model_name} !!!")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Args: {e.args}")
            traceback.print_exc() # Print the full traceback
            self.local_generator = None

    def create_prompt(self, functionality, query, context=None):
        """Creates a base prompt structure."""
        # This structure works reasonably well for Flan-T5.
        # For API models, we might adapt this slightly or use different structures later.
        if functionality == "concept_explanation":
            prompt_core = f"Explain the following concept clearly:\n\nCONCEPT: {query}\n\nProvide a simple explanation with examples."
        elif functionality == "practice_questions":
            prompt_core = f"Generate 3 practice questions about:\n\nTOPIC: {query}\n\nFor each question, also provide the answer."
        elif functionality == "homework_helper":
            prompt_core = f"Help solve this problem step by step:\n\nPROBLEM: {query}\n\nGive hints and guidance without giving the complete answer."
        else:
            prompt_core = query # Default to just the query if functionality unknown

        # Add context if provided
        if context:
            prompt_core += f"\n\nUse the following context:\n{context}"

        return prompt_core

    def _generate_local_response(self, prompt, max_length=512):
        """Generates response using the local Hugging Face model."""
        if not self.local_generator:
            return "Error: Local model not loaded."
        try:
            # Use do_sample=False for flan-t5-small to reduce nonsense, though quality is still low
            response = self.local_generator(
                prompt,
                max_length=max_length,
                do_sample=False
            )[0]['generated_text']
            return response
        except Exception as e:
            print(f"Error during local generation: {e}")
            return "Error generating response from local model."

    def _generate_google_response(self, model_name, prompt, max_length=512):
        """Generates response using the Google Gemini API."""
        if not self.google_model_client or model_name != self.google_model_client.model_name:
             # Re-initialize if model name changed or client not ready
             try:
                 self.google_model_client = genai.GenerativeModel(model_name)
             except Exception as e:
                 print(f"Error initializing Google model {model_name}: {e}")
                 return "Error: Could not initialize the selected Google model."

        print(f"Sending request to Google API ({model_name})...")
        try:
            # Gemini prefers direct instruction over preamble in the prompt itself
            # We will just send the core prompt content
            response = self.google_model_client.generate_content(
                prompt,
                # Adjust generation config if needed, e.g., max_output_tokens
                # generation_config=genai.types.GenerationConfig(max_output_tokens=max_length)
                )
            # Handle potential safety blocks
            if not response.candidates:
                 # Check if prompt was blocked
                 try:
                     prompt_feedback = response.prompt_feedback
                     if prompt_feedback.block_reason:
                         print(f"Google API Prompt Blocked: {prompt_feedback.block_reason}")
                         return f"Request blocked by Google API due to: {prompt_feedback.block_reason}. Please rephrase."
                 except (ValueError, AttributeError):
                      pass # No candidates and no prompt feedback info
                 return "Error: Google API returned no response candidates. The prompt might be inappropriate or the model failed."

            # Check for finish reason and safety ratings if needed
            # finish_reason = response.candidates[0].finish_reason
            # safety_ratings = response.candidates[0].safety_ratings

            return response.text
        except Exception as e:
            print(f"Error during Google API call: {e}")
            return "Error communicating with Google API."

    def _generate_groq_response(self, model_name, prompt, max_length=512):
        """Generates response using the Groq API."""
        if not self.groq_client:
            return "Error: Groq client not initialized (check API key)."

        print(f"Sending request to Groq API ({model_name})...")
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful educational assistant." # System message for context
                    },
                    {
                        "role": "user",
                        "content": prompt, # The user's request/prompt
                    }
                ],
                model=model_name,
                max_tokens=max_length,
                # temperature=0.7, # Adjust if needed
            )
            return chat_completion.choices[0].message.content
        except RateLimitError as e:
             print(f"Groq API Rate Limit Error: {e}")
             # Implement basic exponential backoff or wait
             wait_time = 5
             print(f"Rate limited. Waiting for {wait_time} seconds...")
             time.sleep(wait_time)
             # Consider retrying once, but for simplicity return error now
             return "Error: Groq API rate limit exceeded. Please try again shortly."
        except Exception as e:
            print(f"Error during Groq API call: {e}")
            return "Error communicating with Groq API."

    def generate_response(self, model_id, functionality, query, context=None, max_length=512):
        """Main function to generate response based on selected model."""
        if model_id not in MODEL_CONFIG:
            print(f"Warning: Invalid model_id '{model_id}'. Falling back to default.")
            model_id = DEFAULT_MODEL

        config = MODEL_CONFIG[model_id]
        model_type = config["type"]
        model_name = config["name"] # Specific name for the API or local path

        # Create the base prompt content
        # Note: API models often work better with direct instructions rather than long preambles.
        # We will pass the 'core' of the prompt to the API functions.
        core_prompt = self.create_prompt(functionality, query, context)

        print(f"\n--- Generating with Model ID: {model_id} ({model_type}) ---")
        print(f"Core Prompt Content:\n{core_prompt[:200]}...\n" + "-"*20) # Print start of prompt

        if model_type == "local":
            response = self._generate_local_response(core_prompt, max_length)
        elif model_type == "google":
            response = self._generate_google_response(model_name, core_prompt, max_length)
        elif model_type == "groq":
            response = self._generate_groq_response(model_name, core_prompt, max_length)
        else:
            print(f"Error: Unknown model type '{model_type}' for model_id '{model_id}'")
            response = "Error: Invalid model configuration."

        print(f"--- Raw Response ---\n{response[:200]}...\n" + "-"*20)
        return response