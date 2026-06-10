import yaml
import time
from litellm import Router
import litellm
import os
from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLMRouter
# Monkeypatch tiktoken to prevent LiteLLM from crashing on empty/None responses from rate-limited Cloudflare endpoints
import tiktoken.core
_original_encode = tiktoken.core.Encoding.encode
def safe_encode(self, text, *args, **kwargs):
    if text is None:
        text = ""
    return _original_encode(self, str(text), *args, **kwargs)
tiktoken.core.Encoding.encode = safe_encode

litellm.suppress_debug_info = True
litellm.drop_params = True

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
CLOUDFLARE_API_KEY = os.getenv("CLOUDFLARE_API_KEY")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")

# Set them in os.environ so litellm picks them up natively if needed
if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
if HUGGINGFACE_API_KEY:
    os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY
    os.environ["HF_TOKEN"] = HUGGINGFACE_API_KEY
if CLOUDFLARE_API_KEY:
    os.environ["CLOUDFLARE_API_KEY"] = CLOUDFLARE_API_KEY
if CLOUDFLARE_ACCOUNT_ID:
    os.environ["CLOUDFLARE_ACCOUNT_ID"] = CLOUDFLARE_ACCOUNT_ID

config_path = os.path.join(os.path.dirname(__file__), "litellm_config.yaml")
with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
        
model_list = config_data.get("model_list", [])
router_settings = config_data.get("router_settings", {})

print(f"Initializing LiteLLM Router with {len(model_list)} models...")
# Initialize the Router with our models and settings
router = Router(model_list=model_list, **router_settings)

def model_predict(model_config_type,model_messages,max_tokens=1000,temp=0.2):
    try:
        response_predict = router.completion(
            model=model_config_type,
            messages=model_messages,
            temperature=temp,
            max_tokens=max_tokens
        )
        print(f"Model used : {response_predict.model}")
    except Exception as e:
        print(f"Error testing  {type(e).__name__} - {str(e)}")
        raise
    return response_predict

def get_litellm(model_config_type,max_tokens=1000,temp=0.2):
    try:
        llm = ChatLiteLLMRouter(router = router,model=model_config_type,temperature=temp,
            max_tokens=max_tokens)
        print(f"model used is {llm.model}")
    except Exception as e:
        print(f"Error testing  {type(e).__name__} - {str(e)}")
        raise
    return llm

def get_hf_embedding(model_name="BAAI/bge-small-en-v1.5"):
    try:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError("HuggingFace API key not found. Please set HUGGINGFACE_API_KEY.")
            
        embeddings = HuggingFaceEndpointEmbeddings(
            model=model_name,
            huggingfacehub_api_token=api_key
        )
        print(f"Loaded HuggingFace Endpoint Embeddings: {model_name}")
        return embeddings
    except Exception as e:
        print(f"Error loading HuggingFace Embeddings: {type(e).__name__} - {str(e)}")
        raise

def get_image_description_hf(image_path: str, prompt: str = "Describe this image in detail, including any text, tables, or charts.") -> str:
    import os
    import base64
    import requests
    
    api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
    if not api_key:
        raise ValueError("HuggingFace API key not found. Please set HUGGINGFACE_API_KEY or HF_TOKEN.")
        
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return f"[Error loading image: {e}]"
        
    image_url = f"data:image/jpeg;base64,{img_b64}"
    
def get_image_description(image_path: str, prompt: str = "Describe this image in detail, including any text, tables, or charts.", model="gemini/gemini-2.5-flash") -> str:
    """
    Uses LiteLLM to generate a text description of an image.
    Defaults to Gemini 1.5 Flash (free and highly capable VLM).
    (Hugging Face Serverless API currently blocks image models on their router endpoint, 
    and their legacy endpoint is blocked by your ISP's DNS).
    """
    import base64
    from litellm import completion
    import os
    
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return f"[Error loading image: {e}]"
        
    image_url = f"data:image/jpeg;base64,{img_b64}"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ]
    
    try:
        # We can use the globally configured litellm router, or direct completion
        response = completion(
            model=model,
            messages=messages,
            max_tokens=10000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error calling VLM via LiteLLM: {repr(e)}]"