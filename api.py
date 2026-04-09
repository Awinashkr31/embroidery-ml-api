from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

# Load environment variables
load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
if not url or not key:
    raise ValueError("Missing Supabase credentials in .env")

supabase: Client = create_client(url, key)

app = FastAPI(title="Embroidery ML Recommendation API")

# Allow Vite frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to cache the TF-IDF matrix (Optimization)
trained_df = None
cosine_sim_matrix = None

def train_model():
    """Fetches all products and computes the cosine similarity matrix."""
    global trained_df, cosine_sim_matrix
    try:
        # Fetch only what's necessary (ignore variants, stock, etc. for ml)
        response = supabase.table("products").select("id, name, category, description").execute()
        data = response.data
        if not data:
            return False
        
        df = pd.DataFrame(data)
        
        # Combine text features
        df['description'] = df['description'].fillna('')
        df['combined_features'] = df['name'] + " " + df['category'] + " " + df['description']
        
        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['combined_features'])
        
        # Compute Cosine Similarity
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        trained_df = df
        cosine_sim_matrix = cosine_sim
        print(f"Model trained on {len(df)} products.")
        return True
    except Exception as e:
        print(f"Error training model: {e}")
        return False

@app.on_event("startup")
def startup_event():
    train_model()

@app.get("/api/recommendations/refresh")
def refresh_cache():
    """Endpoint to trigger a manual refresh of the recommendation model when new products are added."""
    success = train_model()
    if success:
        return {"message": "Model refreshed successfully"}
    raise HTTPException(status_code=500, detail="Failed to refresh model")

@app.get("/api/recommendations/{product_id}")
def get_recommendations(product_id: int, limit: int = 4):
    """Returns a list of recommended product IDs based on the provided product ID."""
    global trained_df, cosine_sim_matrix
    
    if trained_df is None or cosine_sim_matrix is None:
        if not train_model():
            raise HTTPException(status_code=500, detail="Recommendation model could not be initialized")
            
    try:
        if product_id not in trained_df['id'].values:
            raise HTTPException(status_code=404, detail="Product not found in catalog")
            
        idx = trained_df.index[trained_df['id'] == product_id].tolist()[0]
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:limit+1]
        
        product_indices = [i[0] for i in sim_scores]
        recommended_ids = trained_df.iloc[product_indices]['id'].tolist()
        return {"product_id": product_id, "recommendations": recommended_ids}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from sales_forecast import get_forecast_json

@app.get("/api/admin/forecast")
def get_sales_forecast_endpoint():
    """Generates a 30-day Prophet time-series sales forecast on demand."""
    try:
        data = get_forecast_json()
        return {"forecast": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========================================================
# 🤖 NLP SENTIMENT ANALYSIS API 
# ========================================================
from pydantic import BaseModel
from typing import List
import json

class ReviewPayload(BaseModel):
    id: str
    text: str

class ReviewsBatch(BaseModel):
    reviews: List[ReviewPayload]

@app.post("/api/admin/analyze_reviews")
def batch_analyze_reviews(payload: ReviewsBatch):
    """Batch analyzes multiple reviews for sentiment and abusive language using Groq."""
    global client
    if not client:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key and groq_api_key != "INSERT_YOUR_GROQ_KEY_HERE":
            from groq import Groq
            client = Groq(api_key=groq_api_key)
        else:
            raise HTTPException(status_code=500, detail="Groq API Key is missing.")
            
    if not payload.reviews:
        return {}

    prompt_lines = ["Analyze the following reviews. Return a JSON object mapping review IDs to their analysis. Output ONLY valid JSON."]
    prompt_lines.append("Format: { \"id1\": {\"sentiment\": \"POSITIVE\"|\"NEGATIVE\"|\"NEUTRAL\", \"is_abusive\": true|false}, ... }")
    prompt_lines.append("Reviews:")
    for r in payload.reviews:
        prompt_lines.append(f"ID {r.id}: {r.text}")

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a strict JSON-only API analyzing customer sentiment and extracting toxic/abusive content. You output ONLY valid JSON."},
                {"role": "user", "content": "\\n".join(prompt_lines)}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        reply = response.choices[0].message.content.strip()
        analysis = json.loads(reply)
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========================================================
# 🤖 CHATBOT IMPLEMENTATION WITH GROQ
# ========================================================
from pydantic import BaseModel

from datetime import datetime, timedelta

# Define Tools
def get_order_status(order_id: str) -> str:
    """Look up the shipping status of a customer's order using their order ID."""
    print(f"DEBUG: AI called get_order_status with ID: '{order_id}'")
    if not order_id or str(order_id).lower() in ["unknown", "none", "null"] or len(str(order_id).strip()) < 4:
        return "Internal AI Error: Stop deciding to look up orders without an ID. Ask the user 'Please provide your 5-digit order ID so I can track it for you.'"
        
    try:
        response = supabase.table("orders").select("*").ilike("id", f"{str(order_id).strip()}%").execute()
        if not response.data:
            return "Order not found. Please verify the order number."
        
        order = response.data[0]
        status = order.get('status', 'Processing')
        tracking = order.get('tracking_url', 'Not available yet')
        total = order.get('total', 'Unknown')
        created_at_str = order.get('created_at')
        
        items = order.get('items', [])
        item_names = []
        if isinstance(items, list):
            for i in items:
                if 'name' in i:
                    item_names.append(f"{i.get('quantity', 1)}x {i['name']}")
                    
        items_str = ", ".join(item_names) if item_names else "Unknown items"
        
        if created_at_str:
            try:
                created_date = datetime.fromisoformat(str(created_at_str).replace('Z', '+00:00'))
            except:
                created_date = datetime.now()
        else:
            created_date = datetime.now()
            
        ship_date = (created_date + timedelta(days=2)).strftime('%B %d, %Y')
        delivery_date = (created_date + timedelta(days=7)).strftime('%B %d, %Y')
        
        return (
            f"Order Status: {status}.\n"
            f"Items: {items_str}.\n"
            f"Total Paid: ₹{total}.\n"
            f"Estimated Shipping Date: {ship_date}.\n"
            f"Estimated Delivery Date: {delivery_date}.\n"
            f"Tracking Link: {tracking}\n"
            f"Provide this link to the user to manage or cancel their order: [Manage Order](/order/{order['id']})"
        )
    except Exception as e:
        return f"Could not look up order: {e}"

def search_products(query: str) -> str:
    """Search for products in the catalog based on a natural language keyword like 'hoops', 'tote', or 'floral'."""
    try:
        # Improved search matching name, category, or description organically
        response = supabase.table("products").select("name, price, id, category").or_(f"name.ilike.%{query}%,category.ilike.%{query}%,description.ilike.%{query}%").limit(5).execute()
        
        if not response.data:
            # Fallback to broader single-word search if strict multi-word query fails
            words = query.split()
            fallback = words[0] if words else ""
            response = supabase.table("products").select("name, price, id, category").or_(f"name.ilike.%{fallback}%,category.ilike.%{fallback}%").limit(5).execute()
            
        if not response.data:
            return "I couldn't find any products perfectly matching that description."
        
        results = [f"Product: {p['name']} (Category: {p.get('category','Unknown')}), Price: ₹{p['price']}, Link: /product/{p['id']}" for p in response.data]
        return " \n".join(results)
    except Exception as e:
        return f"Failed to search products: {e}"

# --------------------------------------------------------
# Setup Groq API Safely using groq package
groq_api_key = os.environ.get("GROQ_API_KEY")
client = None

if groq_api_key and groq_api_key != "INSERT_YOUR_GROQ_KEY_HERE":
    from groq import Groq
    client = Groq(api_key=groq_api_key)

class ChatMessage(BaseModel):
    message: str

chat_sessions = {}

# Map for native execution
native_tools = {
    "get_order_status": get_order_status,
    "search_products": search_products
}

groq_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up the shipping status of an order. CRITICAL: DO NOT call this tool unless the user has explicitly provided a 5-digit number in their message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The unique string ID of the order"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products in the catalog based on a natural language keyword like 'hoops', 'tote', or 'floral'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search term to match against product names"}
                },
                "required": ["query"],
            },
        },
    }
]

import json

@app.post("/api/chat/{session_id}")
def chat_with_bot(session_id: str, payload: ChatMessage):
    global client
    if not client:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key and groq_api_key != "INSERT_YOUR_GROQ_KEY_HERE":
            from groq import Groq
            client = Groq(api_key=groq_api_key)
        else:
            raise HTTPException(status_code=500, detail="Groq API Key is missing. Please add it to the .env file in the backend.")
            
    # Initialize message history
    if session_id not in chat_sessions:
        chat_sessions[session_id] = [
            {
                "role": "system",
                "content": """You are an AI sales and support assistant for an eCommerce website called "Embroidery By Sana".

Your PRIMARY goal is to:
1. Help users find and buy products
2. Increase conversions (suggest products, upsell, cross-sell)
3. Provide fast and helpful support

CORE CAPABILITIES:
- Product discovery (search, recommend, filter)
- Product explanation (price, fabric, use-case, customization)
- Order support (tracking, returns, cancellations)
- Handle complaints professionally
- Guide users through checkout

SALES BEHAVIOR (VERY IMPORTANT):
- Always try to recommend products when user shows buying intent
- Suggest similar or better products
- Use persuasive but natural language
- Highlight benefits (quality, handmade, unique design)

EXAMPLES:
User: "I want a gift"
Bot: "Great choice! 🎁 We have beautiful embroidered gift items like handbags and dupattas. Would you like budget-friendly or premium options?"

User: "Show embroidery kurti"
Bot: "Here are some beautiful embroidered kurtis for you. Would you prefer cotton or party wear styles?"

User: "Too expensive"
Bot: "I understand 😊 I can show you similar designs at a lower price. Let me find some budget-friendly options for you."

TOOL USAGE RULES:
- Use `search_products` whenever user wants to buy, browse, or explore
- Use `get_order_status` ONLY when user provides a valid order ID
- NEVER guess order details
- CRITICAL: Do NOT write `<function>` tags, XML, or raw code in your responses. Always use the native function calling API for tools!

NAVIGATION RULES:
Use ONLY these markdown links to guide users. Do not invent other pages:
[Shop All Products](/shop)
[View Cart](/cart)
[Checkout](/checkout)
[Custom Designs](/custom-design)
[Support](/support)
[Go to Profile](/profile)
[Mehndi Bookings](/mehndi-booking)
[Gallery](/gallery)
[Wishlist](/wishlist)
[Login](/login)
[Home](/)

CONVERSATION STYLE:
- Short, clear, engaging
- Ask questions to guide user
- Be friendly, not robotic

GOAL:
Convert conversations into purchases while helping users efficiently."""
            }
        ]
        
    chat_sessions[session_id].append({"role": "user", "content": payload.message})
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=chat_sessions[session_id],
            tools=groq_tools,
            tool_choice="auto",
            max_tokens=1024
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        # If Groq decides to call a tool
        if tool_calls:
            # We append the assistant's request to the chat history
            chat_sessions[session_id].append(response_message)
            
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = native_tools.get(function_name)
                
                if function_to_call:
                    function_args = json.loads(tool_call.function.arguments)
                    function_response = function_to_call(**function_args)
                    
                    # Append result of tool execution
                    chat_sessions[session_id].append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                    )
                    
            # Send the data back to Groq so it can naturally summarize the data to the user!
            second_response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=chat_sessions[session_id]
            )
            final_reply = second_response.choices[0].message.content
            chat_sessions[session_id].append({"role": "assistant", "content": final_reply})
            return {"reply": final_reply}
            
        else:
            # Standard chat without tool use
            reply_text = response_message.content
            chat_sessions[session_id].append({"role": "assistant", "content": reply_text})
            return {"reply": reply_text}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========================================================
# ✨ AI AUTO-FILL PRODUCT DETAILS
# ========================================================
class AutoFillRequest(BaseModel):
    name: str
    category: str

@app.post("/api/autofill-product")
def autofill_product(payload: AutoFillRequest):
    global client
    if not client:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key and groq_api_key != "INSERT_YOUR_GROQ_KEY_HERE":
            from groq import Groq
            client = Groq(api_key=groq_api_key)
        else:
            raise HTTPException(status_code=500, detail="Groq API Key is missing.")

    prompt = f"""You are a product copywriter for "Embroidery By Sana", an Indian handmade embroidery & fashion brand.

Given the following product:
- Name: {payload.name}
- Category: {payload.category}

Generate the following in **strict JSON format only** (no markdown, no backticks, no explanation):
{{
  "shortDescription": "A compelling 1-2 sentence product summary (max 200 chars)",
  "detailedDescription": "A rich 3-5 sentence product description highlighting craftsmanship, style, and use-case",
  "keyFeatures": ["feature1", "feature2", "feature3", "feature4", "feature5"],
  "metaTitle": "SEO-optimized page title (max 60 chars)",
  "metaDescription": "SEO meta description (max 155 chars)",
  "keywords": "comma,separated,seo,keywords",
  "careInstructions": "Brief care/wash instructions",
  "returnPeriod": 7
}}

Rules:
- Write in a warm, premium, India-focused tone
- Keep descriptions vivid but concise
- keyFeatures must have exactly 5 bullet points
- Return ONLY valid JSON, nothing else"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
        )
        raw = response.choices[0].message.content.strip()
        
        # Clean any markdown wrapping
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()
        
        data = json.loads(raw)
        return data
    except json.JSONDecodeError:
        # Try to extract JSON from mixed content
        import re
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                data = json.loads(match.group())
                return data
            except:
                pass
        raise HTTPException(status_code=500, detail="AI returned invalid JSON. Please try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

