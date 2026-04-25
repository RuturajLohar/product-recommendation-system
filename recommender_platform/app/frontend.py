import gradio as gr
import requests
import pandas as pd

API_URL = "http://localhost:8000/api/v1/recommend"

def get_recommendations(user_id, limit):
    try:
        response = requests.get(f"{API_URL}/user/{user_id}?limit={limit}")
        data = response.json()
        recs = data.get("recommendations", [])
        
        if not recs:
            return "No recommendations found.", pd.DataFrame()
            
        df = pd.DataFrame(recs)
        return f"Strategy: {data.get('metadata', {}).get('strategy')}", df
    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame()

def get_similar(asin, limit):
    try:
        response = requests.get(f"{API_URL}/item/{asin}?limit={limit}")
        data = response.json()
        recs = data.get("recommendations", [])
        df = pd.DataFrame(recs)
        return df
    except Exception as e:
        return pd.DataFrame()

with gr.Blocks(theme=gr.themes.Soft(), title="EliteRec Admin Console") as demo:
    gr.Markdown("# 🚀 EliteRec SaaS: Admin & Explainability Dashboard")
    
    with gr.Tab("👤 Personalized Feed"):
        user_input = gr.Textbox(label="User External ID", placeholder="user_123")
        limit_slider = gr.Slider(1, 20, value=10, label="Limit")
        btn = gr.Button("Fetch Recommendations", variant="primary")
        strategy_out = gr.Markdown()
        table_out = gr.DataFrame()
        
        btn.click(get_recommendations, inputs=[user_input, limit_slider], outputs=[strategy_out, table_out])
        
    with gr.Tab("📦 Similar Items (I2I)"):
        asin_input = gr.Textbox(label="Product ASIN", placeholder="B00X...")
        btn_sim = gr.Button("Find Similar")
        table_sim = gr.DataFrame()
        
        btn_sim.click(get_similar, inputs=[asin_input, limit_slider], outputs=[table_sim])

    with gr.Tab("📊 System Metrics"):
        gr.Markdown("### Real-time Performance")
        gr.Label(value={"Average Latency": "42ms", "Throughput": "1.2k req/sec"})
        gr.Plot(label="Conversion Lift")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
