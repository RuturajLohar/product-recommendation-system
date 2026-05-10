import json
import logging
from typing import List, Dict, Optional
from groq import Groq
import google.generativeai as genai
from ..core.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.groq_enabled = bool(settings.GROQ_API_KEY)
        self.gemini_enabled = bool(settings.GEMINI_API_KEY)
        
        if self.groq_enabled:
            self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        
        if self.gemini_enabled:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')

    async def rerank_with_groq(self, query: str, candidates: List[Dict], user_context: Optional[str] = None) -> List[Dict]:
        """
        Uses Groq's speed to rerank a list of candidates based on deep reasoning.
        """
        if not self.groq_enabled:
            return candidates

        # Prepare a lightweight prompt for Groq
        items_text = "\n".join([f"- {i['asin']}: {i['title']} (${i['price']})" for i in candidates[:20]])
        
        prompt = f"""
        Analyze the following products for a user who is interested in: "{query}".
        User Context: {user_context or "Standard consumer"}
        
        Products:
        {items_text}
        
        Rerank the top 10 most relevant products. 
        Return ONLY a JSON list of ASIN strings in order of relevance.
        Example: ["B001", "B002", ...]
        """

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            response_content = chat_completion.choices[0].message.content
            # The model might wrap the list in an object
            data = json.loads(response_content)
            reranked_asins = data if isinstance(data, list) else list(data.values())[0]
            
            # Reorder original objects
            asin_map = {item['asin']: item for item in candidates}
            result = [asin_map[asin] for asin in reranked_asins if asin in asin_map]
            
            # Add remaining candidates that weren't picked by the LLM
            picked_asins = set(reranked_asins)
            result.extend([item for item in candidates if item['asin'] not in picked_asins])
            
            return result
        except Exception as e:
            logger.error(f"Groq Reranking failed: {e}")
            return candidates

    async def explain_with_gemini(self, product_title: str, user_interest: str) -> str:
        """
        Uses Gemini's deep reasoning to explain WHY a product was recommended.
        """
        if not self.gemini_enabled:
            return "Recommended based on your recent activity."

        prompt = f"""
        Briefly explain (max 15 words) why a user interested in "{user_interest}" 
        would like this product: "{product_title}".
        Make it persuasive and personalized.
        """

        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini Explanation failed: {e}")
            return "Fits your interest in this category."

llm_service = LLMService()
