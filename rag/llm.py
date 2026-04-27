"""
llm.py
Calls Mistral via HuggingFace Inference API (free tier).
Falls back to Mistral's own API if HF_TOKEN not set but MISTRAL_API_KEY is.
"""

from typing import List, Dict
import os


# ------------------------------------------------------------------ #
#  Prompt builder                                                      #
# ------------------------------------------------------------------ #

def build_rag_context(question: str, papers: List[Dict], mode: str = "simple") -> tuple:
    """Returns (system_prompt, user_message). mode='simple' for consumer, 'research' for technical."""
    context_blocks = ""
    for i, p in enumerate(papers, 1):
        abstract_snippet = p["text"].replace("Title:", "").replace("Abstract:", "").strip()[:900]
        context_blocks += f"[{i}] {p['title']} ({p['year']})\n{abstract_snippet}\n\n"

    if mode == "simple":
        system = (
            "You are a friendly and caring health information assistant. "
            "Your job is to explain medical research findings in plain, everyday English "
            "that anyone — including someone with no medical background — can easily understand. "
            "Rules you must always follow:\n"
            "- Use simple words. If you must use a medical term, immediately explain it in plain language in parentheses.\n"
            "- Write in a warm, reassuring, conversational tone — like a knowledgeable friend explaining things.\n"
            "- Structure your answer clearly: start with a direct answer, then explain further.\n"
            "- Keep it to 4-6 sentences. Do NOT use citation brackets like [1] or [2] in your text.\n"
            "- End every answer with this exact line: "
            "'⚕️ Always consult a doctor or healthcare professional for personal medical advice.'\n"
            "- Only use information from the research context provided. Never invent facts."
        )
        user = (
            f"RESEARCH CONTEXT (do not mention these are research papers to the user):\n"
            f"{context_blocks}\n"
            f"USER QUESTION: {question}\n\n"
            f"Write a warm, plain-English answer:"
        )
    else:
        system = (
            "You are MedQuery AI, a precise medical research assistant. "
            "Answer questions using ONLY the PubMed research context provided. "
            "Cite sources using bracket numbers like [1], [2]. "
            "Be concise (3-6 sentences) but evidence-based. "
            "Never fabricate medical facts beyond what the abstracts state."
        )
        user = (
            f"--- RESEARCH CONTEXT ---\n{context_blocks}--- END CONTEXT ---\n\n"
            f"QUESTION: {question}\n\n"
            f"Provide a clear, evidence-based answer with citations:"
        )

    return system, user


# keep old function as alias so nothing else breaks
def build_rag_prompt(question: str, papers: List[Dict]) -> str:
    system, user = build_rag_context(question, papers)
    return f"{system}\n\n{user}"


# ------------------------------------------------------------------ #
#  HuggingFace client                                                  #
# ------------------------------------------------------------------ #

class MistralHF:
    """Mistral via HuggingFace free Inference API using chat_completion."""

    MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

    def __init__(self, hf_token: str):
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(token=hf_token)

    def answer(self, question: str, papers: List[Dict], mode: str = "simple") -> str:
        system, user = build_rag_context(question, papers, mode=mode)
        try:
            response = self.client.chat_completion(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=600,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ HuggingFace API error: {e}"


# ------------------------------------------------------------------ #
#  Mistral official API client (optional fallback)                    #
# ------------------------------------------------------------------ #

class MistralOfficial:
    """Mistral via their own API using plain requests — no mistralai package needed."""

    API_URL = "https://api.mistral.ai/v1/chat/completions"
    MODEL   = "mistral-small-latest"

    def __init__(self, api_key: str):
        import requests as req
        self._req = req
        self.api_key = api_key

    def answer(self, question: str, papers: List[Dict], mode: str = "simple") -> str:
        system, user = build_rag_context(question, papers, mode=mode)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "max_tokens": 600,
            "temperature": 0.3,
        }
        try:
            resp = self._req.post(self.API_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"⚠️ Error: {e}"


# ------------------------------------------------------------------ #
#  Factory                                                             #
# ------------------------------------------------------------------ #

def get_llm():
    """Auto-detect which API key is available and return the right client."""
    hf_token = os.getenv("HF_TOKEN")
    mistral_key = os.getenv("MISTRAL_API_KEY")

    if hf_token:
        return MistralHF(hf_token)
    elif mistral_key:
        return MistralOfficial(mistral_key)
    else:
        raise EnvironmentError(
            "No API key found. Set HF_TOKEN or MISTRAL_API_KEY in your .env file."
        )