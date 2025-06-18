from typing import Optional, Dict, Any
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from app.optimizations.kv_cache import KVCache
from app.optimizations.batching import BatchManager
from app.optimizations.speculative import SpeculativeDecoder

class LLMEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.kv_cache = KVCache()
        self.batch_manager = BatchManager()
        self.speculative_decoder = SpeculativeDecoder()
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer"""
        model_name = "gpt2"  # Start with a small model for testing
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        stream: bool = False,
        use_speculative: bool = True,
        use_kv_cache: bool = True,
        use_batching: bool = True
    ) -> Dict[str, Any]:
        """Generate text from the model with optimizations"""
        start_time = time.time()
        
        if use_speculative:
            response = self.speculative_decoder.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            response["generation_time_ms"] = int((time.time() - start_time) * 1000)
            response["optimization_stats"] = {
                "speculative_tokens_accepted": response.get("speculative_tokens_accepted", 0),
                "speculative_tokens_rejected": response.get("speculative_tokens_rejected", 0),
                "kv_cache_hits": 0,
                "batch_size": 1
            }
            return response

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Apply KV caching if available
        kv_cache_hits = 0
        if use_kv_cache and self.kv_cache.has_cache():
            inputs = self.kv_cache.apply_cache(inputs)
            kv_cache_hits = 1

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Update KV cache
        if use_kv_cache:
            self.kv_cache.update_cache(outputs)

        # Decode and return
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_generated = len(outputs[0]) - len(inputs["input_ids"][0])
        
        return {
            "text": generated_text,
            "tokens_generated": tokens_generated,
            "generation_time_ms": int((time.time() - start_time) * 1000),
            "optimization_stats": {
                "speculative_tokens_accepted": 0,
                "speculative_tokens_rejected": 0,
                "kv_cache_hits": kv_cache_hits,
                "batch_size": 1
            }
        }

    def clear_cache(self):
        """Clear the KV cache"""
        self.kv_cache.clear() 