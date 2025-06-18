from typing import Dict, Any, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SpeculativeDecoder:
    def __init__(self, draft_model_name: str = "gpt2", target_model_name: str = "gpt2-medium"):
        self.draft_model = None
        self.target_model = None
        self.draft_tokenizer = None
        self.target_tokenizer = None
        self.max_draft_tokens = 5  # Maximum number of tokens to generate with draft model
        self._load_models(draft_model_name, target_model_name)

    def _load_models(self, draft_model_name: str, target_model_name: str):
        """Load both draft and target models"""
        # Load draft model (smaller, faster)
        self.draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
        self.draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name)
        
        # Load target model (larger, more accurate)
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        self.target_model = AutoModelForCausalLM.from_pretrained(target_model_name)
        
        # Move models to GPU if available
        if torch.cuda.is_available():
            self.draft_model = self.draft_model.cuda()
            self.target_model = self.target_model.cuda()

    def _get_draft_predictions(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate draft predictions using the smaller model"""
        with torch.no_grad():
            draft_outputs = self.draft_model.generate(
                input_ids,
                max_new_tokens=min(self.max_draft_tokens, max_new_tokens),
                do_sample=True,
                pad_token_id=self.draft_tokenizer.eos_token_id
            )
        return draft_outputs

    def _verify_predictions(
        self,
        input_ids: torch.Tensor,
        draft_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int, int]:
        """Verify draft predictions using the target model"""
        with torch.no_grad():
            # Get target model's predictions for the same sequence
            target_outputs = self.target_model.generate(
                input_ids,
                max_new_tokens=draft_outputs.shape[1] - input_ids.shape[1],
                do_sample=True,
                pad_token_id=self.target_tokenizer.eos_token_id
            )
            
            # Compare predictions
            draft_tokens = draft_outputs[0, input_ids.shape[1]:]
            target_tokens = target_outputs[0, input_ids.shape[1]:]
            
            # Find the first mismatch
            match_mask = (draft_tokens == target_tokens)
            if match_mask.all():
                return target_outputs, len(draft_tokens), len(draft_tokens), 0
            
            first_mismatch = match_mask.argmin().item()
            accepted_tokens = first_mismatch
            rejected_tokens = len(draft_tokens) - accepted_tokens
            return target_outputs[:, :input_ids.shape[1] + first_mismatch + 1], accepted_tokens, accepted_tokens, rejected_tokens

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate text using speculative decoding"""
        # Tokenize input
        input_ids = self.target_tokenizer(prompt, return_tensors="pt")["input_ids"]
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        total_tokens = 0
        current_input = input_ids
        total_accepted = 0
        total_rejected = 0

        while total_tokens < max_tokens:
            # Get draft predictions
            draft_outputs = self._get_draft_predictions(current_input, max_tokens - total_tokens)
            
            # Verify predictions
            verified_outputs, accepted_tokens, accepted, rejected = self._verify_predictions(current_input, draft_outputs)
            
            # Update for next iteration
            current_input = verified_outputs
            total_tokens += accepted_tokens
            total_accepted += accepted
            total_rejected += rejected
            
            # Break if we've generated enough tokens or hit EOS
            if total_tokens >= max_tokens or verified_outputs[0, -1] == self.target_tokenizer.eos_token_id:
                break

        # Decode and return
        generated_text = self.target_tokenizer.decode(verified_outputs[0], skip_special_tokens=True)
        return {
            "text": generated_text,
            "tokens_generated": total_tokens,
            "speculative_tokens_accepted": total_accepted,
            "speculative_tokens_rejected": total_rejected
        } 