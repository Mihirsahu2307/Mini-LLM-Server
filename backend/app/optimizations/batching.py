from typing import List, Dict, Any
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class BatchRequest:
    prompt: str
    max_tokens: int
    temperature: float
    created_at: datetime
    future: asyncio.Future

class BatchManager:
    def __init__(self):
        self.batch_size = 8
        self.max_wait_time = 0.1  # seconds
        self.pending_requests: List[BatchRequest] = []
        self.processing = False

    async def add_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> asyncio.Future:
        """Add a request to the batch and return a future for the result"""
        future = asyncio.Future()
        request = BatchRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            created_at=datetime.now(),
            future=future
        )
        
        self.pending_requests.append(request)
        
        if not self.processing:
            asyncio.create_task(self._process_batch())
            
        return future

    async def _process_batch(self):
        """Process the current batch of requests"""
        self.processing = True
        
        while self.pending_requests:
            # Wait for either batch size or max wait time
            if len(self.pending_requests) >= self.batch_size:
                batch = self.pending_requests[:self.batch_size]
                self.pending_requests = self.pending_requests[self.batch_size:]
            else:
                # Wait for more requests or timeout
                await asyncio.sleep(self.max_wait_time)
                if not self.pending_requests:
                    break
                    
                # Get all requests that have been waiting
                now = datetime.now()
                batch = [
                    req for req in self.pending_requests
                    if now - req.created_at >= timedelta(seconds=self.max_wait_time)
                ]
                self.pending_requests = [
                    req for req in self.pending_requests
                    if req not in batch
                ]
                
            if batch:
                # Process the batch
                # In practice, you'd combine the prompts and process them together
                for request in batch:
                    # Simulate processing
                    result = {"text": f"Processed: {request.prompt}", "tokens_generated": 10}
                    request.future.set_result(result)
                    
        self.processing = False 