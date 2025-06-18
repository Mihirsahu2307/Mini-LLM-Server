# Mini-LLM-Server
 Minimal server for local LLMs equipped with several inference optimization techniques that can be configured via a UI. 
 
 Project is still under development. Few optimizations have been added, few more to come.


## Features (some are TODO)
- Batching (static and dynamic)
- KV Caching
- Prompt Caching (Prefix Caching)
- Tensor Parallelism
- Speculative Decoding


## Papers read for the optimizations
- ZeRO: https://arxiv.org/pdf/1910.02054
- Speculative Decoding: https://arxiv.org/pdf/2211.17192
- Self-Speculative Decoding: https://arxiv.org/pdf/2309.08168
- Survey blog: https://www.aussieai.com/research/inference-optimization
- Survey paper: https://arxiv.org/pdf/2404.14294 (pretty exhaustive)


## Project Structure

```
.
├── backend/                 # Python FastAPI server
│   ├── app/                # Main application code
│   │   ├── core/          # Core LLM functionality
│   │   ├── optimizations/ # LLM optimization techniques
│   │   └── api/           # API endpoints
│   ├── tests/             # Backend tests
│   └── Dockerfile         # Backend Docker configuration
├── frontend/               # TypeScript/React frontend
│   ├── src/               # Source code
│   │   ├── components/    # React components
│   │   ├── hooks/        # Custom React hooks
│   │   └── types/        # TypeScript type definitions
│   └── Dockerfile         # Frontend Docker configuration
└── docker-compose.yml     # Docker compose configuration
```