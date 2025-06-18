import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

export interface GenerateRequest {
  prompt: string;
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
  use_speculative?: boolean;
  use_kv_cache?: boolean;
  use_batching?: boolean;
}

export interface GenerateResponse {
  text: string;
  tokens_generated: number;
  generation_time_ms: number;
  optimization_stats: {
    speculative_tokens_accepted: number;
    speculative_tokens_rejected: number;
    kv_cache_hits: number;
    batch_size: number;
  };
}

export interface Model {
  id: string;
  name: string;
  description: string;
  context_length: number;
}

const client = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const generateText = async (request: GenerateRequest): Promise<GenerateResponse> => {
  const response = await client.post<GenerateResponse>('/generate', request);
  return response.data;
};

export const listModels = async (): Promise<{ models: Model[] }> => {
  const response = await client.get<{ models: Model[] }>('/models');
  return response.data;
}; 