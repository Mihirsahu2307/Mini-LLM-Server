import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  VStack,
  Input,
  Button,
  Text,
  useToast,
  Container,
  Heading,
  Flex,
  Spinner,
  Switch,
  FormControl,
  FormLabel,
  HStack,
  Badge,
  Tooltip,
} from '@chakra-ui/react';
import { generateText } from '../api/client';
import type { GenerateRequest } from '../api/client';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timing?: {
    generation_time_ms: number;
    tokens_generated: number;
    optimization_stats: {
      speculative_tokens_accepted: number;
      speculative_tokens_rejected: number;
      kv_cache_hits: number;
      batch_size: number;
    };
  };
}

export const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [optimizations, setOptimizations] = useState({
    use_speculative: true,
    use_kv_cache: true,
    use_batching: true,
  });
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const toast = useToast();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const request: GenerateRequest = {
        prompt: input,
        max_tokens: 100,
        temperature: 0.7,
        ...optimizations,
      };

      const response = await generateText(request);
      const assistantMessage: Message = {
        role: 'assistant',
        content: response.text,
        timing: {
          generation_time_ms: response.generation_time_ms,
          tokens_generated: response.tokens_generated,
          optimization_stats: response.optimization_stats,
        },
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to generate response',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const OptimizationSwitch = ({ label, isChecked, onChange, tooltip }: {
    label: string;
    isChecked: boolean;
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
    tooltip: string;
  }) => (
    <Tooltip label={tooltip}>
      <FormControl display="flex" alignItems="center">
        <FormLabel htmlFor={label} mb="0">
          {label}
        </FormLabel>
        <Switch id={label} isChecked={isChecked} onChange={onChange} />
      </FormControl>
    </Tooltip>
  );

  return (
    <Container maxW="container.md" h="100vh" py={4}>
      <VStack h="full" spacing={4}>
        <Heading size="lg">Mini LLM Chat</Heading>
        
        <Box w="full" p={4} borderWidth={1} borderRadius="md" bg="white">
          <VStack spacing={2} align="stretch">
            <Heading size="sm">Optimizations</Heading>
            <OptimizationSwitch
              label="Speculative Decoding"
              isChecked={optimizations.use_speculative}
              onChange={(e) => setOptimizations(prev => ({ ...prev, use_speculative: e.target.checked }))}
              tooltip="Uses a smaller model to predict tokens and a larger model to verify them"
            />
            <OptimizationSwitch
              label="KV Cache"
              isChecked={optimizations.use_kv_cache}
              onChange={(e) => setOptimizations(prev => ({ ...prev, use_kv_cache: e.target.checked }))}
              tooltip="Caches key-value pairs to speed up repeated token generation"
            />
            <OptimizationSwitch
              label="Batching"
              isChecked={optimizations.use_batching}
              onChange={(e) => setOptimizations(prev => ({ ...prev, use_batching: e.target.checked }))}
              tooltip="Processes multiple requests together for better throughput"
            />
          </VStack>
        </Box>
        
        <Box
          flex={1}
          w="full"
          overflowY="auto"
          borderWidth={1}
          borderRadius="md"
          p={4}
          bg="white"
        >
          <VStack spacing={4} align="stretch">
            {messages.map((message, index) => (
              <Box
                key={index}
                bg={message.role === 'user' ? 'blue.50' : 'gray.50'}
                p={3}
                borderRadius="md"
                alignSelf={message.role === 'user' ? 'flex-end' : 'flex-start'}
                maxW="80%"
              >
                <Text>{message.content}</Text>
                {message.timing && (
                  <Box mt={2}>
                    <HStack spacing={2} fontSize="sm" color="gray.600">
                      <Badge colorScheme="blue">
                        {message.timing.generation_time_ms}ms
                      </Badge>
                      <Badge colorScheme="green">
                        {message.timing.tokens_generated} tokens
                      </Badge>
                      {message.timing.optimization_stats.speculative_tokens_accepted > 0 && (
                        <Badge colorScheme="purple">
                          {message.timing.optimization_stats.speculative_tokens_accepted} accepted
                        </Badge>
                      )}
                      {message.timing.optimization_stats.speculative_tokens_rejected > 0 && (
                        <Badge colorScheme="red">
                          {message.timing.optimization_stats.speculative_tokens_rejected} rejected
                        </Badge>
                      )}
                    </HStack>
                  </Box>
                )}
              </Box>
            ))}
            {isLoading && (
              <Flex justify="center">
                <Spinner />
              </Flex>
            )}
            <div ref={messagesEndRef} />
          </VStack>
        </Box>

        <form onSubmit={handleSubmit} style={{ width: '100%' }}>
          <Flex gap={2}>
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              disabled={isLoading}
              bg="white"
            />
            <Button
              type="submit"
              colorScheme="blue"
              isLoading={isLoading}
              loadingText="Generating..."
            >
              Send
            </Button>
          </Flex>
        </form>
      </VStack>
    </Container>
  );
}; 