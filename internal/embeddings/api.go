package embeddings

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/sashabaranov/go-openai"
)

// batchResult is used to collect results from embedding API calls
type batchResult struct {
	Texts      []string
	StartIndex int
	Embeddings [][]float32
	Error      error
}

// GetEmbedding generates an embedding for the given text using OpenAI's API
// This is kept for backward compatibility but uses GetBatchEmbeddings internally
func GetEmbedding(text string) ([]float32, error) {
	// Use batch embeddings with a batch of 1
	embeddingMap, err := GetBatchEmbeddings([]string{text}, 1)
	if err != nil {
		return nil, err
	}
	
	if embedding, ok := embeddingMap[text]; ok {
		return embedding, nil
	}
	
	return nil, ErrEmbeddingFailed
}

// GetBatchEmbeddings generates embeddings for multiple texts in batch
func GetBatchEmbeddings(texts []string, batchSize int) (map[string][]float32, error) {
	if batchSize <= 0 {
		batchSize = 20 // Default batch size
	}
	
	// Filter out empty texts and check for length
	var validTexts []string
	var originalTexts []string // Keep track of original texts in same order
	var invalidCount int
	
	for _, text := range texts {
		if trimmed := trimWhitespace(text); trimmed != "" && len(trimmed)/4 <= MaxTokenLimit {
			validTexts = append(validTexts, trimmed)
			originalTexts = append(originalTexts, text) // Store original text
		} else if trimmed != "" {
			log.Printf("Warning: Text too long for embedding API, skipping (%d approximate tokens)", len(trimmed)/4)
			invalidCount++
		} else {
			invalidCount++
		}
	}
	
	if len(validTexts) == 0 {
		return nil, errors.New("no valid texts to embed")
	}
	
	if invalidCount > 0 {
		log.Printf("Warning: Skipped %d texts due to empty content or exceeding token limit", invalidCount)
	}
	
	// Get API key
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, ErrMissingAPIKey
	}
	
	client := openai.NewClient(apiKey)
	embeddings := make(map[string][]float32)
	
	// Create channels for concurrent processing
	resultChan := make(chan batchResult, (len(validTexts)+batchSize-1)/batchSize)
	var wg sync.WaitGroup
	
	// Process texts in batches
	for i := 0; i < len(validTexts); i += batchSize {
		end := min(i+batchSize, len(validTexts))
		batch := validTexts[i:end]
		
		wg.Add(1)
		go func(startIdx int, textBatch []string) {
			defer wg.Done()
			
			var result batchResult
			result.Texts = textBatch
			result.StartIndex = startIdx
			
			// Wait for rate limiter
			apiRateLimiter.Wait()
			defer apiRateLimiter.Release()
			
			// Try up to 3 times with increasing backoff
			var resp openai.EmbeddingResponse
			var err error
			var success bool
			
			for attempt := 1; attempt <= 3; attempt++ {
				ctx, cancel := context.WithTimeout(context.Background(), DefaultAPITimeout)
				resp, err = client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
					Model: openai.SmallEmbedding3,
					Input: textBatch,
				})
				cancel()
				
				if err == nil {
					success = true
					break
				}
				
				// Check if we need to back off due to rate limiting
				if strings.Contains(strings.ToLower(err.Error()), "rate limit") {
					log.Printf("Rate limit hit, backing off for attempt %d", attempt)
					time.Sleep(time.Duration(4<<attempt) * time.Second)
				} else if attempt < 3 {
					// For other errors, use standard backoff
					backoffTime := time.Duration(1<<(attempt-1)) * time.Second
					time.Sleep(backoffTime)
				}
			}
			
			if !success {
				result.Error = fmt.Errorf("batch embedding failed after retries: %w", err)
				resultChan <- result
				return
			}
			
			// Extract embeddings
			if len(resp.Data) > 0 {
				for _, item := range resp.Data {
					if len(item.Embedding) > 0 {
						result.Embeddings = append(result.Embeddings, item.Embedding)
					}
				}
			}
			
			resultChan <- result
		}(i, batch)
	}
	
	// Close result channel when all goroutines finish
	go func() {
		wg.Wait()
		close(resultChan)
	}()
	
	// Collect results
	var errors []error
	for result := range resultChan {
		if result.Error != nil {
			errors = append(errors, result.Error)
			continue
		}
		
		// Match embeddings with their original texts
		for j, embedding := range result.Embeddings {
			if j < len(result.Texts) {
				originalIndex := result.StartIndex + j
				if originalIndex < len(originalTexts) {
					embeddings[originalTexts[originalIndex]] = embedding
				}
			}
		}
	}
	
	// Check if we got any embeddings
	if len(embeddings) == 0 {
		if len(errors) > 0 {
			return nil, fmt.Errorf("all embedding batches failed: %v", errors[0])
		}
		return nil, ErrEmbeddingFailed
	}
	
	// Return partial results with a warning if some failed
	if len(embeddings) < len(validTexts) {
		log.Printf("Warning: Only generated %d/%d embeddings successfully", len(embeddings), len(validTexts))
	}
	
	return embeddings, nil
}