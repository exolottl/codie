package embeddings

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"time"
	"strings"

	"github.com/sashabaranov/go-openai"
)

// Minimum delay between API calls to avoid rate limiting
const MinDelayMS = 20

// ErrMissingAPIKey is returned when the API key is not set
var ErrMissingAPIKey = errors.New("OPENAI_API_KEY is not set in .env file")

// ErrEmbeddingFailed is returned when the embedding API call fails
var ErrEmbeddingFailed = errors.New("failed to generate embedding")

// GetEmbedding generates an embedding for the given text using OpenAI's API
func GetEmbedding(text string) ([]float32, error) {
	// Add a delay before making the API call to avoid rate limiting
	time.Sleep(MinDelayMS * time.Millisecond)
	
	// Check for empty text
	if text = trimWhitespace(text); text == "" {
		return nil, errors.New("cannot embed empty text")
	}
	
	// Get API key
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, ErrMissingAPIKey
	}
	
	// Create client and request
	client := openai.NewClient(apiKey)
	
	// Create embedding with timeout context
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	resp, err := client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Model: openai.AdaEmbeddingV2,
		Input: []string{text},
	})
	
	// Handle errors
	if err != nil {
		log.Printf("Embedding API error: %v", err)
		return nil, fmt.Errorf("%w: %v", ErrEmbeddingFailed, err)
	}
	
	// Validate response
	if len(resp.Data) == 0 || len(resp.Data[0].Embedding) == 0 {
		return nil, errors.New("received empty embedding from API")
	}
	
	return resp.Data[0].Embedding, nil
}

// GetBatchEmbeddings generates embeddings for multiple texts in batch
func GetBatchEmbeddings(texts []string, batchSize int) (map[string][]float32, error) {
	if batchSize <= 0 {
		batchSize = 20 // Default batch size
	}
	
	// Filter out empty texts
	var validTexts []string
	for _, text := range texts {
		if trimmed := trimWhitespace(text); trimmed != "" {
			validTexts = append(validTexts, trimmed)
		}
	}
	
	if len(validTexts) == 0 {
		return nil, errors.New("no valid texts to embed")
	}
	
	// Get API key
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, ErrMissingAPIKey
	}
	
	client := openai.NewClient(apiKey)
	embeddings := make(map[string][]float32)
	
	// Process texts in batches
	for i := 0; i < len(validTexts); i += batchSize {
		// Add a delay before making each batch API call
		time.Sleep(MinDelayMS * time.Millisecond)
		
		end := min(i+batchSize, len(validTexts))
		batch := validTexts[i:end]
		
		fmt.Printf("Processing batch %d to %d (%d texts)\n", i, end-1, len(batch))
		
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		resp, err := client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
			Model: openai.AdaEmbeddingV2,
			Input: batch,
		})
		cancel()
		
		if err != nil {
			log.Printf("Batch embedding API error: %v", err)
			// Continue to next batch on error
			continue
		}
		
		// Validate and store the embeddings
		if len(resp.Data) > 0 {
			for j, item := range resp.Data {
				if j < len(batch) && len(item.Embedding) > 0 {
					embeddings[batch[j]] = item.Embedding
				}
			}
		}
	}
	
	// Check if we got any embeddings
	if len(embeddings) == 0 {
		return nil, ErrEmbeddingFailed
	}
	
	return embeddings, nil
}

// Helper function to trim whitespace and check for empty strings
func trimWhitespace(s string) string {
	// Custom implementation to trim whitespace while preserving code structure
	if len(s) == 0 {
		return ""
	}
	
	// For code, we want to keep indentation but remove empty lines at start/end
	lines := make([]string, 0)
	inContent := false
	lineCount := 0
	
	for _, line := range strings.Split(s, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			inContent = true
		}
		
		if inContent {
			lines = append(lines, line)
			if trimmed != "" {
				lineCount++
			}
		}
	}
	
	// If we have no non-empty lines, return empty string
	if lineCount == 0 {
		return ""
	}
	
	return strings.Join(lines, "\n")
}

// Helper function for Go <1.21 compatibility
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}