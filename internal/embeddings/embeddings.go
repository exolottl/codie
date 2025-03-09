package embeddings

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/sashabaranov/go-openai"
)

// GetEmbedding generates an embedding for the given text using OpenAI's API
// with a 15ms delay to avoid rate limiting
func GetEmbedding(text string) []float32 {
	// Add a delay of 15ms before making the API call
	time.Sleep(15 * time.Millisecond)
	
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY is not set in .env file")
	}
	
	client := openai.NewClient(apiKey)
	fmt.Print(text)
	
	resp, err := client.CreateEmbeddings(context.Background(), openai.EmbeddingRequest{
		Model: openai.AdaEmbeddingV2,
		Input: []string{text},
	})
	
	if err != nil {
		log.Printf("Embedding API error: %v", err)
		return nil
	}
	
	return resp.Data[0].Embedding
}

// GetBatchEmbeddings generates embeddings for multiple texts in batch
// with a 15ms delay between batches
func GetBatchEmbeddings(texts []string, batchSize int) map[string][]float32 {
	if batchSize <= 0 {
		batchSize = 20 // Default batch size
	}
	
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY is not set in .env file")
	}
	
	client := openai.NewClient(apiKey)
	embeddings := make(map[string][]float32)
	
	// Process texts in batches
	for i := 0; i < len(texts); i += batchSize {
		// Add a delay of 15ms before making each batch API call
		time.Sleep(15 * time.Millisecond)
		
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}
		
		batch := texts[i:end]
		fmt.Printf("Processing batch %d to %d\n", i, end-1)
		
		resp, err := client.CreateEmbeddings(context.Background(), openai.EmbeddingRequest{
			Model: openai.AdaEmbeddingV2,
			Input: batch,
		})
		
		if err != nil {
			log.Printf("Batch embedding API error: %v", err)
			continue
		}
		
		// Store the embeddings
		for j, item := range resp.Data {
			if j < len(batch) { // Safety check
				embeddings[batch[j]] = item.Embedding
			}
		}
	}
	
	return embeddings
}