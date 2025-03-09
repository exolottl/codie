package embeddings

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/sashabaranov/go-openai"
)

// GetEmbedding generates an embedding for the given text using OpenAI's API
func GetEmbedding(text string) []float32 {
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