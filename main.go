package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/joho/godotenv"
	"github.com/sashabaranov/go-openai"
)

type CodeChunk struct {
	File    string `json:"file"`
	Content string `json:"content"`
	Embedding []float32 `json:"embedding"`
}

func main() {
	if len(os.Args) < 2 {
		log.Fatal("Usage: go run main.go <directory>")
	}
	dir := os.Args[1]
	files := getCodeFiles(dir)
	var chunks []CodeChunk

	for _, file := range files {
		content, err := os.ReadFile(file)
		if err != nil {
			log.Printf("Failed to read file %s: %v", file, err)
			continue
		}
		code := string(content)
		// Simple chunking by functions and classes (improve with regex/AST parsing)
		chunkedCode := splitCodeIntoChunks(code)
		
		for _, chunk := range chunkedCode {
			embedding := getEmbedding(chunk)
			chunks = append(chunks, CodeChunk{File: file, Content: chunk, Embedding: embedding})
		}
	}

	output, _ := json.MarshalIndent(chunks, "", "  ")
	os.WriteFile("embeddings.json", output, 0644)
	fmt.Println("Embeddings saved to embeddings.json")
}

func getCodeFiles(root string) []string {
	var files []string
	extensions := []string{".py", ".js", ".ts", ".cpp", ".go", ".java", ".lua"}
	filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() {
			for _, ext := range extensions {
				if strings.HasSuffix(info.Name(), ext) {
					files = append(files, path)
					break
				}
			}
		}
		return nil
	})
	return files
}

func splitCodeIntoChunks(code string) []string {
	return strings.Split(code, "\n\n")
}

func getEmbedding(text string) []float32 {
  err := godotenv.Load()
  if err != nil {
    log.Fatal("Error loading .env file")
  }
  fmt.Printf(text)
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
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
