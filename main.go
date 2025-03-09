package main

import (
	"fmt"
	"log"
	"os"

	"codie/internal/embeddings"
	"codie/internal/fileutils"
	"codie/internal/storage"

	"github.com/joho/godotenv"
)

func main() {
	// Load environment variables
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	if len(os.Args) < 2 {
		log.Fatal("Usage: go run main.go <directory>")
	}
	dir := os.Args[1]
	
	// Get all code files from the directory
	files := fileutils.GetCodeFiles(dir)
	
	// Process each file and get code chunks with embeddings
	chunks := processFiles(files)
	
	// Save the results to a JSON file
	err = storage.SaveToJSON(chunks, "embeddings.json")
	if err != nil {
		log.Fatalf("Failed to save embeddings: %v", err)
	}
	
	fmt.Println("Embeddings saved to embeddings.json")
}

func processFiles(files []string) []storage.CodeChunk {
	var chunks []storage.CodeChunk

	for _, file := range files {
		content, err := os.ReadFile(file)
		if err != nil {
			log.Printf("Failed to read file %s: %v", file, err)
			continue
		}
		
		code := string(content)
		// Split code into chunks
		chunkedCode := fileutils.SplitCodeIntoChunks(code)
		
		// Process each chunk
		for _, chunk := range chunkedCode {
			// Get embedding for the chunk
			embedding := embeddings.GetEmbedding(chunk)
			
			// Create a CodeChunk struct and append to the list
			chunks = append(chunks, storage.CodeChunk{
				File:      file, 
				Content:   chunk, 
				Embedding: embedding,
			})
		}
	}
	
	return chunks
}