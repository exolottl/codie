package main

import (
	"fmt"
	"log"
	"os"

	"codie/internal/config"
	"codie/internal/embeddings"
	"codie/internal/fileutils"
	"codie/internal/storage"
)

// Default maximum chunk size for code splitting
const DefaultMaxChunkSize = 8000

func main() {
	// Initialize configuration
	err := config.Init()
	if err != nil {
		log.Fatalf("Failed to initialize configuration: %v", err)
	}

	if len(os.Args) < 2 {
		log.Fatal("Usage: go run main.go <directory>")
	}
	dir := os.Args[1]
	
	// Get all code files from the directory
	files, err := fileutils.GetCodeFiles(dir)
	if err != nil {
		log.Fatalf("Error scanning directory: %v", err)
	}
	
	if len(files) == 0 {
		log.Fatal("No code files found in the specified directory")
	}
	
	fmt.Printf("Found %d code files to process\n", len(files))
	
	// Process each file and get code chunks with embeddings
	chunks, err := processFiles(files)
	if err != nil {
		log.Fatalf("Error processing files: %v", err)
	}
	
	if len(chunks) == 0 {
		log.Fatal("No code chunks were processed successfully")
	}
	
	// Save the results to a JSON file
	err = storage.SaveToJSON(chunks, "embeddings.json")
	if err != nil {
		log.Fatalf("Failed to save embeddings: %v", err)
	}
	
	fmt.Printf("Successfully processed %d code chunks\n", len(chunks))
	fmt.Println("Embeddings saved to embeddings.json")
}

func processFiles(files []string) ([]storage.CodeChunk, error) {
	var chunks []storage.CodeChunk
	totalFiles := len(files)
	
	for i, file := range files {
		fmt.Printf("Processing file %d/%d: %s\n", i+1, totalFiles, file)
		
		content, err := fileutils.ReadFileContent(file)
		if err != nil {
			log.Printf("Failed to read file %s: %v", file, err)
			continue
		}
		
		// Split code into chunks
		chunkedCode := fileutils.SplitCodeIntoChunks(content, DefaultMaxChunkSize)
		fmt.Printf("  Split into %d chunks\n", len(chunkedCode))
		
		// Process each chunk
		for j, chunk := range chunkedCode {
			// Get embedding for the chunk
			embedding, err := embeddings.GetEmbedding(chunk)
			if err != nil {
				log.Printf("  Failed to get embedding for chunk %d in %s: %v", j+1, file, err)
				continue
			}
			
			// Create a CodeChunk struct and append to the list
			chunks = append(chunks, storage.CodeChunk{
				File:      file, 
				Content:   chunk, 
				Embedding: embedding,
			})
		}
	}
	
	return chunks, nil
}