package main

import (
	"fmt"
	"log"
	"os"
	"strings"
	
	"github.com/charmbracelet/glamour"
	"codie/internal/config"
	"codie/internal/embeddings"
	"codie/internal/fileutils"
	"codie/internal/storage"
	"codie/internal/summarization"
)

// Default maximum chunk size for code splitting
const DefaultMaxChunkSize = 8000

// Default embeddings file name
const DefaultEmbeddingsFile = "embeddings.json"

func main() {
	// Initialize configuration with API key validation
	err := config.Init()
	if err != nil {
		log.Fatalf("Configuration error: %v", err)
	}

	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	command := os.Args[1]

	switch command {
	case "help":
		//Print help message
		if len(os.Args) < 2 {
			log.Fatal("Usage: go run main.go help <command>")
		}
		printUsage()

	case "index":
		// Check if directory is provided
		if len(os.Args) < 3 {
			log.Fatal("Usage: go run main.go index <directory>")
		}
		dir := os.Args[2]
		indexCodebase(dir)

	case "summarize":
		// Check if directory is provided
		if len(os.Args) < 3 {
			log.Fatal("Usage: go run main.go summarize <directory> [options]")
		}
		dir := os.Args[2]
		summarizeCodebase(dir, os.Args[3:])

	default:
		// For backward compatibility, treat the first arg as directory
		// if it doesn't match a known command
		dir := os.Args[1]
		indexCodebase(dir)
	}
}

// printUsage prints the usage information
func printUsage() {
	fmt.Println("Usage:")
	fmt.Println("  go run main.go index <directory>     - Index a codebase")
	fmt.Println("  go run main.go summarize <directory> - Generate a summary of a codebase")
	fmt.Println("    Options:")
	fmt.Println("      --detail=<level>   - Set detail level (brief, standard, comprehensive)")
	fmt.Println("      --focus=<path>     - Focus on a specific directory")
	fmt.Println("      --no-metrics       - Exclude code quality metrics")
}

// indexCodebase processes and indexes a codebase directory
func indexCodebase(dir string) {
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
	err = storage.SaveToJSON(chunks, DefaultEmbeddingsFile)
	if err != nil {
		log.Fatalf("Failed to save embeddings: %v", err)
	}
	
	fmt.Printf("Successfully processed %d code chunks\n", len(chunks))
	fmt.Printf("Embeddings saved to %s\n", DefaultEmbeddingsFile)
}

// summarizeCodebase generates a summary of the codebase
func summarizeCodebase(dir string, args []string) {
	embeddingsPath := DefaultEmbeddingsFile

	// Check if embeddings file exists
	_, err := os.Stat(embeddingsPath)
	if os.IsNotExist(err) {
		fmt.Println("Embeddings file not found. Indexing codebase first...")
		indexCodebase(dir)
	}

	// Parse options
	options := summarization.DefaultSummaryOptions()
	
	for _, arg := range args {
		if strings.HasPrefix(arg, "--detail=") {
			options.DetailLevel = strings.TrimPrefix(arg, "--detail=")
		} else if strings.HasPrefix(arg, "--focus=") {
			options.FocusPath = strings.TrimPrefix(arg, "--focus=")
		} else if arg == "--no-metrics" {
			options.IncludeMetrics = false
		}
	}

	// Generate summary
	fmt.Println("Generating codebase summary...")
	summary, err := summarization.GenerateRepoSummary(embeddingsPath, options)
	if err != nil {
		log.Fatalf("Failed to generate summary: %v", err)
	}

	// Output the summary
	fmt.Println("\n--- CODEBASE SUMMARY ---")
	output, _:= glamour.Render(summary, "dark")
	fmt.Println(output)
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