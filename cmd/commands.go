package cmd

import (
	"context"
	"fmt"
	"log"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"
  "encoding/json"

	"codie/internal/embeddings"
	"codie/internal/fileutils"
	"codie/internal/storage"
	"codie/internal/summarization"
	"github.com/charmbracelet/glamour"
	"github.com/redis/go-redis/v9"
	"github.com/schollz/progressbar/v3"
)

// Default maximum chunk size for code splitting
const DefaultMaxChunkSize = 8000

// Default embeddings file name
const DefaultEmbeddingsFile = "embeddings.json"

// Default batch size for sending embeddings to API
const DefaultBatchSize = 20

// Default number of worker goroutines (0 means use NumCPU)
const DefaultNumWorkers = 0

// PrintUsage prints the usage information
func PrintUsage() {
	fmt.Println("Usage:")
	fmt.Println("  go run main.go index <directory>     - Index a codebase")
	fmt.Println("  go run main.go summarize <directory> - Generate a summary of a codebase")
	fmt.Println("    Options:")
	fmt.Println("      --detail=<level>   - Set detail level (brief, standard, comprehensive)")
	fmt.Println("      --focus=<path>     - Focus on a specific directory")
	fmt.Println("      --no-metrics       - Exclude code quality metrics")
}

var ctx = context.Background()

// IndexCodebase processes and indexes a codebase directory
func IndexCodebase(dir string) {
	// Track execution time
	startTime := time.Now()
	
	// Get all code files from the directory
	files, err := fileutils.GetCodeFiles(dir)
	if err != nil {
		log.Fatalf("Error scanning directory: %v", err)
	}
	if len(files) == 0 {
		log.Fatal("No code files found in the specified directory")
	}
	fmt.Printf("Found %d code files to process\n", len(files))
	
	// Determine number of workers based on CPU cores
	numWorkers := DefaultNumWorkers
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU() * 4
	}
	
	// Set up Redis client
	ctx := context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "redislocal",
		DB:       0,
		PoolSize: numWorkers + 2, // Match pool size to worker count
	})
	
	// Test Redis connection
	_, err = rdb.Ping(ctx).Result()
	if err != nil {
		log.Fatalf("Failed to connect to Redis: %v", err)
	}
	defer rdb.Close()
	
	// Clear previous data
	rdb.Del(ctx, "codebase:chunks")
	
	// Set up concurrency channels and wait groups
	filesChan := make(chan string, len(files))
	resultsChan := make(chan []storage.CodeChunk, len(files))
	errorsChan := make(chan error, len(files))
	
	// Create a progress bar
	bar := progressbar.NewOptions(len(files),
		progressbar.OptionSetDescription("Processing files"),
		progressbar.OptionShowCount(),
		progressbar.OptionShowIts(),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer:        "=",
			SaucerHead:    ">",
			SaucerPadding: " ",
			BarStart:      "[",
			BarEnd:        "]",
		}))
	
	// Launch worker pool
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for file := range filesChan {
				chunks, err := processFile(file)
				if err != nil {
					errorsChan <- fmt.Errorf("error processing %s: %w", file, err)
				} else {
					resultsChan <- chunks
				}
				bar.Add(1)
			}
		}()
	}
	
	// Queue files for processing
	for _, file := range files {
		filesChan <- file
	}
	close(filesChan)
	
	// Set up collector goroutines with proper synchronization
	var collectorWg sync.WaitGroup
	collectorWg.Add(2)
	
	// Error collector
	var processingErrors []error
	var errorMutex sync.Mutex
	go func() {
		defer collectorWg.Done()
		for err := range errorsChan {
			errorMutex.Lock()
			processingErrors = append(processingErrors, err)
			errorMutex.Unlock()
		}
	}()
	
	// Results collector that writes to Redis
	totalChunksCount := 0
	var countMutex sync.Mutex
	go func() {
		defer collectorWg.Done()
		for chunks := range resultsChan {
			if len(chunks) == 0 {
				continue
			}
			
			pipe := rdb.Pipeline()
			
			for _, chunk := range chunks {
				// Generate a unique ID for the chunk
				chunkID := fmt.Sprintf("chunk:%s:%d", chunk.File, totalChunksCount)
				
				// Store embedding as a binary string (more efficient)
				embeddingBytes, err := json.Marshal(chunk.Embedding)
				if err != nil {
					errorsChan <- fmt.Errorf("error serializing embedding: %w", err)
					continue
				}
				
				// Store in Redis with your actual fields
				pipe.HSet(ctx, chunkID, map[string]interface{}{
					"file":      chunk.File,
					"content":   chunk.Content,
					"embedding": embeddingBytes,
				})
				
				// Add to the index set
				pipe.SAdd(ctx, "codebase:chunks", chunkID)
			}
			
			// Execute Redis pipeline
			_, err := pipe.Exec(ctx)
			if err != nil {
				errorsChan <- fmt.Errorf("error writing to Redis: %w", err)
				continue
			}
			
			// Update chunk count
			countMutex.Lock()
			totalChunksCount += len(chunks)
			countMutex.Unlock()
		}
	}()
	
	// Wait for all workers to finish
	wg.Wait()
	close(resultsChan)
	close(errorsChan)
	
	// Wait for collectors to finish
	collectorWg.Wait()
	
	// Report errors (but continue with saving results)
	if len(processingErrors) > 0 {
		fmt.Printf("\nEncountered %d errors during processing:\n", len(processingErrors))
		for i, err := range processingErrors {
			if i < 10 { // Only show first 10 errors
				fmt.Printf("- %v\n", err)
			} else {
				fmt.Printf("- ... and %d more errors\n", len(processingErrors)-10)
				break
			}
		}
	}
	
	// Report results
	if totalChunksCount > 0 {
		fmt.Printf("\nSuccessfully stored %d code chunks in Redis\n", totalChunksCount)
	} else {
		log.Fatal("No code chunks were processed successfully")
	}
	
	elapsedTime := time.Since(startTime)
	fmt.Printf("Total indexing time: %v\n", elapsedTime)
}

// processFile handles a single file, extracting and embedding its chunks
func processFile(file string) ([]storage.CodeChunk, error) {
	content, err := fileutils.ReadFileContent(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	// Split code into chunks
	chunkedCode := fileutils.SplitCodeIntoChunks(content, DefaultMaxChunkSize)
	if len(chunkedCode) == 0 {
		return nil, nil // No valid chunks found
	}

	// Prepare data for batch processing
	var chunksToEmbed []string
	fileChunks := make([]storage.CodeChunk, len(chunkedCode))

	for i, chunk := range chunkedCode {
		chunksToEmbed = append(chunksToEmbed, chunk)
		fileChunks[i] = storage.CodeChunk{
			File:    file,
			Content: chunk,
			// Embedding will be added later
		}
	}

	// Get embeddings for all chunks in batch
	embedMap, err := embeddings.GetBatchEmbeddings(chunksToEmbed, DefaultBatchSize)
	if err != nil {
		return nil, fmt.Errorf("failed to get embeddings: %w", err)
	}

	// Associate embeddings with their chunks
	var validChunks []storage.CodeChunk
	for i, chunk := range fileChunks {
		if embedding, ok := embedMap[chunksToEmbed[i]]; ok {
			chunk.Embedding = embedding
			validChunks = append(validChunks, chunk)
		}
	}

	return validChunks, nil
}

// SummarizeCodebase generates a summary of the codebase
func SummarizeCodebase(dir string, args []string) {
	start := time.Now()
	embeddingsPath := DefaultEmbeddingsFile

	// Check if embeddings file exists
	_, err := os.Stat(embeddingsPath)
	if os.IsNotExist(err) {
		fmt.Println("Embeddings file not found. Indexing codebase first...")
		IndexCodebase(dir)
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
	output, _ := glamour.Render(summary, "dark")
	fmt.Println(output)
	elapsedTime := time.Since(start)
	fmt.Printf("Total summarizing time: %v\n", elapsedTime)

}
