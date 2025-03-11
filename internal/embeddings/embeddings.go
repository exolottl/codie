package embeddings

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/sashabaranov/go-openai"
	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/golang"
	"github.com/smacker/go-tree-sitter/javascript"
	"github.com/smacker/go-tree-sitter/python"
)

// RateLimiter manages rate limiting for API calls
type RateLimiter struct {
	ticker    *time.Ticker
	mu        sync.Mutex
	semaphore chan struct{}
}

// NewRateLimiter creates a new rate limiter with the specified requests per minute
func NewRateLimiter(requestsPerMinute int, maxConcurrent int) *RateLimiter {
	if requestsPerMinute <= 0 {
		requestsPerMinute = 60 // Default: 1 per second
	}
	if maxConcurrent <= 0 {
		maxConcurrent = 5 // Default: 5 concurrent requests
	}
	
	interval := time.Minute / time.Duration(requestsPerMinute)
	return &RateLimiter{
		ticker:    time.NewTicker(interval),
		semaphore: make(chan struct{}, maxConcurrent),
	}
}

// Wait blocks until a request can be made according to rate limits
func (r *RateLimiter) Wait() {
	r.semaphore <- struct{}{} // Acquire semaphore
	r.mu.Lock()
	<-r.ticker.C
	r.mu.Unlock()
}

// Release releases the semaphore
func (r *RateLimiter) Release() {
	<-r.semaphore
}

// Global rate limiter for OpenAI API (3,500 RPM for ada-002 embeddings is the limit)
// Using 3,000 to be safe
var apiRateLimiter = NewRateLimiter(3000, 5)

// Minimum delay between API calls to avoid rate limiting
const MinDelayMS = 10

// Maximum token limit for OpenAI embeddings
const MaxTokenLimit = 8192

// Default timeout for API requests
const DefaultAPITimeout = 30 * time.Second

// ErrMissingAPIKey is returned when the API key is not set
var ErrMissingAPIKey = errors.New("OPENAI_API_KEY is not set in .env file")

// ErrEmbeddingFailed is returned when the embedding API call fails
var ErrEmbeddingFailed = errors.New("failed to generate embedding")

// CodeEmbedding represents a code embedding with metadata
type CodeEmbedding struct {
	Embedding []float32        `json:"embedding"`
	Metadata  CodeChunkMetadata `json:"metadata"`
}

// CodeChunkMetadata contains information about the code chunk
type CodeChunkMetadata struct {
	Filename  string `json:"filename"`
	Function  string `json:"function,omitempty"`
	Class     string `json:"class,omitempty"`
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`
	Content   string `json:"content"`
}

// nodeType defines types of syntax nodes we're interested in
type nodeType string

const (
	functionNode nodeType = "function"
	methodNode   nodeType = "method"
	classNode    nodeType = "class"
	structNode   nodeType = "struct"
	importNode   nodeType = "import"
)

// Language-specific Tree-sitter queries
var languageQueries = map[*sitter.Language][]string{
	golang.GetLanguage(): {
		// Functions
		"(function_declaration name: (identifier) @function_name) @function_def",
		// Methods
		"(method_declaration name: (field_identifier) @method_name) @method_def",
		// Structs
		"(type_declaration (type_spec name: (identifier) @struct_name type: (struct_type)) @struct_def)",
		// Imports
		"(import_declaration) @import",
	},
	python.GetLanguage(): {
		// Functions
		"(function_definition name: (identifier) @function_name) @function_def",
		// Classes
		"(class_definition name: (identifier) @class_name) @class_def",
		// Imports
		"(import_statement) @import",
		"(import_from_statement) @import",
	},
	javascript.GetLanguage(): {
		// Functions - including arrow functions
		"(function_declaration name: (identifier) @function_name) @function_def",
		"(arrow_function) @function_def",
		"(function) @function_def",
		// Classes
		"(class_declaration name: (identifier) @class_name) @class_def",
		// Methods
		"(method_definition name: (property_identifier) @method_name) @method_def",
		// Variable declarations with functions
		"(variable_declarator name: (identifier) @var_name value: [(function) (arrow_function)]) @function_def",
		// Imports
		"(import_statement) @import",
	},
}

// Cached parsers to avoid recreating them for each file
var parserCache = make(map[*sitter.Language]*sitter.Parser)
var parserMutex sync.Mutex

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

// GetCodeEmbeddings generates embeddings for code with semantic chunks
func GetCodeEmbeddings(filePath string, content string) ([]CodeEmbedding, error) {
	// Parse the code to extract semantic chunks using Tree-sitter
	chunks, err := extractSemanticChunksWithTreeSitter(filePath, content)
	if err != nil {
		return nil, fmt.Errorf("failed to extract semantic chunks: %w", err)
	}
	
	// Create embeddings for each chunk
	var embeddings []CodeEmbedding
	
	// Get content for each chunk
	var chunkTexts []string
	for _, chunk := range chunks {
		chunkTexts = append(chunkTexts, chunk.Content)
	}
	
	// Get embeddings in batch
	embeddingsMap, err := GetBatchEmbeddings(chunkTexts, 20)
	if err != nil {
		return nil, err
	}
	
	// Match embeddings with their metadata
	for i, chunk := range chunks {
		if embedding, ok := embeddingsMap[chunk.Content]; ok {
			embeddings = append(embeddings, CodeEmbedding{
				Embedding: embedding,
				Metadata:  chunk,
			})
		} else {
			log.Printf("Warning: Failed to get embedding for chunk %d in %s", i, filePath)
		}
	}
	
	return embeddings, nil
}

// batchResult is used to collect results from embedding API calls
type batchResult struct {
	Texts      []string
	StartIndex int
	Embeddings [][]float32
	Error      error
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
					Model: openai.AdaEmbeddingV2,
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

// extractSemanticChunksWithTreeSitter uses Tree-sitter to parse code and extract meaningful chunks
func extractSemanticChunksWithTreeSitter(filePath string, content string) ([]CodeChunkMetadata, error) {
	ext := strings.ToLower(filepath.Ext(filePath))
	filename := filepath.Base(filePath)
	
	var language *sitter.Language
	
	// Select the appropriate Tree-sitter language parser
	switch ext {
	case ".go":
		language = golang.GetLanguage()
	case ".py":
		language = python.GetLanguage()
	case ".js", ".ts", ".jsx", ".tsx":
		language = javascript.GetLanguage()
	default:
		// Fall back to generic chunking for unsupported languages
		return extractGenericChunks(filename, strings.Split(content, "\n"))
	}
	
	// Use or create a parser from cache with mutex protection
	parserMutex.Lock()
	var parser *sitter.Parser
	var ok bool
	
	if parser, ok = parserCache[language]; !ok {
		parser = sitter.NewParser()
		parser.SetLanguage(language)
		parserCache[language] = parser
	}
	parserMutex.Unlock()
	
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	tree, err := parser.ParseCtx(ctx, nil, []byte(content))
	if err != nil {
		return nil, fmt.Errorf("tree-sitter parsing failed: %w", err)
	}
	defer tree.Close()
	
	rootNode := tree.RootNode()
	
	// Extract chunks based on language-specific AST queries
	chunks, err := extractChunksFromAST(filename, content, rootNode, language)
	if err != nil {
		return nil, err
	}
	
	// If no chunks were found, fall back to generic chunking
	if len(chunks) == 0 {
		return extractGenericChunks(filename, strings.Split(content, "\n"))
	}
	
	return chunks, nil
}

// extractChunksFromAST extracts code chunks from the AST using language-specific queries
func extractChunksFromAST(filename, content string, rootNode *sitter.Node, language *sitter.Language) ([]CodeChunkMetadata, error) {
	var chunks []CodeChunkMetadata
	lines := strings.Split(content, "\n")
	
	// Get queries for this language
	queries, ok := languageQueries[language]
	if !ok {
		return nil, fmt.Errorf("no queries defined for language")
	}
	
	for _, queryStr := range queries {
		query, err := sitter.NewQuery([]byte(queryStr), language)
		if err != nil {
			log.Printf("Error creating query '%s': %v", queryStr, err)
			continue
		}
		
		cursor := sitter.NewQueryCursor()
		cursor.Exec(query, rootNode)
		
		for {
			match, ok := cursor.NextMatch()
			if !ok {
				break
			}
			
			for _, capture := range match.Captures {
				node := capture.Node
				
				// Get node type from the capture name
				captureName := query.CaptureNameForId(capture.Index)
				
				if strings.HasSuffix(captureName, "_def") {
					// This is a definition node (function, class, etc.)
					nodeStart := node.StartPoint()
					nodeEnd := node.EndPoint()
					
					var chunk CodeChunkMetadata
					chunk.Filename = filename
					chunk.StartLine = int(nodeStart.Row) + 1 // Convert to 1-indexed
					chunk.EndLine = int(nodeEnd.Row) + 1     // Convert to 1-indexed
					
					// Get the actual code content - fix index calculation
					nodeContent := getNodeContent(lines, nodeStart.Row, nodeEnd.Row)
					chunk.Content = nodeContent
					
					// Find the name capture if present
					for _, nameCapture := range match.Captures {
						nameCaptureType := query.CaptureNameForId(nameCapture.Index)
						if strings.HasSuffix(nameCaptureType, "_name") {
							name := content[nameCapture.Node.StartByte():nameCapture.Node.EndByte()]
							
							if strings.Contains(captureName, "function") || strings.Contains(captureName, "method") {
								chunk.Function = name
							} else if strings.Contains(captureName, "class") || strings.Contains(captureName, "struct") {
								chunk.Class = name
							}
						}
					}
					
					// Only add if there's actual content
					if len(strings.TrimSpace(chunk.Content)) > 0 {
						chunks = append(chunks, chunk)
					}
				}
			}
		}
	}
	
	return chunks, nil
}

// getNodeContent extracts text content from source lines for a node
func getNodeContent(lines []string, startRow, endRow uint32) string {
	// Fix for zero-based indexing
	if int(startRow) >= len(lines) {
		return ""
	}
	
	endIdx := int(endRow)
	if endIdx >= len(lines) {
		endIdx = len(lines) - 1
	}
	
	// Handle single-line nodes correctly
	if startRow == endRow {
		if int(startRow) < len(lines) {
			return lines[startRow]
		}
		return ""
	}
	
	return strings.Join(lines[startRow:endIdx+1], "\n")
}

// extractGenericChunks provides fallback generic chunking for unsupported languages
func extractGenericChunks(filename string, lines []string) ([]CodeChunkMetadata, error) {
	var chunks []CodeChunkMetadata
	
	// For unsupported languages, create larger chunks based on empty lines
	// as separators, simulating paragraph breaks
	
	var chunkStart int
	var currentChunk []string
	
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		
		if trimmed == "" && len(currentChunk) > 0 {
			// End of a paragraph-like chunk
			chunks = append(chunks, CodeChunkMetadata{
				Filename:  filename,
				StartLine: chunkStart + 1, // Convert to 1-indexed
				EndLine:   i,              // Convert to 1-indexed
				Content:   strings.Join(currentChunk, "\n"),
			})
			currentChunk = nil
		} else if trimmed != "" {
			if len(currentChunk) == 0 {
				chunkStart = i
			}
			currentChunk = append(currentChunk, line)
		}
	}
	
	// Add the final chunk if any
	if len(currentChunk) > 0 {
		chunks = append(chunks, CodeChunkMetadata{
			Filename:  filename,
			StartLine: chunkStart + 1,     // Convert to 1-indexed
			EndLine:   len(lines),         // Convert to 1-indexed
			Content:   strings.Join(currentChunk, "\n"),
		})
	}
	
	return chunks, nil
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