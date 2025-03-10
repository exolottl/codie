package embeddings

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"
	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/golang"
	"github.com/smacker/go-tree-sitter/python"
	"github.com/smacker/go-tree-sitter/javascript"
)

// Minimum delay between API calls to avoid rate limiting
const MinDelayMS = 15

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

// extractSemanticChunksWithTreeSitter uses Tree-sitter to parse code and extract meaningful chunks
func extractSemanticChunksWithTreeSitter(filePath string, content string) ([]CodeChunkMetadata, error) {
	ext := strings.ToLower(filepath.Ext(filePath))
	filename := filepath.Base(filePath)
	
	var parser *sitter.Parser
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
	
	parser = sitter.NewParser()
	parser.SetLanguage(language)
	
	tree, err := parser.ParseCtx(context.Background(), nil, []byte(content))
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
	
	// Create queries appropriate for the language
	var queries []string
	
	if language == golang.GetLanguage() {
		queries = []string{
			// Functions
			"(function_declaration name: (identifier) @function_name) @function_def",
			// Methods
			"(method_declaration name: (field_identifier) @method_name) @method_def",
			// Structs
			"(type_declaration (type_spec name: (identifier) @struct_name type: (struct_type)) @struct_def)",
			// Imports
			"(import_declaration) @import",
		}
	} else if language == python.GetLanguage() {
		queries = []string{
			// Functions
			"(function_definition name: (identifier) @function_name) @function_def",
			// Classes
			"(class_definition name: (identifier) @class_name) @class_def",
			// Imports
			"(import_statement) @import",
			"(import_from_statement) @import",
		}
	} else if language == javascript.GetLanguage() {
		queries = []string{
			// Functions
			"(function_declaration name: (identifier) @function_name) @function_def",
			// Classes
			"(class_declaration name: (identifier) @class_name) @class_def",
			// Methods
			"(method_definition name: (property_identifier) @method_name) @method_def",
			// Imports
			"(import_statement) @import",
		}
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
					chunk.StartLine = int(nodeStart.Row)
					chunk.EndLine = int(nodeEnd.Row)
					
					// Get the actual code content
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
					
					chunks = append(chunks, chunk)
				}
			}
		}
	}
	
	return chunks, nil
}

// getNodeContent extracts text content from source lines for a node
func getNodeContent(lines []string, startRow, endRow uint32) string {
	if int(startRow) >= len(lines) {
		return ""
	}
	
	endIdx := int(endRow)
	if endIdx >= len(lines) {
		endIdx = len(lines) - 1
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
				StartLine: chunkStart,
				EndLine:   i - 1,
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
			StartLine: chunkStart,
			EndLine:   len(lines) - 1,
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