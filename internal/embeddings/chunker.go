package embeddings

import (
	"fmt"
	"log"
	"strings"
)

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