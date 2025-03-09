package storage

import (
	"encoding/json"
	"os"
)

// CodeChunk represents a chunk of code with its embedding
type CodeChunk struct {
	File      string    `json:"file"`
	Content   string    `json:"content"`
	Embedding []float32 `json:"embedding"`
}

// SaveToJSON saves a slice of CodeChunks to a JSON file
func SaveToJSON(chunks []CodeChunk, filename string) error {
	output, err := json.MarshalIndent(chunks, "", "  ")
	if err != nil {
		return err
	}
	
	return os.WriteFile(filename, output, 0644)
}