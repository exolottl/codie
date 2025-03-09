package fileutils

import (
	"os"
	"path/filepath"
	"strings"
)

// Common code file extensions to process
var codeExtensions = map[string]bool{
	".py":   true,
	".js":   true,
	".ts":   true,
	".cpp":  true,
	".go":   true,
	".java": true,
	".lua":  true,
	".jsx":  true,
	".tsx":  true,
	".html": true,
	".css":  true,
	".php":  true,
	".rb":   true,
	".rs":   true,
	".cs":   true,
	".swift": true,
	".kt":   true,
}

// Common directories to skip
var skipDirs = map[string]bool{
	".git":       true,
	"node_modules": true,
	"venv":       true,
	"__pycache__": true,
	"dist":       true,
	"build":      true,
	".idea":      true,
	".vscode":    true,
}

// GetCodeFiles returns a list of code files in the given directory
func GetCodeFiles(root string) ([]string, error) {
	var files []string
	
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		
		// Skip directories we want to exclude
		if info.IsDir() {
			if skipDirs[info.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		
		// Check if file has code extension
		ext := filepath.Ext(info.Name())
		if codeExtensions[ext] {
			files = append(files, path)
		}
		
		return nil
	})
	
	return files, err
}

// SplitCodeIntoChunks splits a code string into chunks with improved logic
func SplitCodeIntoChunks(code string, maxChunkSize int) []string {
	if maxChunkSize <= 0 {
		maxChunkSize = 1000 // Default max chunk size
	}
	
	// Split by natural code separators
	rawChunks := strings.Split(code, "\n\n")
	
	var chunks []string
	var currentChunk strings.Builder
	
	for _, chunk := range rawChunks {
		// Skip empty chunks
		trimmedChunk := strings.TrimSpace(chunk)
		if trimmedChunk == "" {
			continue
		}
		
		// If adding this chunk would exceed max size, finalize current chunk and start a new one
		if currentChunk.Len() > 0 && currentChunk.Len()+len(trimmedChunk) > maxChunkSize {
			chunks = append(chunks, currentChunk.String())
			currentChunk.Reset()
		}
		
		// Add the current chunk
		if currentChunk.Len() > 0 {
			currentChunk.WriteString("\n\n")
		}
		currentChunk.WriteString(trimmedChunk)
		
		// If the chunk itself is already bigger than max size, add it directly
		if currentChunk.Len() >= maxChunkSize {
			chunks = append(chunks, currentChunk.String())
			currentChunk.Reset()
		}
	}
	
	// Add any remaining content
	if currentChunk.Len() > 0 {
		chunks = append(chunks, currentChunk.String())
	}
	
	return chunks
}

// ReadFileContent reads a file and returns its content as a string
func ReadFileContent(filePath string) (string, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}
	return string(content), nil
}