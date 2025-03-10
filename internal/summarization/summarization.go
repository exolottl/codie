package summarization

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"
	"codie/internal/storage"
)

// FileStructure represents the structure of a file in the codebase
type FileStructure struct {
	Path     string `json:"path"`
	Language string `json:"language"`
	LOC      int    `json:"loc"`
}

// SummaryOptions configures the behavior of the summarization process
type SummaryOptions struct {
	DetailLevel   string // "brief", "standard", or "comprehensive"
	FocusPath     string // Optional subdirectory to focus on
	IncludeMetrics bool   // Include code metrics in summary
}

// DefaultSummaryOptions returns the default options for summarization
func DefaultSummaryOptions() SummaryOptions {
	return SummaryOptions{
		DetailLevel:   "standard",
		FocusPath:     "",
		IncludeMetrics: true,
	}
}

// GenerateRepoSummary creates a summary of the codebase using OpenAI
func GenerateRepoSummary(embeddingsPath string, options SummaryOptions) (string, error) {
	// Load embeddings from file
	chunks, err := loadCodeChunks(embeddingsPath)
	if err != nil {
		return "", fmt.Errorf("failed to load embeddings: %v", err)
	}

	// Create a map of files and their code chunks
	fileChunks := organizeChunksByFile(chunks)

	// Get high-level file structure
	repoStructure := analyzeRepoStructure(fileChunks)

	// Generate file importance/relevance metrics
	fileImportance := calculateFileImportance(repoStructure, fileChunks)

	// Build the prompt for OpenAI
	prompt := buildSummaryPrompt(repoStructure, fileChunks, fileImportance, options)

	// Get summary from OpenAI
	summary, err := getAISummary(prompt)
	if err != nil {
		return "", fmt.Errorf("failed to generate summary: %v", err)
	}

	return summary, nil
}

// loadCodeChunks loads the code chunks from the embeddings file
func loadCodeChunks(embeddingsPath string) ([]storage.CodeChunk, error) {
	data, err := os.ReadFile(embeddingsPath)
	if err != nil {
		return nil, err
	}

	var chunks []storage.CodeChunk
	err = json.Unmarshal(data, &chunks)
	if err != nil {
		return nil, err
	}

	return chunks, nil
}

// organizeChunksByFile groups code chunks by their source file
func organizeChunksByFile(chunks []storage.CodeChunk) map[string][]string {
	fileChunks := make(map[string][]string)

	for _, chunk := range chunks {
		fileChunks[chunk.File] = append(fileChunks[chunk.File], chunk.Content)
	}

	return fileChunks
}

// analyzeRepoStructure extracts the structure of the repository
func analyzeRepoStructure(fileChunks map[string][]string) []FileStructure {
	var structure []FileStructure

	for filePath, chunks := range fileChunks {
		// Calculate total lines of code in file
		loc := 0
		for _, chunk := range chunks {
			loc += len(strings.Split(chunk, "\n"))
		}

		// Determine language from file extension
		ext := filepath.Ext(filePath)
		language := getLanguageFromExtension(ext)

		structure = append(structure, FileStructure{
			Path:     filePath,
			Language: language,
			LOC:      loc,
		})
	}

	return structure
}

// getLanguageFromExtension maps file extensions to programming languages
func getLanguageFromExtension(ext string) string {
	languages := map[string]string{
		".py":    "Python",
		".js":    "JavaScript",
		".ts":    "TypeScript",
		".go":    "Go",
		".java":  "Java",
		".cpp":   "C++",
		".c":     "C",
		".rb":    "Ruby",
		".php":   "PHP",
		".html":  "HTML",
		".css":   "CSS",
		".rs":    "Rust",
		".swift": "Swift",
		".kt":    "Kotlin",
		".cs":    "C#",
		".jsx":   "React JSX",
		".tsx":   "React TSX",
		".lua":   "Lua",
	}

	if lang, ok := languages[ext]; ok {
		return lang
	}
	return "Unknown"
}

// calculateFileImportance determines which files are most important in the codebase
func calculateFileImportance(repoStructure []FileStructure, fileChunks map[string][]string) map[string]float64 {
	importance := make(map[string]float64)

	// Simple importance metric based on:
	// 1. Files in the root directory
	// 2. Files with more code
	// 3. Files that appear to be entry points (main, index)
	for _, file := range repoStructure {
		// Base importance on lines of code (normalized)
		locFactor := float64(file.LOC) / 1000
		if locFactor > 1 {
			locFactor = 1
		}

		// Higher importance for root files
		dirDepth := float64(len(strings.Split(file.Path, string(os.PathSeparator))))
		depthFactor := 1.0 / dirDepth

		// Higher importance for main/entry files
		fileName := filepath.Base(file.Path)
		entryFactor := 1.0
		if strings.HasPrefix(fileName, "main") || strings.HasPrefix(fileName, "index") {
			entryFactor = 2.0
		}

		// Calculate final importance score
		importance[file.Path] = (locFactor*0.3 + depthFactor*0.3 + entryFactor*0.4) * 10
	}

	return importance
}

// buildSummaryPrompt creates the prompt for the OpenAI API
func buildSummaryPrompt(repoStructure []FileStructure, fileChunks map[string][]string, fileImportance map[string]float64, options SummaryOptions) string {
	var sb strings.Builder

	// Basic instruction
	sb.WriteString("Please provide a basic and short yet whole summary of this codebase. ")
	sb.WriteString("Describe the overall architecture, main components, and how they interact. ")
	
	if options.DetailLevel == "comprehensive" {
		sb.WriteString("Include detailed explanations of key functionality and design patterns. ")
	} else if options.DetailLevel == "brief" {
		sb.WriteString("Keep the summary concise and high-level. ")
	}

	// File structure section
	sb.WriteString("\n\nCodebase structure:\n")
	for _, file := range repoStructure {
		sb.WriteString(fmt.Sprintf("- %s (%s, %d lines)\n", file.Path, file.Language, file.LOC))
	}

	// Include most important files content
	sb.WriteString("\n\nKey files content:\n")
	
	// Find top 5-10 important files
	type fileScore struct {
		path  string
		score float64
	}
	var scores []fileScore
	for path, score := range fileImportance {
		scores = append(scores, fileScore{path, score})
	}
	
	// Sort by importance (higher score first)
	for i := 0; i < len(scores); i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[i].score < scores[j].score {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}
	
	// Include top files based on detail level
	topFilesCount := 5
	if options.DetailLevel == "comprehensive" {
		topFilesCount = 10
	} else if options.DetailLevel == "brief" {
		topFilesCount = 3
	}
	
	// Add content of important files
	for i := 0; i < len(scores) && i < topFilesCount; i++ {
		filePath := scores[i].path
		
		// Focus check - if focus path is set, only include files in that path
		if options.FocusPath != "" && !strings.HasPrefix(filePath, options.FocusPath) {
			continue
		}
		
		sb.WriteString(fmt.Sprintf("\n--- %s ---\n", filePath))
		
		// Join chunks for this file
		content := strings.Join(fileChunks[filePath], "\n...\n")
		
		// If file is too large, include just beginning and end
		if len(content) > 4000 && options.DetailLevel != "comprehensive" {
			contentLines := strings.Split(content, "\n")
			if len(contentLines) > 100 {
				beginLines := contentLines[:50]
				endLines := contentLines[len(contentLines)-50:]
				content = strings.Join(beginLines, "\n") + "\n...\n" + strings.Join(endLines, "\n")
			}
		}
		
		sb.WriteString(content)
		sb.WriteString("\n")
	}

	// Instructions for output format
	sb.WriteString("\n\nPlease format the summary with the following sections:\n")
	sb.WriteString("1. Overview - What the project does and its main purpose\n")
	sb.WriteString("2. Architecture - Main components and how they're organized\n")
	sb.WriteString("3. Key Features - Important functionality implemented\n")
	sb.WriteString("4. Implementation Details - Notable code patterns or techniques\n")
	
	if options.IncludeMetrics {
		sb.WriteString("5. Code Quality - Assessment of structure, organization, and maintainability\n")
	}
	
	return sb.String()
}

// getAISummary sends the prompt to OpenAI and gets the summary
func getAISummary(prompt string) (string, error) {
	// Get API key from environment
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return "", fmt.Errorf("OPENAI_API_KEY is not set")
	}

	// Create client
	client := openai.NewClient(apiKey)

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	// Make API request
	resp, err := client.CreateChatCompletion(
		ctx,
		openai.ChatCompletionRequest{
			Model: openai.GPT4o,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleSystem,
					Content: "You are a helpful assistant specialized in analyzing and summarizing codebases.",
				},
				{
					Role:    openai.ChatMessageRoleUser,
					Content: prompt,
				},
			},
			MaxTokens: 4000,
		},
	)

	if err != nil {
		return "", err
	}

	if len(resp.Choices) == 0 || resp.Choices[0].Message.Content == "" {
		return "", fmt.Errorf("empty response from OpenAI")
	}

	return resp.Choices[0].Message.Content, nil
}