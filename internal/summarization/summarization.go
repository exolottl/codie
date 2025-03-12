package summarization

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
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
	DetailLevel    string // "brief", "standard", or "comprehensive"
	FocusPath      string // Optional subdirectory to focus on
	IncludeMetrics bool   // Include code metrics in summary
}

// DefaultSummaryOptions returns the default options for summarization
func DefaultSummaryOptions() SummaryOptions {
	return SummaryOptions{
		DetailLevel:    "standard",
		FocusPath:      "",
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

	// Analyze dependencies
	dependencies := extractDependencies(fileChunks)

	// Build the prompt for OpenAI
	prompt := buildSummaryPrompt(repoStructure, fileChunks, fileImportance, dependencies, options)

	// Get summary from OpenAI
	summary, err := getAISummary(prompt, options)
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
		".yml":   "YAML",
		".yaml":  "YAML",
		".json":  "JSON",
		".md":    "Markdown",
		".sql":   "SQL",
		".sh":    "Shell",
		".bat":   "Batch",
		".ps1":   "PowerShell",
	}

	if lang, ok := languages[ext]; ok {
		return lang
	}
	return "Unknown"
}

// getMainLanguages returns a comma-separated list of the most common languages in the repo
func getMainLanguages(repoStructure []FileStructure) string {
	langCount := make(map[string]int)
	
	for _, file := range repoStructure {
		if file.Language != "Unknown" {
			langCount[file.Language] += file.LOC
		}
	}
	
	type langStats struct {
		name string
		loc  int
	}
	
	var stats []langStats
	for lang, loc := range langCount {
		stats = append(stats, langStats{lang, loc})
	}
	
	// Sort by LOC descending
	sort.Slice(stats, func(i, j int) bool {
		return stats[i].loc > stats[j].loc
	})
	
	// Take top 3 languages
	var mainLangs []string
	for i := 0; i < len(stats) && i < 3; i++ {
		mainLangs = append(mainLangs, stats[i].name)
	}
	
	return strings.Join(mainLangs, ", ")
}

// calculateTotalLOC returns the total lines of code in the repository
func calculateTotalLOC(repoStructure []FileStructure) int {
	total := 0
	for _, file := range repoStructure {
		total += file.LOC
	}
	return total
}

// extractDependencies analyzes project files to identify dependencies
func extractDependencies(fileChunks map[string][]string) string {
	var sb strings.Builder
	
	// Check for Go modules
	if content, exists := fileChunks["go.mod"]; exists {
		sb.WriteString("Go Dependencies:\n")
		for _, chunk := range content {
			// Extract require statements
			lines := strings.Split(chunk, "\n")
			for _, line := range lines {
				if strings.HasPrefix(strings.TrimSpace(line), "require ") || 
				   strings.HasPrefix(strings.TrimSpace(line), "require(") ||
				   (strings.TrimSpace(line) != "" && !strings.HasPrefix(strings.TrimSpace(line), "module ") && !strings.HasPrefix(strings.TrimSpace(line), "go ")) {
					sb.WriteString("- " + strings.TrimSpace(line) + "\n")
				}
			}
		}
		sb.WriteString("\n")
	}
	
	// Check for package.json (Node.js)
	if content, exists := fileChunks["package.json"]; exists {
		sb.WriteString("Node.js Dependencies:\n")
		var packageJson string
		for _, chunk := range content {
			packageJson += chunk
		}
		
		// Simple regex to extract dependencies
		depsRegex := regexp.MustCompile(`"dependencies"\s*:\s*{([^}]*)}`)
		devDepsRegex := regexp.MustCompile(`"devDependencies"\s*:\s*{([^}]*)}`)
		
		if matches := depsRegex.FindStringSubmatch(packageJson); len(matches) > 1 {
			deps := matches[1]
			deps = strings.ReplaceAll(deps, "\n", "")
			deps = strings.ReplaceAll(deps, "\"", "")
			deps = strings.ReplaceAll(deps, " ", "")
			for _, dep := range strings.Split(deps, ",") {
				if dep = strings.TrimSpace(dep); dep != "" {
					sb.WriteString("- " + dep + "\n")
				}
			}
		}
		
		if matches := devDepsRegex.FindStringSubmatch(packageJson); len(matches) > 1 {
			sb.WriteString("Dev Dependencies:\n")
			deps := matches[1]
			deps = strings.ReplaceAll(deps, "\n", "")
			deps = strings.ReplaceAll(deps, "\"", "")
			deps = strings.ReplaceAll(deps, " ", "")
			for _, dep := range strings.Split(deps, ",") {
				if dep = strings.TrimSpace(dep); dep != "" {
					sb.WriteString("- " + dep + "\n")
				}
			}
		}
		sb.WriteString("\n")
	}
	
	// Check for requirements.txt (Python)
	if content, exists := fileChunks["requirements.txt"]; exists {
		sb.WriteString("Python Dependencies:\n")
		for _, chunk := range content {
			lines := strings.Split(chunk, "\n")
			for _, line := range lines {
				if line = strings.TrimSpace(line); line != "" && !strings.HasPrefix(line, "#") {
					sb.WriteString("- " + line + "\n")
				}
			}
		}
		sb.WriteString("\n")
	}
	
	if sb.Len() == 0 {
		return "No standard dependency files detected."
	}
	return sb.String()
}

// calculateFileImportance determines which files are most important in the codebase
func calculateFileImportance(repoStructure []FileStructure, fileChunks map[string][]string) map[string]float64 {
	importance := make(map[string]float64)
	
	// Map to track imports between files
	importMap := make(map[string]int)
	importedBy := make(map[string]int)
	
	// Scan for imports and key patterns
	for filePath, chunks := range fileChunks {
		// Join chunks for analysis
		content := strings.Join(chunks, "\n")
		
		// Count imports in this file
		importCount := 0
		
		// Check for imports based on language patterns
		if strings.HasSuffix(filePath, ".go") {
			importCount += countMatches(content, `import\s+\(([^)]*)\)`) // Go multi imports
			importCount += countMatches(content, `import\s+"[^"]+"`) // Go single imports
		} else if strings.HasSuffix(filePath, ".js") || strings.HasSuffix(filePath, ".ts") {
			importCount += countMatches(content, `import\s+.*\s+from\s+['"]`) // JS/TS imports
			importCount += countMatches(content, `require\(['"]`) // JS/TS requires
		} else if strings.HasSuffix(filePath, ".py") {
			importCount += countMatches(content, `import\s+[a-zA-Z0-9_]+`) // Python imports
			importCount += countMatches(content, `from\s+[a-zA-Z0-9_]+\s+import`) // Python from imports
		} else if strings.HasSuffix(filePath, ".java") {
			importCount += countMatches(content, `import\s+[a-zA-Z0-9_.]+;`) // Java imports
		}
		
		importMap[filePath] = importCount
		
		// Check for patterns suggesting importance
		patternScore := 0.0
		
		// Check for interfaces/abstractions
		if strings.HasSuffix(filePath, ".go") {
			patternScore += float64(countMatches(content, `type\s+[A-Z][a-zA-Z0-9_]*\s+interface`)) * 2
			patternScore += float64(countMatches(content, `func\s+main\(`)) * 5 // Main function
		} else if strings.HasSuffix(filePath, ".java") {
			patternScore += float64(countMatches(content, `interface\s+[A-Z][a-zA-Z0-9_]*`)) * 2
			patternScore += float64(countMatches(content, `abstract\s+class`)) * 2.5
			patternScore += float64(countMatches(content, `public\s+static\s+void\s+main`)) * 5 // Main method
		} else if strings.HasSuffix(filePath, ".ts") || strings.HasSuffix(filePath, ".js") {
			patternScore += float64(countMatches(content, `interface\s+[A-Z][a-zA-Z0-9_]*`)) * 2
			patternScore += float64(countMatches(content, `class\s+[A-Z][a-zA-Z0-9_]*`)) * 1.5
			patternScore += float64(countMatches(content, `export\s+default`)) * 1.2
		} else if strings.HasSuffix(filePath, ".py") {
			patternScore += float64(countMatches(content, `class\s+[A-Z][a-zA-Z0-9_]*`)) * 1.5
			patternScore += float64(countMatches(content, `def\s+__init__`)) * 0.5
			patternScore += float64(countMatches(content, `if\s+__name__\s*==\s*["']__main__["']`)) * 5 // Main block
		}
		
		// Cross-reference imports to determine imported-by count
		for otherFilePath, otherChunks := range fileChunks {
			if otherFilePath == filePath {
				continue
			}
			
			otherContent := strings.Join(otherChunks, "\n")
			
			// Extract filename without extension
			baseNameWithExt := filepath.Base(filePath)
			baseName := strings.TrimSuffix(baseNameWithExt, filepath.Ext(baseNameWithExt))
			
			// Count references to this file in other files
			if strings.Contains(otherContent, baseName) {
				importedBy[filePath]++
			}
		}
		
		// Calculate file path depth score
		pathSegments := strings.Split(filePath, string(os.PathSeparator))
		depth := len(pathSegments)
		
		// Files in important directories get a boost
		pathScore := 0.0
		lowerPath := strings.ToLower(filePath)
		if strings.Contains(lowerPath, "main") || strings.Contains(lowerPath, "cmd") {
			pathScore += 2.0
		}
		if strings.Contains(lowerPath, "api") || strings.Contains(lowerPath, "internal") {
			pathScore += 1.5
		}
		if strings.Contains(lowerPath, "core") || strings.Contains(lowerPath, "model") || 
		   strings.Contains(lowerPath, "service") || strings.Contains(lowerPath, "controller") {
			pathScore += 1.8
		}
		if strings.Contains(lowerPath, "util") || strings.Contains(lowerPath, "helper") {
			pathScore += 0.7
		}
		
		// File size factor (normalize LOC)
		var fileLOC int
		for _, fs := range repoStructure {
			if fs.Path == filePath {
				fileLOC = fs.LOC
				break
			}
		}
		locFactor := float64(fileLOC) / 1000
		if locFactor > 1 {
			locFactor = 1
		}
		
		// Calculate final importance score
		importance[filePath] = (
			locFactor * 0.2 +                        // Size of file
			(1.0 / float64(depth)) * 0.15 +          // Depth in directory tree
			pathScore * 0.2 +                        // Important directory names
			float64(importMap[filePath]) * 0.15 +    // Number of imports (complexity)
			float64(importedBy[filePath]) * 0.2 +    // How many files import this one (centrality)
			patternScore * 0.1) * 10                       // Important code patterns
	}
	
	return importance
}

// countMatches counts the number of matches for a regex pattern in text
func countMatches(text, pattern string) int {
	re := regexp.MustCompile(pattern)
	return len(re.FindAllString(text, -1))
}

// buildSummaryPrompt creates the prompt for the OpenAI API
func buildSummaryPrompt(repoStructure []FileStructure, fileChunks map[string][]string, 
	fileImportance map[string]float64, dependencies string, options SummaryOptions) string {
	var sb strings.Builder
	
	// Enhanced instruction with professional guidance
	sb.WriteString("You are analyzing a software codebase. Your task is to create a professional, ")
	sb.WriteString("technically precise summary that would help a developer understand this project quickly. ")
	sb.WriteString("Focus on identifying architectural patterns, key abstractions, and the overall design philosophy. ")
	sb.WriteString("When code follows well-known patterns or frameworks, explicitly name them. ")
	
	if options.DetailLevel == "comprehensive" {
		sb.WriteString("Provide detailed explanations of key functionality, design patterns, and implementation decisions. ")
		sb.WriteString("Include technical nuances and considerations for future development.")
	} else if options.DetailLevel == "brief" {
		sb.WriteString("Keep the summary concise and focused on the most essential components. ")
		sb.WriteString("Prioritize clarity and high-level understanding over implementation details.")
	} else {
		sb.WriteString("Balance high-level architectural insights with important implementation details. ")
		sb.WriteString("Include enough context for developers to understand the project's approach.")
	}
	
	// Add structured context about the codebase
	sb.WriteString("\n\nCodebase Context:\n")
	sb.WriteString("- Primary Languages: " + getMainLanguages(repoStructure) + "\n")
	sb.WriteString("- Total Files: " + fmt.Sprintf("%d", len(repoStructure)) + "\n") 
	sb.WriteString("- Total Lines of Code: " + fmt.Sprintf("%d", calculateTotalLOC(repoStructure)) + "\n")
	
	// Add chain-of-thought prompting
	sb.WriteString("\n\nAnalysis approach:\n")
	sb.WriteString("1. First, examine the project structure to identify the architecture pattern\n")
	sb.WriteString("2. Then, analyze key files to understand core functionality\n")
	sb.WriteString("3. Next, identify relationships between components\n")
	sb.WriteString("4. Finally, synthesize findings into a cohesive summary\n")
	
	// File structure section
	sb.WriteString("\n\nCodebase structure:\n")
	
	// Group files by directory for better organization
	dirMap := make(map[string][]FileStructure)
	for _, file := range repoStructure {
		dir := filepath.Dir(file.Path)
		dirMap[dir] = append(dirMap[dir], file)
	}
	
	// Print directories and their files
	var dirs []string
	for dir := range dirMap {
		dirs = append(dirs, dir)
	}
	sort.Strings(dirs)
	
	for _, dir := range dirs {
		if dir == "." {
			sb.WriteString("Root directory:\n")
		} else {
			sb.WriteString(fmt.Sprintf("Directory %s:\n", dir))
		}
		
		for _, file := range dirMap[dir] {
			sb.WriteString(fmt.Sprintf("  - %s (%s, %d lines)\n", 
				filepath.Base(file.Path), file.Language, file.LOC))
		}
	}
	
	// Add dependency information
	sb.WriteString("\n\nProject Dependencies:\n")
	sb.WriteString(dependencies)
	
	// Include most important files content
	sb.WriteString("\n\nKey files content:\n")
	
	// Find top important files
	type fileScore struct {
		path  string
		score float64
	}
	var scores []fileScore
	for path, score := range fileImportance {
		scores = append(scores, fileScore{path, score})
	}
	
	// Sort by importance (higher score first)
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})
	
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
		
		sb.WriteString(fmt.Sprintf("\n--- %s (Importance: %.2f) ---\n", filePath, scores[i].score))
		
		// Join chunks for this file
		content := strings.Join(fileChunks[filePath], "\n...\n")
		
		// If file is too large, include just beginning and end
		if len(content) > 4000 && options.DetailLevel != "comprehensive" {
			contentLines := strings.Split(content, "\n")
			if len(contentLines) > 100 {
				beginLines := contentLines[:50]
				endLines := contentLines[len(contentLines)-50:]
				content = strings.Join(beginLines, "\n") + "\n...[middle section omitted]...\n" + strings.Join(endLines, "\n")
			}
		}
		
		sb.WriteString(content)
		sb.WriteString("\n")
	}
	
	// Example of good summary style for guidance
	if options.DetailLevel != "brief" {
		sb.WriteString("\n\nExample of good summary style:\n")
		sb.WriteString("\"This project implements a REST API service using a hexagonal architecture. ")
		sb.WriteString("The core domain logic is isolated in the 'domain' package, with separate ")
		sb.WriteString("adapter layers for HTTP routing (using Echo framework), persistence (PostgreSQL), ")
		sb.WriteString("and external integrations. The codebase follows dependency injection principles ")
		sb.WriteString("with interfaces defined at domain boundaries...\"\n")
	}
	
	// Instructions for output format with self-critique
	sb.WriteString("\n\nPlease format the summary with the following sections:\n")
	sb.WriteString("1. Overview - What the project does and its main purpose\n")
	sb.WriteString("2. Architecture - Main components and how they're organized\n")
	sb.WriteString("3. Key Features - Important functionality implemented\n")
	sb.WriteString("4. Implementation Details - Notable code patterns or techniques\n")
	
	if options.IncludeMetrics {
		sb.WriteString("5. Code Quality - Assessment of structure, organization, and maintainability\n")
	}
	
	// Request self-critique
	sb.WriteString("\nAfter drafting your summary, please review it against these quality criteria:\n")
	sb.WriteString("- Technical accuracy: Are architectural terms used correctly?\n")
	sb.WriteString("- Comprehensiveness: Does it cover all major aspects of the codebase?\n")
	sb.WriteString("- Clarity: Would a developer understand the project from this description?\n")
	sb.WriteString("- Insight: Does it provide useful insights beyond what's immediately obvious?\n")
	
	return sb.String()
}

// getAISummary sends the prompt to OpenAI and gets the summary
func getAISummary(prompt string, options SummaryOptions) (string, error) {
	// Get API key from environment
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return "", fmt.Errorf("OPENAI_API_KEY is not set")
	}

	// Create client
	client := openai.NewClient(apiKey)

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	// Adjust temperature based on detail level
	temperature := 0.2 // Default for standard
	if options.DetailLevel == "comprehensive" {
		temperature = 0.3 // Slightly more creative for detailed analysis
	} else if options.DetailLevel == "brief" {
		temperature = 0.1 // More focused for brief summaries
	}

	// Make API request with enhanced parameters
	resp, err := client.CreateChatCompletion(
		ctx,
		openai.ChatCompletionRequest{
			Model: openai.O3Mini,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleSystem,
					Content: "You are a senior software engineer specialized in analyzing and summarizing codebases. Your summaries are technically precise, insightful, and focused on helping developers understand architectural patterns and design decisions.",
				},
				{
					Role:    openai.ChatMessageRoleUser,
					Content: prompt,
				},
			},
			MaxTokens:   4000,
			Temperature: float32(temperature),
			TopP:        0.95,
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