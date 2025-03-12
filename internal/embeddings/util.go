package embeddings

import (
	"strings"
)

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