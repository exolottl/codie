package fileutils

import (
	"os"
	"path/filepath"
	"strings"
)

// GetCodeFiles returns a list of code files in the given directory
func GetCodeFiles(root string) []string {
	var files []string
	extensions := []string{".py", ".js", ".ts", ".cpp", ".go", ".java", ".lua"}
	
	filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() {
			for _, ext := range extensions {
				if strings.HasSuffix(info.Name(), ext) {
					files = append(files, path)
					break
				}
			}
		}
		return nil
	})
	
	return files
}

// SplitCodeIntoChunks splits a code string into chunks
func SplitCodeIntoChunks(code string) []string {
	return strings.Split(code, "\n\n")
}