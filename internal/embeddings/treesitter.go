package embeddings

import (
	"context"
	"fmt"
	"log"
	"path/filepath"
	"strings"
	"sync"
	"time"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/golang"
	"github.com/smacker/go-tree-sitter/javascript"
	"github.com/smacker/go-tree-sitter/python"
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