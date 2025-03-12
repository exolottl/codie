package embeddings

import(
	"errors"
	"time"
)

// CodeEmbedding represents a code embedding with metadata
type CodeEmbedding struct {
	Embedding []float32         `json:"embedding"`
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

// Common errors
var (
	ErrMissingAPIKey    = errors.New("OPENAI_API_KEY is not set in .env file")
	ErrEmbeddingFailed  = errors.New("failed to generate embedding")
)

// Constants
const (
	MaxTokenLimit     = 8192
	DefaultAPITimeout = 30 * time.Second
	MinDelayMS        = 10
)