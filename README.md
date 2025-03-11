# Codie - Terminal-Based AI Code Assistant

## 🚀 Built for Devs Doing Real Work

Codie is a powerful CLI tool designed to help developers quickly understand and navigate codebases. Whether you're onboarding a new project or exploring an unfamiliar repository, Codie provides AI-powered insights to accelerate your workflow.

## 🛠 Requirements

- Go 1.18 or higher
- OpenAI API key (you'll be prompted to provide this on first run, or you can set it in a `.env` file)

## 📥 Installation

```sh
# Clone the repository 
git clone https://github.com/yourusername/codie
cd codie

# Install dependencies
go mod tidy
```

## 🔑 Environment Setup

Create a `.env` file in the project root with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## 🚀 Usage

### Indexing a Codebase

Before generating a summary, you need to index your codebase:

```sh
go run main.go index <directory path>
```

This scans your codebase, processes code files, and generates embeddings using OpenAI's API. The embeddings are saved to `embeddings.json` for future use.

### Generating a Summary

After indexing, you can generate a summary of the codebase:

```sh
go run main.go summarize <directory path> [options]
```

Options:
- `--detail=<level>` - Set detail level (brief, standard, comprehensive)
- `--focus=<path>` - Focus on a specific directory
- `--no-metrics` - Exclude code quality metrics

For backward compatibility, running just `go run main.go <directory path>` will perform the indexing operation.

## 💡 How It Works

1. **Code Scanning**: Codie scans your codebase for supported file types (.py, .js, .go, etc.)
2. **Smart Chunking**: Files are broken into meaningful semantic chunks using Tree-sitter parsers for better analysis
3. **Efficient Batch Processing**: Code chunks are processed in batches through OpenAI's embedding API for optimal performance
4. **AI Embeddings**: Generated embeddings capture the semantic meaning of your code
5. **AI Analysis**: Codie builds a prompt based on your code and uses OpenAI's models to generate insightful summaries
6. **Structured Output**: Summaries include overview, architecture, key features, and implementation details

## 🤖 Features

- [x] **Code Repo Indexing** – Process and embed code for AI analysis
- [x] **Code Repo Summarization** – AI-powered summaries for understanding large codebases
- [x] **Focused Analysis** – Zoom in on specific directories or parts of your codebase
- [x] **Configurable Detail Levels** – Choose between brief, standard, or comprehensive summaries
- [x] **Syntax-Aware Parsing** – Uses Tree-sitter to understand code structure for smarter analysis

## 🔮 Upcoming Features

- [ ] **Functionality Finder** – Locate specific functions or features within a project
- [ ] **Code Requests** – Ask Codie for code snippets, explanations, and recommendations

## 🔧 Technical Details

Codie uses a combination of technologies to provide intelligent code analysis:

- **Tree-sitter** for language-specific parsing of code structures (functions, classes, methods)
- **Anthropic SmallEmbedding3** model for semantic code processing
- **OpenAI's GPT models** for generating the final analysis
- **Concurrent processing** for efficient handling of large codebases
- **Rate limiting** to manage API usage and prevent throttling

## 📚 Dependencies

- `github.com/charmbracelet/glamour` - For Markdown rendering in terminal
- `github.com/schollz/progressbar/v3` - For progress visualization
- `github.com/sashabaranov/go-openai` - OpenAI API client
- `github.com/smacker/go-tree-sitter` - Code parsing and analysis
- Tree-sitter language parsers for Go, JavaScript, Python, and more

## 📄 License

[MIT License](LICENSE)