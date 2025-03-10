# Codie - Terminal-Based AI Code Assistant

## 🚀 Built for Devs Doing Real Work

Codie is a powerful CLI tool designed to help developers quickly understand and navigate codebases. Whether you're onboarding a new project or exploring an unfamiliar repository, Codie provides AI-powered insights to accelerate your workflow.

## 🛠 Requirements

- Go 1.22 or higher
- OpenAI API key (you'll be prompted to provide this on first run)

## 📥 Installation

```sh
git clone https://github.com/exolottl/codie
cd codie
go mod tidy
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

1. **Code Scanning**: Codie scans your codebase for supported file types (.py, .js, .go, .java, etc.)
2. **Smart Chunking**: Large files are broken into meaningful chunks for better analysis
3. **AI Embeddings**: Code chunks are processed through OpenAI's embedding API
4. **AI Analysis**: Codie builds a prompt based on your code and uses OpenAI's GPT-4o to generate insightful summaries
5. **Structured Output**: Summaries include overview, architecture, key features, and implementation details

## 🤖 Features

- [x] **Code Repo Indexing** – Process and embed code for AI analysis
- [x] **Code Repo Summarization** – AI-powered summaries for understanding large codebases
- [x] **Focused Analysis** – Zoom in on specific directories or parts of your codebase
- [x] **Configurable Detail Levels** – Choose between brief, standard, or comprehensive summaries

## 🔮 Upcoming Features

- [ ] **Functionality Finder** – Locate specific functions or features within a project
- [ ] **Code Requests** – Ask Codie for code snippets, explanations, and recommendations

## 🔧 Technical Details

Codie uses OpenAI's embedding model (Ada) to process code chunks and GPT-4o for generating the final analysis. It's designed to work offline once the initial indexing is complete, making it ideal for working with private codebases.