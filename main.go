package main

import (
	"log"
	"os"
	
	"codie/cmd"
	"codie/internal/config"
)

func main() {
	// Initialize configuration with API key validation
	err := config.Init()
	if err != nil {
		log.Fatalf("Configuration error: %v", err)
	}

	if len(os.Args) < 2 {
		cmd.PrintUsage()
		os.Exit(1)
	}
	
	command := os.Args[1]
	
	switch command {
	case "help":
		cmd.PrintUsage()
		
	case "index":
		// Check if directory is provided
		if len(os.Args) < 3 {
			log.Fatal("Usage: go run main.go index <directory>")
		}
		dir := os.Args[2]
		cmd.IndexCodebase(dir)
		
	case "summarize":
		// Check if directory is provided
		if len(os.Args) < 3 {
			log.Fatal("Usage: go run main.go summarize <directory> [options]")
		}
		dir := os.Args[2]
		cmd.SummarizeCodebase(dir, os.Args[3:])
		
	default:
		// For backward compatibility, treat the first arg as directory
		// if it doesn't match a known command
		dir := os.Args[1]
		cmd.IndexCodebase(dir)
	}
}