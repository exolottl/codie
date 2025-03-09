package config

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/joho/godotenv"
)

// Init initializes the application configuration
// It loads environment variables and ensures the OpenAI API key is set
func Init() error {
	// Load environment variables
	err := godotenv.Load()
	if err != nil {
		fmt.Println("No .env file found. Let's create one.")
		err = createEnvFile()
		if err != nil {
			return fmt.Errorf("failed to create .env file: %v", err)
		}
	}

	// Check if OPENAI_API_KEY is set
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("OPENAI_API_KEY not found in environment variables.")
		apiKey, err = promptForAPIKey()
		if err != nil {
			return fmt.Errorf("failed to get API key: %v", err)
		}
		
		// Save the API key to .env file
		err = updateEnvFile("OPENAI_API_KEY", apiKey)
		if err != nil {
			return fmt.Errorf("failed to save API key to .env file: %v", err)
		}
		
		// Set the API key in the current environment
		os.Setenv("OPENAI_API_KEY", apiKey)
	}

	return nil
}

// promptForAPIKey asks the user to input their OpenAI API key
func promptForAPIKey() (string, error) {
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Please enter your OpenAI API key: ")
	apiKey, err := reader.ReadString('\n')
	if err != nil {
		return "", err
	}
	
	// Clean the input
	apiKey = strings.TrimSpace(apiKey)
	
	// Basic validation
	if apiKey == "" {
		return "", fmt.Errorf("API key cannot be empty")
	}
	
	// Verify the API key format (basic check)
	if !strings.HasPrefix(apiKey, "sk-") {
		fmt.Println("Warning: OpenAI API keys typically start with 'sk-'. Please verify your key is correct.")
	}
	
	return apiKey, nil
}

// createEnvFile creates a new .env file with default values
func createEnvFile() error {
	file, err := os.Create(".env")
	if err != nil {
		return err
	}
	defer file.Close()
	
	// Prompt user for API key
	apiKey, err := promptForAPIKey()
	if err != nil {
		return err
	}
	
	// Write API key to file
	_, err = file.WriteString(fmt.Sprintf("OPENAI_API_KEY=%s\n", apiKey))
	if err != nil {
		return err
	}
	
	fmt.Println("Created .env file with your API key.")
	return nil
}

// updateEnvFile updates a value in the .env file or adds it if it doesn't exist
func updateEnvFile(key, value string) error {
	// Try to read existing .env file
	content, err := os.ReadFile(".env")
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	
	// Check if file exists
	if os.IsNotExist(err) {
		// If file doesn't exist, create it with the key-value pair
		return os.WriteFile(".env", []byte(fmt.Sprintf("%s=%s\n", key, value)), 0644)
	}
	
	// Convert content to string and split by lines
	lines := strings.Split(string(content), "\n")
	keyFound := false
	
	// Check if key exists and update it
	for i, line := range lines {
		if strings.HasPrefix(line, key+"=") {
			lines[i] = fmt.Sprintf("%s=%s", key, value)
			keyFound = true
			break
		}
	}
	
	// If key wasn't found, add it
	if !keyFound {
		lines = append(lines, fmt.Sprintf("%s=%s", key, value))
	}
	
	// Join lines and write back to file
	return os.WriteFile(".env", []byte(strings.Join(lines, "\n")), 0644)
}