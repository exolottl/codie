package config

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"github.com/sashabaranov/go-openai"
)

// Init initializes the application configuration
// It loads environment variables and ensures the OpenAI API key is set and valid
func Init() error {
	// Load environment variables if .env file exists
	envFileExists := true
	err := godotenv.Load()
	if err != nil {
		envFileExists = false
		fmt.Println("No .env file found.")
	}

	// Check if OPENAI_API_KEY is already set in environment
	apiKey := os.Getenv("OPENAI_API_KEY")
	
	// If key is present, validate it first before proceeding
	if apiKey != "" {
		if err := validateAPIKey(apiKey); err != nil {
			fmt.Printf("Existing API key is invalid: %v\n", err)
			// Clear the invalid key from environment
			apiKey = ""
		} else {
			fmt.Println("Existing OpenAI API key validated successfully.")
			return nil
		}
	}
	
	// If we reach here, we need a new API key from user
	fmt.Println("Please provide a valid OpenAI API key.")
	var validKey string
	maxAttempts := 3
	
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if attempt > 1 {
			fmt.Printf("Attempt %d of %d to provide a valid API key.\n", attempt, maxAttempts)
		}
		
		newKey, err := promptForAPIKey()
		if err != nil {
			return fmt.Errorf("failed to get API key: %v", err)
		}
		
		// Validate the new key before saving
		if err := validateAPIKey(newKey); err != nil {
			fmt.Printf("Invalid API key: %v\nPlease try again.\n", err)
			continue
		}
		
		// If we reach here, the key is valid
		validKey = newKey
		break
	}
	
	// Check if we got a valid key after attempts
	if validKey == "" {
		return fmt.Errorf("failed to obtain a valid OpenAI API key after %d attempts", maxAttempts)
	}
	
	// At this point we have a validated key - now save it to .env and environment
	fmt.Println("API key validated successfully.")
	
	// Set the API key in the current environment
	os.Setenv("OPENAI_API_KEY", validKey)
	
	// Now save to .env file
	if !envFileExists {
		err = createEnvFile(validKey)
		if err != nil {
			return fmt.Errorf("failed to create .env file: %v", err)
		}
	} else {
		// Update existing .env file
		err = updateEnvFile("OPENAI_API_KEY", validKey)
		if err != nil {
			return fmt.Errorf("failed to update API key in .env file: %v", err)
		}
	}
	
	return nil
}

// validateAPIKey checks if the provided API key is valid by making a small test request
func validateAPIKey(apiKey string) error {
	// Verify the API key format first (basic check)
	if !strings.HasPrefix(apiKey, "sk-") {
		fmt.Println("Warning: OpenAI API keys typically start with 'sk-'. Proceeding with validation anyway.")
	}
	
	client := openai.NewClient(apiKey)
	
	// Create a context with timeout to avoid hanging
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	// Try to create a small embedding to validate the API key
	fmt.Println("Validating OpenAI API key...")
	_, err := client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Model: openai.AdaEmbeddingV2,
		Input: []string{"test"},
	})
	
	if err != nil {
		// Check for common error patterns that indicate an invalid API key
		errStr := err.Error()
		if strings.Contains(errStr, "401") || 
		   strings.Contains(errStr, "invalid_api_key") || 
		   strings.Contains(errStr, "authentication") {
			return fmt.Errorf("authentication failed: %v (the API key appears to be invalid)", err)
		}
		
		// For rate limiting or other temporary issues, provide a clearer message
		if strings.Contains(errStr, "429") || strings.Contains(errStr, "rate_limit") {
			return fmt.Errorf("rate limit exceeded: %v (the API key is valid but you've hit rate limits)", err)
		}
		
		// For other errors, still return but with a more nuanced message
		return fmt.Errorf("API key may be valid but encountered an error: %v", err)
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
	
	return apiKey, nil
}

// createEnvFile creates a new .env file with a validated API key
func createEnvFile(validatedKey string) error {
	file, err := os.Create(".env")
	if err != nil {
		return err
	}
	defer file.Close()
	
	// Write the validated API key to file
	_, err = file.WriteString(fmt.Sprintf("OPENAI_API_KEY=%s\n", validatedKey))
	if err != nil {
		return err
	}
	
	fmt.Println("Created .env file with your validated API key.")
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