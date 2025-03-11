package fileutils

import (
	"bufio"
	"bytes"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// Common code file extensions to process
var codeExtensions = map[string]bool{
	".py":    true,
	".js":    true,
	".ts":    true,
	".cpp":   true,
	".go":    true,
	".java":  true,
	".lua":   true,
	".jsx":   true,
	".tsx":   true,
	".html":  true,
	".css":   true,
	".php":   true,
	".rb":    true,
	".rs":    true,
	".cs":    true,
	".swift": true,
	".kt":    true,
}

// Common directories to skip
var skipDirs = map[string]bool{
	".git":         true,
	"node_modules": true,
	"venv":         true,
	"__pycache__":  true,
	"dist":         true,
	"build":        true,
	".idea":        true,
	".vscode":      true,
}

// ContentCache provides file content caching
type ContentCache struct {
	cache  map[string]CachedContent
	mutex  sync.RWMutex
	maxAge time.Duration // Maximum time to keep cache entries
}

// CachedContent holds cached file content and metadata
type CachedContent struct {
	content    string
	modTime    time.Time
	accessTime time.Time
}

// NewContentCache creates a new content cache
func NewContentCache(maxAge time.Duration) *ContentCache {
	if maxAge <= 0 {
		maxAge = 5 * time.Minute // Default cache expiration
	}
	return &ContentCache{
		cache:  make(map[string]CachedContent),
		maxAge: maxAge,
	}
}

// Get retrieves content from cache if available and not expired
func (c *ContentCache) Get(filePath string) (string, bool) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	cached, exists := c.cache[filePath]
	if !exists {
		return "", false
	}

	// Check if cache entry has expired
	if time.Since(cached.accessTime) > c.maxAge {
		return "", false
	}

	// Check if file has been modified
	info, err := os.Stat(filePath)
	if err != nil || info.ModTime().After(cached.modTime) {
		return "", false
	}

	// Update access time (requires write lock)
	c.mutex.RUnlock()
	c.mutex.Lock()
	if entry, stillExists := c.cache[filePath]; stillExists {
		entry.accessTime = time.Now()
		c.cache[filePath] = entry
	}
	c.mutex.Unlock()
	c.mutex.RLock()

	return cached.content, true
}

// Set adds a file's content to the cache
func (c *ContentCache) Set(filePath, content string) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	info, err := os.Stat(filePath)
	if err == nil {
		c.cache[filePath] = CachedContent{
			content:    content,
			modTime:    info.ModTime(),
			accessTime: time.Now(),
		}
	}
}

// Clear empties the cache
func (c *ContentCache) Clear() {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.cache = make(map[string]CachedContent)
}

// Prune removes expired entries from the cache
func (c *ContentCache) Prune() {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	
	now := time.Now()
	for path, content := range c.cache {
		if now.Sub(content.accessTime) > c.maxAge {
			delete(c.cache, path)
		}
	}
}

// Buffer pool for reusing buffers
var bufferPool = sync.Pool{
	New: func() interface{} {
		return new(bytes.Buffer)
	},
}

// GetCodeFiles returns a list of code files in the given directory (serial version)
func GetCodeFiles(root string) ([]string, error) {
	// Pre-allocate slice with reasonable capacity
	files := make([]string, 0, 1000)
	
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		
		// Skip directories we want to exclude
		if info.IsDir() {
			if skipDirs[info.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		
		// Check if file has code extension
		ext := filepath.Ext(info.Name())
		if codeExtensions[ext] {
			files = append(files, path)
		}
		
		return nil
	})
	
	return files, err
}

// GetCodeFilesParallel returns a list of code files using concurrent directory traversal
func GetCodeFilesParallel(root string, maxWorkers int) ([]string, error) {
	if maxWorkers <= 0 {
		maxWorkers = runtime.NumCPU()
	}

	var files []string
	var mutex sync.Mutex
	errChan := make(chan error, 1)
	
	// Create a worker pool using semaphore pattern
	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup
	
	// Process directories concurrently
	var processDir func(path string)
	processDir = func(path string) {
		defer func() {
			<-sem // Release the semaphore slot
			wg.Done()
		}()
		
		entries, err := os.ReadDir(path)
		if err != nil {
			select {
			case errChan <- err:
			default:
			}
			return
		}
		
		// Process all directory entries
		for _, entry := range entries {
			entryPath := filepath.Join(path, entry.Name())
			
			if entry.IsDir() {
				if skipDirs[entry.Name()] {
					continue
				}
				
				wg.Add(1)
				// Try to acquire a semaphore slot
				select {
				case sem <- struct{}{}:
					// We got a slot, process in a new goroutine
					go processDir(entryPath)
				default:
					// No free slots, process in the current goroutine
					sem <- struct{}{} // Will block until a slot is available
					processDir(entryPath)
				}
			} else {
				ext := filepath.Ext(entry.Name())
				if codeExtensions[ext] {
					mutex.Lock()
					files = append(files, entryPath)
					mutex.Unlock()
				}
			}
		}
	}
	
	// Start the root directory
	wg.Add(1)
	sem <- struct{}{} // Acquire a slot
	go processDir(root)
	
	// Wait for all goroutines to finish
	wg.Wait()
	
	// Check for errors
	select {
	case err := <-errChan:
		return nil, err
	default:
		return files, nil
	}
}

// ReadFileContent reads a file and returns its content as a string
func ReadFileContent(filePath string) (string, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}
	return string(content), nil
}

// ReadFileContentCached reads a file with caching support
func ReadFileContentCached(filePath string, cache *ContentCache) (string, error) {
	// Try to get from cache first
	if cache != nil {
		if content, found := cache.Get(filePath); found {
			return content, nil
		}
	}
	
	// Read from disk if not in cache
	content, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}
	
	// Update cache
	if cache != nil {
		cache.Set(filePath, string(content))
	}
	
	return string(content), nil
}

// ReadLargeFile reads a large file using buffered I/O
func ReadLargeFile(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()
	
	// Get a buffer from the pool
	buffer := bufferPool.Get().(*bytes.Buffer)
	buffer.Reset()
	defer bufferPool.Put(buffer)
	
	// Use buffered reader for efficiency
	reader := bufio.NewReader(file)
	
	// Read in chunks
	buf := make([]byte, 32*1024) // 32KB chunks
	for {
		n, err := reader.Read(buf)
		if err != nil && err != io.EOF {
			return "", err
		}
		if n == 0 {
			break
		}
		
		buffer.Write(buf[:n])
	}
	
	return buffer.String(), nil
}

// ReadFilesInParallel reads multiple files concurrently
func ReadFilesInParallel(filePaths []string, maxWorkers int) (map[string]string, error) {
	if maxWorkers <= 0 {
		maxWorkers = runtime.NumCPU()
	}
	
	results := make(map[string]string)
	var mutex sync.Mutex
	errChan := make(chan error, 1)
	
	// Create worker pool
	jobs := make(chan string, len(filePaths))
	var wg sync.WaitGroup
	
	// Start workers
	for i := 0; i < maxWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for path := range jobs {
				content, err := os.ReadFile(path)
				if err != nil {
					select {
					case errChan <- err:
					default:
					}
					continue
				}
				
				mutex.Lock()
				results[path] = string(content)
				mutex.Unlock()
			}
		}()
	}
	
	// Send jobs
	for _, path := range filePaths {
		jobs <- path
	}
	close(jobs)
	
	// Wait for all workers to finish
	wg.Wait()
	
	// Check for errors
	select {
	case err := <-errChan:
		return nil, err
	default:
		return results, nil
	}
}

// SplitCodeIntoChunks splits a code string into chunks with improved logic
func SplitCodeIntoChunks(code string, maxChunkSize int) []string {
	if maxChunkSize <= 0 {
		maxChunkSize = 1000 // Default max chunk size
	}
	
	// Split by natural code separators
	rawChunks := strings.Split(code, "\n\n")
	
	chunks := make([]string, 0, len(rawChunks)/2) // Pre-allocate with conservative estimate
	var currentChunk strings.Builder
	currentChunk.Grow(maxChunkSize) // Pre-allocate builder capacity
	
	for _, chunk := range rawChunks {
		// Skip empty chunks
		trimmedChunk := strings.TrimSpace(chunk)
		if trimmedChunk == "" {
			continue
		}
		
		// If adding this chunk would exceed max size, finalize current chunk and start a new one
		if currentChunk.Len() > 0 && currentChunk.Len()+len(trimmedChunk) > maxChunkSize {
			chunks = append(chunks, currentChunk.String())
			currentChunk.Reset()
			currentChunk.Grow(maxChunkSize)
		}
		
		// Add the current chunk
		if currentChunk.Len() > 0 {
			currentChunk.WriteString("\n\n")
		}
		currentChunk.WriteString(trimmedChunk)
		
		// If the chunk itself is already bigger than max size, add it directly
		if currentChunk.Len() >= maxChunkSize {
			chunks = append(chunks, currentChunk.String())
			currentChunk.Reset()
			currentChunk.Grow(maxChunkSize)
		}
	}
	
	// Add any remaining content
	if currentChunk.Len() > 0 {
		chunks = append(chunks, currentChunk.String())
	}
	
	return chunks
}

// StreamChunksFromFile processes a large file in chunks without loading it all into memory
func StreamChunksFromFile(filePath string, maxChunkSize int, processor func(chunk string) error) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer file.Close()
	
	scanner := bufio.NewScanner(file)
	var currentChunk strings.Builder
	currentChunk.Grow(maxChunkSize)
	
	for scanner.Scan() {
		line := scanner.Text()
		
		if currentChunk.Len()+len(line)+1 > maxChunkSize && currentChunk.Len() > 0 {
			if err := processor(currentChunk.String()); err != nil {
				return err
			}
			currentChunk.Reset()
			currentChunk.Grow(maxChunkSize)
		}
		
		if currentChunk.Len() > 0 {
			currentChunk.WriteString("\n")
		}
		currentChunk.WriteString(line)
	}
	
	// Process the final chunk
	if currentChunk.Len() > 0 {
		if err := processor(currentChunk.String()); err != nil {
			return err
		}
	}
	
	return scanner.Err()
}

// ProcessFilesWithWorkerPool processes multiple files using a worker pool
func ProcessFilesWithWorkerPool(filePaths []string, workerCount int, processor func(path string) error) error {
	if workerCount <= 0 {
		workerCount = runtime.NumCPU()
	}
	
	jobs := make(chan string, len(filePaths))
	errChan := make(chan error, 1)
	done := make(chan struct{})
	
	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < workerCount; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for path := range jobs {
				if err := processor(path); err != nil {
					select {
					case errChan <- err:
					default:
					}
					return
				}
			}
		}()
	}
	
	// Close jobs channel when all workers finish
	go func() {
		wg.Wait()
		close(done)
	}()
	
	// Send jobs
	for _, path := range filePaths {
		select {
		case jobs <- path:
		case <-done:
			// If workers are done (possibly due to an error), stop sending jobs
			break
		}
	}
	close(jobs)
	
	// Wait for workers to finish
	<-done
	
	// Check for errors
	select {
	case err := <-errChan:
		return err
	default:
		return nil
	}
}