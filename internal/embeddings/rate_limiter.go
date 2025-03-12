package embeddings

import (
	"sync"
	"time"
)

// RateLimiter manages rate limiting for API calls
type RateLimiter struct {
	ticker    *time.Ticker
	mu        sync.Mutex
	semaphore chan struct{}
}

// NewRateLimiter creates a new rate limiter with the specified requests per minute
func NewRateLimiter(requestsPerMinute int, maxConcurrent int) *RateLimiter {
	if requestsPerMinute <= 0 {
		requestsPerMinute = 60 // Default: 1 per second
	}
	if maxConcurrent <= 0 {
		maxConcurrent = 5 // Default: 5 concurrent requests
	}
	
	interval := time.Minute / time.Duration(requestsPerMinute)
	return &RateLimiter{
		ticker:    time.NewTicker(interval),
		semaphore: make(chan struct{}, maxConcurrent),
	}
}

// Wait blocks until a request can be made according to rate limits
func (r *RateLimiter) Wait() {
	r.semaphore <- struct{}{} // Acquire semaphore
	r.mu.Lock()
	<-r.ticker.C
	r.mu.Unlock()
}

// Release releases the semaphore
func (r *RateLimiter) Release() {
	<-r.semaphore
}

// Global rate limiter for OpenAI API (3,500 RPM for ada-002 embeddings is the limit)
// Using 3,000 to be safe
var apiRateLimiter = NewRateLimiter(3000, 5)