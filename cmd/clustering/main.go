package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type CaseData struct {
	Id        string    `json:"Id"`
	CaseDate  string    `json:"CaseDate"`
	CityName  string    `json:"CityName"`
	CityId    int       `json:"CityId"`
	Summary   string    `json:"Summary"`
	CaseTitle string    `json:"CaseTitle"`
	Embedding []float32 `json:"embedding"`
}

type Document struct {
	Case      CaseData
	Embedding []float64
}

type Cluster struct {
	Id        string     `json:"id"`
	Name      string     `json:"name"`
	Documents []Document `json:"-"` // Not included in JSON output
	DocIds    []string   `json:"docIds"`
	Keywords  []string   `json:"keywords"`
	Summary   string     `json:"summary"`
	Centroid  []float64  `json:"centroid"`
}

// ClusterIndex maps document IDs to their cluster IDs for quick lookup
type ClusterIndex map[string]string

var openaiApiKey string

func kmeans(embeddings [][]float64, k int, maxIter int) ([]int, [][]float64) {
	n := len(embeddings)
	dim := len(embeddings[0])
	labels := make([]int, n)
	centroids := make([][]float64, k)

	// Initialize centroids randomly
	for i := 0; i < k; i++ {
		centroids[i] = make([]float64, dim)
		copy(centroids[i], embeddings[i])
	}

	for iter := 0; iter < maxIter; iter++ {
		// Assign points to nearest centroid
		changed := false
		for i := 0; i < n; i++ {
			minDist := math.Inf(1)
			bestCluster := 0

			for j := 0; j < k; j++ {
				dist := euclideanDistance(embeddings[i], centroids[j])
				if dist < minDist {
					minDist = dist
					bestCluster = j
				}
			}

			if labels[i] != bestCluster {
				labels[i] = bestCluster
				changed = true
			}
		}

		if !changed {
			break
		}

		// Update centroids
		newCentroids := make([][]float64, k)
		counts := make([]int, k)
		for i := 0; i < k; i++ {
			newCentroids[i] = make([]float64, dim)
		}

		for i := 0; i < n; i++ {
			cluster := labels[i]
			counts[cluster]++
			for j := 0; j < dim; j++ {
				newCentroids[cluster][j] += embeddings[i][j]
			}
		}

		for i := 0; i < k; i++ {
			if counts[i] > 0 {
				for j := 0; j < dim; j++ {
					newCentroids[i][j] /= float64(counts[i])
				}
			}
		}

		centroids = newCentroids
	}

	return labels, centroids
}

func nameCluster(cluster Cluster) string {
	// Extract keywords using TF-IDF
	keywords := extractKeywords(cluster.Documents)
	cluster.Keywords = keywords

	// For simplicity, just use keywords as the name
	// Removed the use of CaseTitle as per requirement
	if len(keywords) > 0 {
		return strings.Join(keywords, " ")
	}

	return fmt.Sprintf("Cluster %s", cluster.Id)
}

func extractKeywords(docs []Document) []string {
	if len(docs) == 0 {
		return []string{}
	}

	// Take the center documents to represent the cluster
	centerDocs := findCenterDocuments(docs, 3)

	// Compile the summaries from center documents
	var summaries []string
	for _, doc := range centerDocs {
		summaries = append(summaries, doc.Case.Summary)
	}

	content := strings.Join(summaries, "\n\n")

	// Call OpenAI API to generate keywords
	keywords, err := getKeywordsFromOpenAI(content)
	if err != nil {
		fmt.Printf("Error getting keywords from OpenAI: %v\n", err)
		// Return a default value to avoid breaking the program
		return []string{"無法分類"}
	}

	return parseKeywords(keywords)
}

func findCenterDocuments(docs []Document, n int) []Document {
	if len(docs) <= n {
		return docs
	}

	centroid := calculateCentroid(docs)

	type docDistance struct {
		doc      Document
		distance float64
	}

	distances := make([]docDistance, len(docs))
	for i, doc := range docs {
		distances[i] = docDistance{
			doc:      doc,
			distance: euclideanDistance(doc.Embedding, centroid),
		}
	}

	sort.Slice(distances, func(i, j int) bool {
		return distances[i].distance < distances[j].distance
	})

	result := make([]Document, n)
	for i := 0; i < n; i++ {
		result[i] = distances[i].doc
	}

	return result
}

func euclideanDistance(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func calculateCentroid(docs []Document) []float64 {
	if len(docs) == 0 {
		return nil
	}

	dim := len(docs[0].Embedding)
	centroid := make([]float64, dim)

	for _, doc := range docs {
		for i, val := range doc.Embedding {
			centroid[i] += val
		}
	}

	for i := range centroid {
		centroid[i] /= float64(len(docs))
	}

	return centroid
}

// Convert float32 array to float64 array
func convertEmbedding(embedding []float32) []float64 {
	result := make([]float64, len(embedding))
	for i, v := range embedding {
		result[i] = float64(v)
	}
	return result
}

// Save individual cluster to a JSON file
func saveClusterToFile(cluster Cluster, outputDir string) error {
	filename := filepath.Join(outputDir, fmt.Sprintf("cluster-%s.json", cluster.Id))
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(cluster)
}

// Save cluster index to a JSON file
func saveClusterIndexToFile(index ClusterIndex, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(index)
}

func loadCaseData(embeddingDir string, id string) (*CaseData, error) {
	filePath := filepath.Join(embeddingDir, fmt.Sprintf("%s.json", id))

	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open case file %s: %w", filePath, err)
	}
	defer file.Close()

	content, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read case file %s: %w", filePath, err)
	}

	var caseData CaseData
	if err := json.Unmarshal(content, &caseData); err != nil {
		return nil, fmt.Errorf("failed to parse JSON file %s: %w", filePath, err)
	}

	if len(caseData.Summary) <= 100 {
		return nil, fmt.Errorf("summary length for case %s is not greater than 100 characters", id)
	}
	return &caseData, nil
}

// Load all case data from a directory
func loadAllCaseData(embeddingDir string) ([]CaseData, error) {
	// Get all JSON files in the directory
	files, err := filepath.Glob(filepath.Join(embeddingDir, "*.json"))
	if err != nil {
		return nil, fmt.Errorf("failed to read directory %s: %w", embeddingDir, err)
	}

	if len(files) == 0 {
		return nil, fmt.Errorf("no JSON files found in directory %s", embeddingDir)
	}

	var cases []CaseData
	var loadErrors []string

	// Process each file
	for _, file := range files {
		// Extract ID from filename (remove path and .json extension)
		id := filepath.Base(file)
		id = id[:len(id)-5] // Remove .json extension

		caseData, err := loadCaseData(embeddingDir, id)
		if err != nil {
			loadErrors = append(loadErrors, fmt.Sprintf("Error loading %s: %v", id, err))
			continue
		}

		cases = append(cases, *caseData)
	}

	// Print load errors but continue processing
	if len(loadErrors) > 0 {
		fmt.Println("Encountered errors loading some files:")
		for _, err := range loadErrors {
			fmt.Println("  -", err)
		}
		fmt.Printf("Successfully loaded %d out of %d files\n", len(cases), len(files))
	}

	if len(cases) == 0 {
		return nil, fmt.Errorf("no valid case data could be loaded from directory %s", embeddingDir)
	}

	return cases, nil
}

// Main function
func main() {
	// Parse command-line flags
	embeddingDir := flag.String("embedding-dir", "", "Directory containing case embedding JSON files")
	numClusters := flag.Int("clusters", 5, "Number of clusters to create")
	outputDir := flag.String("output-dir", ".", "Directory to save output files")
	flag.Parse()

	if *embeddingDir == "" {
		fmt.Println("Error: embedding-dir flag is required")
		flag.Usage()
		os.Exit(1)
	}

	// Get API key from environment
	apiKey := os.Getenv("OPENAI_API_KEY")
	openaiApiKey = apiKey
	if openaiApiKey == "" {
		fmt.Println("Error: please specify OPENAI_API_KEY in .env or your system environment variable")
		return
	}

	// Load all case data
	fmt.Printf("Loading case data from %s...\n", *embeddingDir)
	casesData, err := loadAllCaseData(*embeddingDir)
	if err != nil {
		fmt.Printf("Error loading case data: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Loaded %d cases successfully\n", len(casesData))

	// Convert CaseData to Document
	docs := make([]Document, len(casesData))
	for i, c := range casesData {
		docs[i] = Document{
			Case:      c,
			Embedding: convertEmbedding(c.Embedding),
		}
	}

	// Extract embeddings
	embeddings := make([][]float64, len(docs))
	for i, doc := range docs {
		embeddings[i] = doc.Embedding
	}

	// Perform clustering
	k := *numClusters
	fmt.Printf("Performing k-means clustering with k=%d...\n", k)
	labels, centroids := kmeans(embeddings, k, 100)

	// Create clusters
	clusters := make([]Cluster, k)
	for i := range clusters {
		// Initial ID assignment, will be updated later
		clusters[i].Id = fmt.Sprintf("%d%d", time.Now().UnixNano(), i)
		clusters[i].Centroid = centroids[i]
	}

	// Assign documents to clusters and store document IDs
	for i := 0; i < len(docs); i++ {
		clusterIdx := labels[i]
		clusters[clusterIdx].Documents = append(clusters[clusterIdx].Documents, docs[i])
		clusters[clusterIdx].DocIds = append(clusters[clusterIdx].DocIds, docs[i].Case.Id)
	}

	// Update cluster IDs using center document's CaseData.Id
	for i := range clusters {
		centerDocs := findCenterDocuments(clusters[i].Documents, 1)
		if len(centerDocs) > 0 {
			// Use the Id from the closest document to centroid
			clusters[i].Id = centerDocs[0].Case.Id
		}
	}

	// Create a document ID to cluster ID index
	clusterIndex := make(ClusterIndex)
	for _, cluster := range clusters {
		for _, doc := range cluster.Documents {
			clusterIndex[doc.Case.Id] = cluster.Id
		}
	}

	// Name clusters
	for i := range clusters {
		clusters[i].Name = nameCluster(clusters[i])
		clusters[i].Summary = generateClusterSummary(clusters[i])
	}

	// Create output directory if it doesn't exist
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		fmt.Printf("Error creating output directory: %v\n", err)
		return
	}

	// Save the cluster index to a JSON file
	clusterIndexPath := filepath.Join(*outputDir, "cluster-index.json")
	err = saveClusterIndexToFile(clusterIndex, clusterIndexPath)
	if err != nil {
		fmt.Printf("Error saving cluster index: %v\n", err)
		return
	}

	// Save each cluster to an individual file
	for _, cluster := range clusters {
		err := saveClusterToFile(cluster, *outputDir)
		if err != nil {
			fmt.Printf("Error saving cluster %s: %v\n", cluster.Id, err)
		}
	}

	fmt.Printf("Clustering completed and saved to:\n- %s\n", clusterIndexPath)
	fmt.Printf("- Individual cluster files in format cluster-{CaseData.Id}.json\n")
}

func generateClusterSummary(cluster Cluster) string {
	// Placeholder for summary generation
	// In a real application, you might generate a summary from the documents
	return fmt.Sprintf("Summary for cluster %s with %d documents", cluster.Id, len(cluster.Documents))
}

// generateClusterName combines information from center documents and keywords
func generateClusterName(centerDocs []Document, keywords []string) string {
	// Placeholder for name generation logic
	if len(keywords) > 0 {
		return keywords[0]
	}

	return "Unnamed Cluster"
}

// getKeywordsFromOpenAI calls the OpenAI API to generate keywords for a given content
func getKeywordsFromOpenAI(content string) (string, error) {
	// Structure for OpenAI Chat API request
	type Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}

	type OpenAIChatRequest struct {
		Model    string    `json:"model"`
		Messages []Message `json:"messages"`
	}

	type OpenAIChatResponse struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	// Prepare the API request with the prompt
	prompt := "請幫我用10個繁體中文字，為這篇內容下標題，請僅回傳標題即可"

	requestData := OpenAIChatRequest{
		Model: "gpt-4o",
		Messages: []Message{
			{Role: "system", Content: "You are a helpful assistant that creates concise titles in traditional Chinese."},
			{Role: "user", Content: content},
			{Role: "user", Content: prompt},
		},
	}

	jsonData, err := json.Marshal(requestData)
	if err != nil {
		return "", fmt.Errorf("error marshalling request: %w", err)
	}

	// Make the API request
	client := &http.Client{}
	req, err := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("error creating request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", openaiApiKey))

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("error making request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("error reading response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API request failed with status code %d: %s", resp.StatusCode, string(body))
	}

	// Parse the response
	var chatResponse OpenAIChatResponse
	err = json.Unmarshal(body, &chatResponse)
	if err != nil {
		return "", fmt.Errorf("error unmarshalling response: %w", err)
	}

	if len(chatResponse.Choices) == 0 {
		return "", fmt.Errorf("no choices in API response")
	}

	return chatResponse.Choices[0].Message.Content, nil
}

// parseKeywords splits the response from OpenAI into individual keywords
func parseKeywords(response string) []string {
	// Trim any whitespace
	response = strings.TrimSpace(response)

	// Split by any common separators (comma, space, newline)
	separators := []string{",", "，", " ", "、", "\n", "；", ";"}

	var keywords []string
	// Start with the full response as first keyword
	keywords = append(keywords, response)

	// Try to split the response by different separators
	for _, sep := range separators {
		if strings.Contains(response, sep) {
			// Split by this separator and clean the results
			parts := strings.Split(response, sep)
			var cleanParts []string

			for _, part := range parts {
				part = strings.TrimSpace(part)
				if part != "" {
					cleanParts = append(cleanParts, part)
				}
			}

			// Only use this split if it resulted in multiple parts
			if len(cleanParts) > 1 {
				keywords = cleanParts
				break
			}
		}
	}

	return keywords
}
