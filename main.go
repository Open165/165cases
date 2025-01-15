package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/joho/godotenv"
	"github.com/pkoukk/tiktoken-go"
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

const (
	ModelEmbedding          = "text-embedding-3-large"
	ModelEmbeddingMaxTokens = 8191
	EncodingEmbeding        = "cl100k_base"
	ApiUrlEmbedding         = "https://api.openai.com/v1/embeddings"
)

type DocumentSimilarity struct {
	Similarity float32
	Index      string
}

type OpenaiEmbeddingRequest struct {
	Input string `json:"input"`
	Model string `json:"model"`
}

type OpenaiEmbeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

type Response struct {
	Similarity []DocumentSimilarity `json:"similarity"`
	Documents  map[string]CaseData  `json:"documents"`
}

var openaiApiKey string
var force bool
var embeddingDir string
var resultLength int
var embeddings map[string][]float32

func cosineSimilarity(v1, v2 []float32) float32 {
	dot := dotProduct(v1, v2)
	return dot
	/* https://platform.openai.com/docs/guides/embeddings/which-distance-function-should-i-use
	v1Magnitude := math.Sqrt(float64(dotProduct(v1, v1)))
	v2Magnitude := math.Sqrt(float64(dotProduct(v2, v2)))
	return float32(float64(dot) / (v1Magnitude * v2Magnitude))
	*/
}

func dotProduct(v1, v2 []float32) float32 {
	var result float32
	for i := 0; i < len(v1); i++ {
		result += v1[i] * v2[i]
	}
	return result
}

func orderDocumentSectionsInQuerySimilarity(query string, resultLen int) ([]DocumentSimilarity, error) {
	queryEmbedding, err := openaiGetEmbedding(query)
	if err != nil {
		return nil, err
	}

	documentSimilarities := make([]DocumentSimilarity, 0, len(embeddings))
	for docIndex, docEmbedding := range embeddings {
		similarity := cosineSimilarity(queryEmbedding, docEmbedding)
		documentSimilarities = append(documentSimilarities, DocumentSimilarity{similarity, docIndex})
	}

	sort.Slice(documentSimilarities, func(i, j int) bool {
		return documentSimilarities[i].Similarity > documentSimilarities[j].Similarity
	})

	// Return the specified number of results
	if resultLen > len(documentSimilarities) {
		resultLen = len(documentSimilarities)
	}

	return documentSimilarities[:resultLen], nil
}

func loadEmbeddings(embeddingDir string) (map[string][]float32, error) {
	// Initialize map to store embeddings
	embedd := make(map[string][]float32)

	// Check if directory exists
	if _, err := os.Stat(embeddingDir); os.IsNotExist(err) {
		return nil, fmt.Errorf("directory does not exist: %s", embeddingDir)
	}

	// Walk through all JSON files in the directory
	err := filepath.Walk(embeddingDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Process only .json files
		if !info.IsDir() && filepath.Ext(path) == ".json" {
			// Open file
			file, err := os.Open(path)
			if err != nil {
				return fmt.Errorf("unable to open file %s: %w", path, err)
			}
			defer file.Close()

			// Read file content
			content, err := io.ReadAll(file)
			if err != nil {
				return fmt.Errorf("unable to read file %s: %w", path, err)
			}

			// Parse JSON
			var caseData CaseData
			if err := json.Unmarshal(content, &caseData); err != nil {
				return fmt.Errorf("unable to parse JSON file %s: %w", path, err)
			}

			// Store in map using Id as key
			embedd[caseData.Id] = caseData.Embedding
		}
		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("error processing files: %w", err)
	}

	return embedd, nil
}

func openaiGetEmbedding(inputText string) ([]float32, error) {
	// currently, cl100k_base is v2 of openai, max input tokens are 8191
	// we hard coded max string length by 8191
	inputLength := numTokensFromString(inputText, ModelEmbedding)
	if inputLength > ModelEmbeddingMaxTokens {
		inputText = string([]rune(inputText)[:ModelEmbeddingMaxTokens])
		fmt.Println("Truncate string from inputLength to maxLength. Original string:")
		fmt.Println(inputText)
	}

	requestData := OpenaiEmbeddingRequest{
		Input: inputText,
		Model: ModelEmbedding,
	}
	jsonData, err := json.Marshal(requestData)
	if err != nil {
		return nil, err
	}

	client := &http.Client{}
	req, err := http.NewRequest("POST", ApiUrlEmbedding, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", openaiApiKey))

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("request failed with status code %d: %s", resp.StatusCode, string(body))
	}

	var embeddingResponse OpenaiEmbeddingResponse
	err = json.Unmarshal(body, &embeddingResponse)
	if err != nil {
		return nil, err
	}

	if len(embeddingResponse.Data) == 0 {
		return nil, fmt.Errorf("empty data field in the API response")
	}

	return embeddingResponse.Data[0].Embedding, nil
}

func numTokensFromString(input string, model string) (num_tokens int) {
	tkm, err := tiktoken.EncodingForModel(model)
	if err != nil {
		log.Printf("EncodingForModel: %v", err)
		return
	}

	return len(tkm.Encode(input, nil, nil))
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

	return &caseData, nil
}

// Handler function for the incoming HTTP request
func requestHandler(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query().Get("q")

	// serve html when no query
	if query == "" {
		frontHtmlPath := "public/index.html"
		if _, err := os.Stat(frontHtmlPath); os.IsNotExist(err) {
			w.WriteHeader(http.StatusNotFound)
			w.Write([]byte("index.html not found"))
			return
		}
		http.ServeFile(w, r, frontHtmlPath)
		return
	}

	documentSimilarities, err := orderDocumentSectionsInQuerySimilarity(query, resultLength)
	if err != nil {
		http.Error(w, fmt.Sprintf("Error ordering document sections: %s", err), http.StatusInternalServerError)
		return
	}

	relevantDocs := make(map[string]CaseData)
	for _, sim := range documentSimilarities {
		doc, err := loadCaseData(embeddingDir, sim.Index)
		if err != nil {
			fmt.Printf("failed to load document %s: %w", sim.Index, err)
			continue
		}

		// Create a copy without the embedding field
		docWithoutEmbedding := *doc
		docWithoutEmbedding.Embedding = nil
		relevantDocs[sim.Index] = docWithoutEmbedding
	}

	// Prepare and send the response
	response := struct {
		Similarity []DocumentSimilarity `json:"similarity"`
		Documents  map[string]CaseData  `json:"documents"`
	}{
		Similarity: documentSimilarities,
		Documents:  relevantDocs,
	}

	jsonResponse, err := json.Marshal(response)
	if err != nil {
		http.Error(w, "Error marshalling response", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(jsonResponse)
}

func init() {
	flag.StringVar(&embeddingDir, "embedding-dir", "", "[Required] Path to the OpenAI embedding dir")
	flag.IntVar(&resultLength, "result-number", 5, "[Optional] Result number of similarity")
  flag.BoolVar(&force, "force", false, "[Optional] Skip user confirmation if set")
}

func promptUserConfirmation() bool {
	reader := bufio.NewReader(os.Stdin)

	fmt.Println("\nIMPORTANT NOTICE:")
	fmt.Println("This service will use OpenAI API with your provided API key.")
	fmt.Println("Each request will incur charges to your OpenAI account.")
	fmt.Print("Do you want to continue? (y/n): ")

	response, err := reader.ReadString('\n')
	if err != nil {
		fmt.Println("Error reading input:", err)
		return false
	}

	// Clean the input and convert to lowercase
	response = strings.ToLower(strings.TrimSpace(response))
	return response == "y" || response == "yes"
}

func main() {
	flag.Parse()

	// Check if the required flags are provided
	if embeddingDir == "" {
		flag.Usage()
		return
	}

	if err := godotenv.Load(); err != nil {
		log.Printf("Empty .env, use OS env instead: %v", err)
	}

	apiKey := os.Getenv("OPENAI_API_KEY")
	openaiApiKey = apiKey
	if openaiApiKey == "" {
		fmt.Println("Error: please specify OPENAI_API_KEY in .env or your system environment variable")
		return
	}
	log.Printf("OpenAI API Key: %s ... done\n", openaiApiKey[0:10])

	// Prompt for user confirmation
  if !force && !promptUserConfirmation() {
		fmt.Println("Service startup cancelled by user")
		return
	}

	// Load embeddings
	embedd, err := loadEmbeddings(embeddingDir)
	if err != nil {
		log.Fatalf("Error loading embeddings: %s", err)
		return
	}
	embeddings = embedd

	// Start the HTTP server and listen for incoming requests
	http.HandleFunc("/", requestHandler)
	log.Println("Starting server on :9989")
	log.Fatal(http.ListenAndServe(":9989", nil))
}
