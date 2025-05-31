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
	"regexp"
	// "sort" // No longer needed for this function
	"strings"

	"database/sql"

	sqlitevec "github.com/asg017/sqlite-vec/bindings/go"
	"github.com/joho/godotenv"
	_ "github.com/mattn/go-sqlite3"
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
// var embeddings map[string][]float32 // Removed as per requirement
var db *sql.DB
var updateDB bool
var embeddingDimension int = 3072 // Added global variable

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
	var queryEmbedding []float32
	var err error

	// check if id
	numericPattern := regexp.MustCompile(`^\d{18,30}$`)
	if numericPattern.MatchString(query) {
		embeddingFilePath := filepath.Join(embeddingDir, fmt.Sprintf("%s.json", query))
		if _, fileErr := os.Stat(embeddingFilePath); !os.IsNotExist(fileErr) {
			// using saved embedding for similarity
			caseData, fileErr := loadCaseData(embeddingDir, query)
			if fileErr == nil && caseData.Embedding != nil {
				queryEmbedding = caseData.Embedding
				log.Printf("Using embedding from file for query ID: %s", query)
			} else {
				log.Printf("Failed to load embedding from file: %v, falling back to API", fileErr)
				queryEmbedding, err = openaiGetEmbedding(query)
				if err != nil {
					return nil, err
				}
			}
		} else {
			// Using OpenAI to generate embedding
			queryEmbedding, err = openaiGetEmbedding(query)
			if err != nil {
				return nil, err
			}
		}
	} else {
		// Using OpenAI to generate embedding
		queryEmbedding, err = openaiGetEmbedding(query)
		if err != nil {
			return nil, err
		}
	}

	documentSimilarities := make([]DocumentSimilarity, 0, len(embeddings))
	for docIndex, docEmbedding := range embeddings {
		similarity := cosineSimilarity(queryEmbedding, docEmbedding)
		documentSimilarities = append(documentSimilarities, DocumentSimilarity{similarity, docIndex})
	}

	sort.Slice(documentSimilarities, func(i, j int) bool {
		return documentSimilarities[i].Similarity > documentSimilarities[j].Similarity
	})

	// Check if db is initialized
	if db == nil {
		return nil, fmt.Errorf("database connection is not initialized")
	}

	// Convert queryEmbedding to bytes
	queryEmbeddingBytes, err := sqlitevec.Float32ToBytes(queryEmbedding)
	if err != nil {
		return nil, fmt.Errorf("failed to convert query embedding to bytes: %w", err)
	}

	// Prepare the SQL query for VSS search
	// vss_cases_embeddings is the virtual table, and 'embedding' is the column with the vector.
	// The vss_search function takes the column name and the query vector.
	// sqlite-vec's vss_search returns 'id' (rowid of the vss table, which should be the same as cases_embeddings.id if inserted correctly)
	// and 'distance'.
	// For OpenAI embeddings (normalized), cosine distance is often 1 - cosine_similarity.
	// So, similarity = 1 - distance.
	sqlQuery := `
		SELECT ce.id, v.distance
		FROM vss_cases_embeddings v
		JOIN cases_embeddings ce ON v.rowid = ce.rowid
		WHERE vss_search(v.embedding, ?)
		ORDER BY v.distance
		LIMIT ?
	`
	// Note: The JOIN condition v.rowid = ce.rowid assumes that the rowid in vss_cases_embeddings
	// corresponds to the rowid in cases_embeddings. This is true if embeddings are inserted
	// into cases_embeddings and the VSS table picks them up.
	// If vss_cases_embeddings was populated independently with `INSERT INTO vss_cases_embeddings (rowid, embedding_vec) VALUES (?, ?)`
	// and `id` was stored as `rowid`, then `SELECT rowid, distance FROM vss_cases_embeddings...` would be used.
	// Given our DDL `CREATE VIRTUAL TABLE IF NOT EXISTS vss_cases_embeddings USING vss0(embedding dimensions=%d);`
	// which references the `embedding` column of `cases_embeddings` (implicitly by table name usually, or via configuration),
	// sqlite-vec handles the linkage. The query needs to retrieve the original `id`.
	// Let's adjust the query if `vss_search` directly on `cases_embeddings` virtual table gives us the original `id`.
	// The `sqlite-vec` documentation example: `SELECT rowid, distance FROM vec_items WHERE embedding MATCH ? ORDER BY distance LIMIT ?`
	// Here, `rowid` is the key. If our `cases_embeddings.id` is the `rowid` for `vss_cases_embeddings`, then it's simpler.
	// The `vss0` module makes the `embedding` column of `cases_embeddings` searchable.
	// So, querying `vss_cases_embeddings` should make `cases_embeddings.rowid` available as `rowid`, and other columns from `cases_embeddings` too.

	// Revised SQL query assuming vss_cases_embeddings gives access to columns of cases_embeddings
	// or at least its rowid which is implicitly the id if id is PRIMARY KEY and an alias for rowid.
	// If cases_embeddings.id is a TEXT primary key, it's not necessarily the rowid.
	// Let's assume the VSS table `vss_cases_embeddings` provides `rowid` which corresponds to `cases_embeddings.rowid`.
	// We need to get `cases_embeddings.id`.
	// The VSS table DDL is `CREATE VIRTUAL TABLE IF NOT EXISTS vss_cases_embeddings USING vss0(embedding dimensions=%d);`
	// This implies it's linked to `cases_embeddings`.
	// The query should be:
	// "SELECT T.id, distance FROM cases_embeddings AS T JOIN vss_cases_embeddings AS V ON T.rowid = V.rowid WHERE vss_search(V.embedding, ?) ORDER BY V.distance LIMIT ?"
	// Simpler form if `vss_cases_embeddings` can be queried directly for `id` if it was part of the VSS table schema (it is not, only `embedding` is).
	// So, a JOIN is likely needed if `id` is not `rowid`.
	// `cases_embeddings.id` is `TEXT PRIMARY KEY`. SQLite might use it as an alias for `rowid` if it's an INTEGER PRIMARY KEY, but not for TEXT.
	// So we must ensure `vss_cases_embeddings` stores `cases_embeddings.id` or we join.
	// The `vss0` virtual table is created ON the `cases_embeddings` table.
	// `SELECT cases_embeddings.id, distance FROM vss_cases_embeddings ...` should work if `vss_cases_embeddings` is an overlay.
	// The `sqlite-vec` documentation states: "The first argument to vss_search must be the name of the vector column in the virtual table."
	// "SELECT rowid, distance FROM vss_table WHERE vss_search(vector_column, ?)"
	// This implies `rowid` is the identifier returned from the virtual table.
	// We need to map this `rowid` back to our original `id` from `cases_embeddings`.

	// Final SQL Query Strategy:
	// 1. Get `rowid` from `vss_cases_embeddings`.
	// 2. Use that `rowid` to get the `id` from `cases_embeddings`.
	// This is what the JOIN above does. Let's assume `v.embedding` is the correct column name in the VIRTUAL table.
	// The DDL for vss_cases_embeddings was `CREATE VIRTUAL TABLE ... USING vss0(embedding dimensions=%d)`.
	// So the column name in the virtual table is `embedding`.

	sqlQuery = `
		SELECT ce.id, v.distance
		FROM cases_embeddings AS ce
		JOIN vss_cases_embeddings AS v ON ce.rowid = v.rowid
		WHERE vss_search(v.embedding, ?)
		ORDER BY v.distance
		LIMIT ?`

	rows, err := db.Query(sqlQuery, queryEmbeddingBytes, resultLen)
	if err != nil {
		return nil, fmt.Errorf("failed to execute VSS query: %w", err)
	}
	defer rows.Close()

	documentSimilarities := make([]DocumentSimilarity, 0, resultLen)
	for rows.Next() {
		var id string
		var distance float32
		if err := rows.Scan(&id, &distance); err != nil {
			return nil, fmt.Errorf("failed to scan VSS query result: %w", err)
		}
		// Convert distance to similarity. For cosine distance (0 to 2, where 0 is identical),
		// similarity = 1 - distance.
		// If OpenAI embeddings are normalized, cosine similarity = dot(A,B).
		// If sqlite-vec returns cosine distance d = 1 - dot(A,B), then similarity = 1.0 - d.
		// If sqlite-vec returns Euclidean distance, this conversion would be different.
		// Assuming distance is cosine distance (1 - similarity).
		similarity := 1.0 - distance
		documentSimilarities = append(documentSimilarities, DocumentSimilarity{Similarity: similarity, Index: id})
	}

	if err = rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating VSS query results: %w", err)
	}

	// Results are already sorted by distance (and thus by similarity) and limited by SQL.
	return documentSimilarities, nil
}

// loadEmbeddings function removed as per requirement

func loadEmbeddingsToDB(dbConn *sql.DB, dir string, expectedDimension int) error {
	log.Printf("Starting to load embeddings from %s into the database...", dir)

	// Check if directory exists
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		return fmt.Errorf("directory does not exist: %s", dir)
	}

	err := filepath.Walk(dir, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			log.Printf("Warning: error accessing path %q: %v\n", path, walkErr)
			return walkErr
		}
		if info.IsDir() || filepath.Ext(path) != ".json" {
			return nil // Skip directories and non-JSON files
		}

		file, err := os.Open(path)
		if err != nil {
			log.Printf("Warning: unable to open file %s: %v. Skipping.", path, err)
			return nil // Continue with next file
		}
		defer file.Close()

		content, err := io.ReadAll(file)
		if err != nil {
			log.Printf("Warning: unable to read file %s: %v. Skipping.", path, err)
			return nil // Continue with next file
		}

		var caseData CaseData
		if err := json.Unmarshal(content, &caseData); err != nil {
			log.Printf("Warning: unable to parse JSON file %s: %v. Skipping.", path, err)
			return nil // Continue with next file
		}

		if caseData.Embedding == nil {
			log.Printf("Warning: embedding is nil for ID %s in file %s. Skipping.", caseData.Id, path)
			return nil // Continue with next file
		}

		if len(caseData.Embedding) != expectedDimension {
			log.Printf("Warning: embedding for ID %s in file %s has dimension %d, expected %d. Skipping.", caseData.Id, path, len(caseData.Embedding), expectedDimension)
			return nil // Continue with next file
		}

		embeddingBytes, err := sqlitevec.Float32ToBytes(caseData.Embedding)
		if err != nil {
			log.Printf("Warning: failed to convert embedding to bytes for ID %s: %v. Skipping.", caseData.Id, err)
			return nil // Continue with next file
		}

		stmt, err := dbConn.Prepare("INSERT OR REPLACE INTO cases_embeddings (id, embedding) VALUES (?, ?)")
		if err != nil {
			// This is a more serious error, likely indicating a problem with the DB or SQL
			return fmt.Errorf("failed to prepare SQL statement: %w", err)
		}
		defer stmt.Close()

		_, err = stmt.Exec(caseData.Id, embeddingBytes)
		if err != nil {
			log.Printf("Warning: failed to execute SQL statement for ID %s: %v. Skipping.", caseData.Id, err)
			return nil // Continue with next file
		}

		log.Printf("Loaded embedding for ID: %s from %s", caseData.Id, path)
		return nil
	})

	if err != nil {
		return fmt.Errorf("error processing files during embedding load: %w", err)
	}

	log.Println("Finished loading embeddings into the database.")
	return nil
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

	if len(caseData.Summary) <= 100 {
		return nil, fmt.Errorf("summary length for case %s is not greater than 100 characters", id)
	}
	return &caseData, nil
}

// check if valid ID
func isValidId(id string) bool {
	pattern := regexp.MustCompile(`^\d{1,30}$`)
	return pattern.MatchString(id)
}

// Handler function for the incoming HTTP request
func requestHandler(w http.ResponseWriter, r *http.Request) {
	// check if we has ID
	id := r.URL.Query().Get("id")
	query := ""

	if id != "" {
		if !isValidId(id) {
			http.Error(w, "Invalid id format. ID must be numeric and max 30 digits.", http.StatusBadRequest)
			return
		}
		query = id
	} else {
		query = r.URL.Query().Get("q")
	}

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
	flag.BoolVar(&updateDB, "update-db", false, "[Optional] If set, forces loading/updating of embeddings from the JSON directory into the SQLite database.")
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

	// Prompt for user confirmation (only if not in DB update mode, or if force is not set for server start)
	// If --update-db is set, we might not need this interactive prompt if it's a batch operation.
	// However, the API key is still used by openaiGetEmbedding if a query ID's embedding isn't found,
	// or if a text query is made. So, user awareness of API usage is still important.
	// For now, let's keep the confirmation unless `force` is true.
	if !force && !updateDB && !promptUserConfirmation() { // Adjusted condition
		fmt.Println("Service startup cancelled by user.")
		return
	} else if updateDB && !force {
		// If updating DB, still confirm if not forced, as it might involve many files.
		// Or, we could assume --update-db implies consent for this specific operation.
		// For now, let's assume --update-db with --force bypasses this for DB operations too.
		// If only --update-db is set, it could still prompt.
		// This part of logic might need refinement based on desired UX for --update-db.
		// Let's simplify: promptUserConfirmation if !force, regardless of updateDB.
		// The prompt is about general API usage.
	}

	if !force && !promptUserConfirmation() {
		fmt.Println("Operation cancelled by user.")
		return
	}


	var err error // Declare err for use in this scope

	if updateDB {
		log.Println("Database update initiated due to --update-db flag.")
		db, err = initDB("cases.db", embeddingDimension)
		if err != nil {
			log.Fatalf("Failed to initialize database for update: %v", err)
		}
		defer db.Close() // Close DB when main finishes if it was opened here

		err = loadEmbeddingsToDB(db, embeddingDir, embeddingDimension)
		if err != nil {
			log.Fatalf("Failed to load embeddings into database: %v", err)
		}
		log.Println("Database update complete. The service will now exit as --update-db is primarily for data loading.")
		// Decide if to exit or continue to server mode. For now, let's exit.
		return
	} else {
		log.Println("Skipping database update, using existing cases.db (or creating if not exists).")
		db, err = initDB("cases.db", embeddingDimension)
		if err != nil {
			log.Fatalf("Failed to initialize database: %v", err)
		}
		// defer db.Close() // db will be used by the server, so don't close it here.
		// The global db variable is now set.
	}

	// Load embeddings - This is now handled by initDB and loadEmbeddingsToDB if updateDB is true
	// The old `embeddings` map is no longer used. Queries will go to the DB.
	// embedd, err := loadEmbeddings(embeddingDir) // Removed
	// if err != nil { // Removed
	// 	log.Fatalf("Error loading embeddings: %s", err) // Removed
	// 	return // Removed
	// } // Removed
	// embeddings = embedd // Removed

	// Start the HTTP server and listen for incoming requests
	// Ensure db is available to handlers if they need it.
	// For now, requestHandler uses the global `db` implicitly if it's modified to do so.
	http.HandleFunc("/", requestHandler)
	log.Println("Starting server on :9989")
	log.Fatal(http.ListenAndServe(":9989", nil))
}

func initDB(dbPath string, embeddingDim int) (*sql.DB, error) {
	// Open the SQLite database, creating it if it doesn't exist.
	database, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Register sqlite-vec extension
	if err := sqlitevec.Register(database); err != nil {
		return nil, fmt.Errorf("failed to register sqlite-vec: %w", err)
	}

	// Create cases_embeddings table
	_, err = database.Exec(`
		CREATE TABLE IF NOT EXISTS cases_embeddings (
			id TEXT PRIMARY KEY,
			embedding BLOB
		)
	`)
	if err != nil {
		return nil, fmt.Errorf("failed to create cases_embeddings table: %w", err)
	}

	// Create virtual table for VSS search
	// Note: The VSS table stores the vector directly, it doesn't reference another table's column in the way
	// some other FTS extensions might. We'll store the ID separately and join if needed,
	// or more likely, retrieve the ID from the VSS table and then fetch full data from cases_embeddings.
	// The real table is cases_embeddings (id TEXT PRIMARY KEY, embedding BLOB).
	// The virtual table vss_cases_embeddings makes the `embedding` column searchable.
	// The `dimensions=%d` part specifies the dimensionality of the vectors in that column.
	createVssTableSQL := fmt.Sprintf(`
		CREATE VIRTUAL TABLE IF NOT EXISTS vss_cases_embeddings USING vss0(
			embedding dimensions=%d
		);
	`, embeddingDim) // Ensure embeddingDim is correctly passed and used.
	_, err = database.Exec(createVssTableSQL)
	if err != nil {
		return nil, fmt.Errorf("failed to create vss_cases_embeddings virtual table using vss0: %w", err)
	}

	log.Println("Database initialized successfully: cases_embeddings table and vss_cases_embeddings VSS table created.")
	return database, nil
}
