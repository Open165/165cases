package main

import (
	"os"
	"testing"
)

func TestInitDB(t *testing.T) {
	const testDBPath = "test_cases.db"
	const testEmbeddingDimension = 3072

	// Ensure the test database file is cleaned up after the test.
	defer os.Remove(testDBPath)

	// Call initDB.
	// Note: initDB returns a new db connection, it doesn't set the global `db` variable directly.
	// The global `db` is set in the main function.
	testDB, err := initDB(testDBPath, testEmbeddingDimension)

	// Check if the error returned is nil.
	if err != nil {
		t.Fatalf("initDB() error = %v, want nil", err)
	}

	// Check if the db object returned is not nil.
	if testDB == nil {
		t.Fatal("initDB() returned a nil db object, want non-nil")
	}
	defer testDB.Close() // Ensure the database is closed

	// Try to ping the database to ensure the connection is live.
	if err := testDB.Ping(); err != nil {
		t.Fatalf("db.Ping() error = %v, want nil", err)
	}

	// Optional: Verify table existence
	tablesToVerify := []string{"cases_embeddings", "vss_cases_embeddings"}
	for _, tableName := range tablesToVerify {
		var name string
		query := "SELECT name FROM sqlite_master WHERE type='table' AND name=?;"
		err := testDB.QueryRow(query, tableName).Scan(&name)
		if err != nil {
			t.Fatalf("QueryRow for table %s failed: %v", tableName, err)
		}
		if name != tableName {
			t.Fatalf("Table %s not found in sqlite_master, got name '%s'", tableName, name)
		}
	}

	t.Log("initDB test completed successfully.")
}
