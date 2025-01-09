Make sure your env OPENAI_API_KEY available. 
You can also editing .env, put OPENAI_API_KEY inside. eg.
```
echo "export OPENAI_API_KEY=sk-proj-tLl...PcA" > .env
```

Usage:
```
./165embedding.sh test3.json
```

When the file exists will be skipped. Delete file to re-fetch the embedding
