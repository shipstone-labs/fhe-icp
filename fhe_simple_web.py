#!/usr/bin/env python3
"""Simple web interface for FHE document comparison."""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import mimetypes
import hashlib
from datetime import datetime

from key_management import FHEKeyManager
from encrypted_storage import EncryptedDocumentStore
from batch_operations import BatchProcessor, BatchConfig


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🔐 FHE Document Comparison</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .tabs {
            display: flex;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 30px;
        }
        .tab {
            flex: 1;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 16px;
            transition: all 0.3s;
        }
        .tab:hover {
            background: #f0f0f0;
        }
        .tab.active {
            color: #2196F3;
            border-bottom: 3px solid #2196F3;
            margin-bottom: -2px;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            resize: vertical;
            min-height: 120px;
            font-family: inherit;
        }
        textarea:focus {
            outline: none;
            border-color: #2196F3;
        }
        button {
            background: #2196F3;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover {
            background: #1976D2;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .document-list {
            margin: 20px 0;
        }
        .document-item {
            background: #f8f8f8;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .document-item:hover {
            background: #e8e8e8;
            transform: translateX(5px);
        }
        .document-item.selected {
            background: #E3F2FD;
            border: 2px solid #2196F3;
        }
        .document-item input[type="radio"] {
            width: 20px;
            height: 20px;
        }
        .document-info {
            flex: 1;
        }
        .document-id {
            font-weight: bold;
            color: #333;
        }
        .document-preview {
            color: #666;
            font-size: 14px;
            margin-top: 5px;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .result.success {
            background: #E8F5E9;
            border: 2px solid #4CAF50;
        }
        .result.info {
            background: #E3F2FD;
            border: 2px solid #2196F3;
        }
        .result.error {
            background: #FFEBEE;
            border: 2px solid #F44336;
        }
        .similarity-score {
            font-size: 48px;
            font-weight: bold;
            margin: 20px 0;
        }
        .similarity-high { color: #4CAF50; }
        .similarity-medium { color: #FF9800; }
        .similarity-low { color: #F44336; }
        .loading {
            text-align: center;
            color: #666;
            margin: 20px 0;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #2196F3;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .quick-compare {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        @media (max-width: 600px) {
            .quick-compare {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔐 FHE Document Comparison</h1>
        <p class="subtitle">Compare documents while keeping them encrypted</p>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('add')">📝 Add Document</button>
            <button class="tab" onclick="showTab('compare')">🔍 Compare</button>
            <button class="tab" onclick="showTab('quick')">⚡ Quick Compare</button>
        </div>
        
        <div id="add-tab" class="tab-content active">
            <h2>Add a New Document</h2>
            <p>Type or paste your document content below:</p>
            <textarea id="new-document" placeholder="Enter your document text here..."></textarea>
            <br><br>
            <button onclick="addDocument()">🔒 Encrypt & Save</button>
            <div id="add-result"></div>
        </div>
        
        <div id="compare-tab" class="tab-content">
            <h2>Compare Two Documents</h2>
            <p>Select two documents to compare:</p>
            
            <div id="document-list" class="document-list">
                <div class="loading">Loading documents...</div>
            </div>
            
            <br>
            <button onclick="compareDocuments()" id="compare-button" disabled>🔍 Compare Selected</button>
            <div id="compare-result"></div>
        </div>
        
        <div id="quick-tab" class="tab-content">
            <h2>Quick Compare</h2>
            <p>Compare two texts without saving them:</p>
            
            <div class="quick-compare">
                <div>
                    <h3>Text 1</h3>
                    <textarea id="quick-text-1" placeholder="Enter first text..."></textarea>
                </div>
                <div>
                    <h3>Text 2</h3>
                    <textarea id="quick-text-2" placeholder="Enter second text..."></textarea>
                </div>
            </div>
            
            <button onclick="quickCompare()">⚡ Compare</button>
            <div id="quick-result"></div>
        </div>
    </div>
    
    <script>
        let selectedDocs = [];
        
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
            
            // Load documents if compare tab
            if (tabName === 'compare') {
                loadDocuments();
            }
        }
        
        async function addDocument() {
            const text = document.getElementById('new-document').value.trim();
            if (!text) {
                showResult('add-result', 'Please enter some text!', 'error');
                return;
            }
            
            showLoading('add-result');
            
            try {
                const response = await fetch('/api/add', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text})
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showResult('add-result', 
                        `✅ Document encrypted and saved!<br>ID: <strong>${result.doc_id}</strong>`, 
                        'success');
                    document.getElementById('new-document').value = '';
                } else {
                    showResult('add-result', `❌ Error: ${result.error}`, 'error');
                }
            } catch (e) {
                showResult('add-result', `❌ Error: ${e.message}`, 'error');
            }
        }
        
        async function loadDocuments() {
            showLoading('document-list');
            
            try {
                const response = await fetch('/api/documents');
                const result = await response.json();
                
                if (result.success && result.documents.length > 0) {
                    let html = '<h3>Select exactly 2 documents:</h3>';
                    
                    result.documents.forEach((doc, index) => {
                        html += `
                            <div class="document-item" onclick="selectDocument('${doc.doc_id}', this)">
                                <input type="checkbox" id="doc-${index}" value="${doc.doc_id}">
                                <div class="document-info">
                                    <div class="document-id">${doc.doc_id}</div>
                                    <div class="document-preview">${doc.preview || 'No preview available'}</div>
                                </div>
                            </div>
                        `;
                    });
                    
                    document.getElementById('document-list').innerHTML = html;
                } else {
                    document.getElementById('document-list').innerHTML = 
                        '<p style="text-align: center; color: #666;">No documents yet. Add some documents first!</p>';
                }
            } catch (e) {
                document.getElementById('document-list').innerHTML = 
                    '<p style="color: red;">Error loading documents: ' + e.message + '</p>';
            }
        }
        
        function selectDocument(docId, element) {
            const checkbox = element.querySelector('input[type="checkbox"]');
            
            if (checkbox.checked) {
                // Unselect
                checkbox.checked = false;
                element.classList.remove('selected');
                selectedDocs = selectedDocs.filter(id => id !== docId);
            } else if (selectedDocs.length < 2) {
                // Select
                checkbox.checked = true;
                element.classList.add('selected');
                selectedDocs.push(docId);
            } else {
                alert('Please select exactly 2 documents');
            }
            
            // Enable/disable compare button
            document.getElementById('compare-button').disabled = selectedDocs.length !== 2;
        }
        
        async function compareDocuments() {
            if (selectedDocs.length !== 2) {
                showResult('compare-result', 'Please select exactly 2 documents', 'error');
                return;
            }
            
            showLoading('compare-result');
            
            try {
                const response = await fetch('/api/compare', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        doc1: selectedDocs[0],
                        doc2: selectedDocs[1]
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    const score = result.similarity;
                    const percent = (score * 100).toFixed(1);
                    let colorClass = 'similarity-low';
                    let interpretation = 'Very different';
                    
                    if (score > 0.8) {
                        colorClass = 'similarity-high';
                        interpretation = 'Very similar!';
                    } else if (score > 0.6) {
                        colorClass = 'similarity-medium';
                        interpretation = 'Moderately similar';
                    }
                    
                    showResult('compare-result', 
                        `<div class="similarity-score ${colorClass}">${percent}%</div>
                         <p><strong>${interpretation}</strong></p>
                         <p>Comparing: ${selectedDocs[0]} ↔ ${selectedDocs[1]}</p>`, 
                        'info');
                } else {
                    showResult('compare-result', `❌ Error: ${result.error}`, 'error');
                }
            } catch (e) {
                showResult('compare-result', `❌ Error: ${e.message}`, 'error');
            }
        }
        
        async function quickCompare() {
            const text1 = document.getElementById('quick-text-1').value.trim();
            const text2 = document.getElementById('quick-text-2').value.trim();
            
            if (!text1 || !text2) {
                showResult('quick-result', 'Please enter both texts!', 'error');
                return;
            }
            
            showLoading('quick-result');
            
            try {
                const response = await fetch('/api/quick-compare', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text1: text1, text2: text2})
                });
                
                const result = await response.json();
                
                if (result.success) {
                    const score = result.similarity;
                    const percent = (score * 100).toFixed(1);
                    let colorClass = 'similarity-low';
                    let interpretation = 'These texts are very different';
                    
                    if (score > 0.8) {
                        colorClass = 'similarity-high';
                        interpretation = 'These texts are very similar!';
                    } else if (score > 0.6) {
                        colorClass = 'similarity-medium';
                        interpretation = 'These texts are moderately similar';
                    }
                    
                    showResult('quick-result', 
                        `<div class="similarity-score ${colorClass}">${percent}%</div>
                         <p><strong>${interpretation}</strong></p>`, 
                        'info');
                } else {
                    showResult('quick-result', `❌ Error: ${result.error}`, 'error');
                }
            } catch (e) {
                showResult('quick-result', `❌ Error: ${e.message}`, 'error');
            }
        }
        
        function showLoading(elementId) {
            document.getElementById(elementId).innerHTML = 
                '<div class="loading"><div class="spinner"></div>Processing...</div>';
        }
        
        function showResult(elementId, message, type) {
            document.getElementById(elementId).innerHTML = 
                `<div class="result ${type}">${message}</div>`;
        }
        
        // Initialize
        window.onload = function() {
            // Check if system is ready
            fetch('/api/status').then(r => r.json()).then(result => {
                if (!result.ready) {
                    alert('System is initializing. Please wait a moment and refresh the page.');
                }
            });
        };
    </script>
</body>
</html>
"""


class FHEWebHandler(BaseHTTPRequestHandler):
    """Simple web server for FHE operations."""
    
    # Class-level storage
    key_manager = None
    storage = None
    processor = None
    
    @classmethod
    def initialize(cls):
        """Initialize the FHE system."""
        if cls.key_manager is None:
            cls.key_manager = FHEKeyManager()
            cls.storage = EncryptedDocumentStore()
            
            # Check if keys exist
            if cls.key_manager.get_current_key() is None:
                print("Generating FHE keys (this takes 30-60 seconds)...")
                cls.key_manager.generate_keys()
                
            cls.processor = BatchProcessor(
                key_manager=cls.key_manager,
                storage=cls.storage,
                config=BatchConfig(show_progress=False)
            )
            
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
            
        elif self.path == '/api/status':
            self.send_json({'ready': self.processor is not None})
            
        elif self.path == '/api/documents':
            docs = self.storage.list_documents()
            formatted_docs = []
            
            for doc in docs:
                formatted_docs.append({
                    'doc_id': doc['doc_id'],
                    'preview': doc.get('metadata', {}).get('preview', 'No preview'),
                    'timestamp': doc.get('timestamp', 'Unknown')
                })
                
            self.send_json({
                'success': True,
                'documents': formatted_docs
            })
            
        else:
            self.send_error(404)
            
    def do_POST(self):
        """Handle POST requests."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            
            if self.path == '/api/add':
                text = data.get('text', '').strip()
                if not text:
                    self.send_json({'success': False, 'error': 'No text provided'})
                    return
                    
                # Create document ID
                words = text.split()[:3]
                doc_id = '_'.join(word.lower() for word in words if word.isalnum())[:20]
                doc_id = f"{doc_id}_{datetime.now().strftime('%H%M%S')}"
                
                # Encrypt
                self.processor.encrypt_documents(
                    [text],
                    doc_ids=[doc_id],
                    metadata=[{'preview': text[:50] + '...' if len(text) > 50 else text}]
                )
                
                self.send_json({
                    'success': True,
                    'doc_id': doc_id
                })
                
            elif self.path == '/api/compare':
                doc1 = data.get('doc1')
                doc2 = data.get('doc2')
                
                if not doc1 or not doc2:
                    self.send_json({'success': False, 'error': 'Two documents required'})
                    return
                    
                similarity = self.processor.compare_encrypted(doc1, doc2)
                
                self.send_json({
                    'success': True,
                    'similarity': float(similarity)
                })
                
            elif self.path == '/api/quick-compare':
                text1 = data.get('text1', '').strip()
                text2 = data.get('text2', '').strip()
                
                if not text1 or not text2:
                    self.send_json({'success': False, 'error': 'Two texts required'})
                    return
                    
                # Create temporary documents
                temp_id1 = f"temp_{hashlib.md5(text1.encode()).hexdigest()[:8]}"
                temp_id2 = f"temp_{hashlib.md5(text2.encode()).hexdigest()[:8]}"
                
                # Encrypt temporarily
                self.processor.encrypt_documents([text1, text2], doc_ids=[temp_id1, temp_id2])
                
                # Compare
                similarity = self.processor.compare_encrypted(temp_id1, temp_id2)
                
                # Clean up
                self.storage.delete(temp_id1)
                self.storage.delete(temp_id2)
                
                self.send_json({
                    'success': True,
                    'similarity': float(similarity)
                })
                
            else:
                self.send_error(404)
                
        except Exception as e:
            self.send_json({
                'success': False,
                'error': str(e)
            })
            
    def send_json(self, data):
        """Send JSON response."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
        
    def log_message(self, format, *args):
        """Suppress request logging."""
        pass


def main():
    """Run the web server."""
    print("🔐 FHE Document Comparison Web Interface")
    print("="*50)
    
    # Initialize FHE system
    print("Initializing FHE system...")
    FHEWebHandler.initialize()
    
    # Start server
    port = 8080
    server = HTTPServer(('localhost', port), FHEWebHandler)
    
    print(f"\n✅ Server running at: http://localhost:{port}")
    print("\nOpen this URL in your browser to use the interface.")
    print("Press Ctrl+C to stop the server.\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        server.shutdown()
        

if __name__ == "__main__":
    main()