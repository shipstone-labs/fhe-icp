#!/usr/bin/env python3
"""Interactive conversational interface for FHE document comparison."""

import os
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import textwrap
import hashlib

from key_management import FHEKeyManager
from encrypted_storage import EncryptedDocumentStore
from batch_operations import BatchProcessor, BatchConfig


class FHEInteractiveChat:
    """Conversational interface for document encryption and comparison."""
    
    def __init__(self):
        self.key_manager = FHEKeyManager()
        self.storage = EncryptedDocumentStore()
        self.processor = None
        self.documents_cache = {}
        
        # Colors for terminal
        self.BLUE = '\033[94m'
        self.GREEN = '\033[92m'
        self.YELLOW = '\033[93m'
        self.RED = '\033[91m'
        self.ENDC = '\033[0m'
        self.BOLD = '\033[1m'
        
    def _print_colored(self, text: str, color: str = ""):
        """Print colored text."""
        print(f"{color}{text}{self.ENDC}")
        
    def _get_processor(self) -> BatchProcessor:
        """Get or create batch processor."""
        if self.processor is None:
            self.processor = BatchProcessor(
                key_manager=self.key_manager,
                storage=self.storage,
                config=BatchConfig(show_progress=False)
            )
        return self.processor
        
    def _ensure_keys(self) -> bool:
        """Ensure FHE keys are available."""
        if self.key_manager.get_current_key() is None:
            self._print_colored("\n🔐 No encryption keys found. Let's set that up first!", self.YELLOW)
            print("\nI'll generate secure encryption keys for you.")
            print("This is a one-time setup that takes about 30-60 seconds.\n")
            
            response = input("Ready to create your keys? (yes/no): ").lower()
            if response in ['yes', 'y']:
                try:
                    print("\nGenerating keys... (this takes a moment)")
                    self.key_manager.generate_keys()
                    self._print_colored("✅ Keys created successfully!", self.GREEN)
                    return True
                except Exception as e:
                    self._print_colored(f"❌ Error creating keys: {e}", self.RED)
                    return False
            else:
                return False
        return True
        
    def _show_documents(self) -> List[Dict]:
        """Show available documents in a friendly way."""
        docs = self.storage.list_documents()
        
        if not docs:
            self._print_colored("\n📭 No documents stored yet.", self.YELLOW)
            return []
            
        self._print_colored("\n📚 Your encrypted documents:", self.BLUE)
        print("-" * 50)
        
        for i, doc in enumerate(docs, 1):
            doc_id = doc['doc_id']
            timestamp = doc.get('timestamp', 'Unknown time')
            
            # Try to get a preview from metadata
            preview = "No preview available"
            if 'metadata' in doc and 'preview' in doc['metadata']:
                preview = doc['metadata']['preview']
            elif 'metadata' in doc and 'original_text' in doc['metadata']:
                # Store first 50 chars as preview
                preview = doc['metadata']['original_text'][:50] + "..."
                
            print(f"\n{i}. {self.BOLD}{doc_id}{self.ENDC}")
            print(f"   📝 {preview}")
            print(f"   🕐 Created: {timestamp}")
            
        print("-" * 50)
        return docs
        
    def _select_document(self, prompt: str = "Select a document") -> Optional[str]:
        """Let user select a document interactively."""
        docs = self._show_documents()
        
        if not docs:
            return None
            
        while True:
            try:
                choice = input(f"\n{prompt} (number or 'cancel'): ")
                
                if choice.lower() == 'cancel':
                    return None
                    
                idx = int(choice) - 1
                if 0 <= idx < len(docs):
                    return docs[idx]['doc_id']
                else:
                    print("Please enter a valid number.")
            except ValueError:
                print("Please enter a number or 'cancel'.")
                
    def add_document(self):
        """Add a new document conversationally."""
        self._print_colored("\n📝 Let's add a new document!", self.BLUE)
        
        print("\nYou can type or paste your text below.")
        print("(For multiple lines, end with a line containing just 'END')\n")
        
        lines = []
        if sys.stdin.isatty():
            # Interactive mode
            while True:
                line = input()
                if line.strip() == 'END':
                    break
                lines.append(line)
        else:
            # Handle piped input
            lines = sys.stdin.readlines()
            
        text = '\n'.join(lines).strip()
        
        if not text:
            self._print_colored("No text entered.", self.YELLOW)
            return
            
        # Create a friendly document ID
        words = text.split()[:3]
        doc_id = '_'.join(word.lower() for word in words if word.isalnum())[:20]
        doc_id = f"{doc_id}_{datetime.now().strftime('%H%M%S')}"
        
        print(f"\n💾 Saving as: {doc_id}")
        
        # Show preview
        preview = text[:100] + "..." if len(text) > 100 else text
        print(f"📄 Preview: {preview}")
        
        # Encrypt and save
        try:
            processor = self._get_processor()
            processor.encrypt_documents(
                [text],
                doc_ids=[doc_id],
                metadata=[{
                    'preview': preview[:50] + "...",
                    'original_text': text  # In production, you might not want to store this
                }]
            )
            
            self._print_colored(f"\n✅ Document encrypted and saved!", self.GREEN)
            self._print_colored(f"ID: {doc_id}", self.GREEN)
            
        except Exception as e:
            self._print_colored(f"❌ Error: {e}", self.RED)
            
    def compare_documents(self):
        """Compare two documents conversationally."""
        self._print_colored("\n🔍 Let's compare two documents!", self.BLUE)
        
        # Select first document
        doc1 = self._select_document("Select the FIRST document")
        if not doc1:
            return
            
        # Select second document
        print()  # Add spacing
        doc2 = self._select_document("Select the SECOND document")
        if not doc2:
            return
            
        if doc1 == doc2:
            self._print_colored("\n🤔 That's the same document! Try selecting two different ones.", self.YELLOW)
            return
            
        # Compare
        print(f"\n⏳ Comparing '{doc1}' with '{doc2}'...")
        
        try:
            processor = self._get_processor()
            similarity = processor.compare_encrypted(doc1, doc2)
            
            # Interpret the score
            self._print_colored(f"\n📊 Similarity Score: {similarity:.2%}", self.BOLD)
            
            if similarity > 0.8:
                self._print_colored("These documents are very similar! 🎯", self.GREEN)
                print("They likely discuss the same topic or have similar content.")
            elif similarity > 0.6:
                self._print_colored("These documents are moderately similar. 🤝", self.BLUE)
                print("They share some common themes or concepts.")
            elif similarity > 0.4:
                self._print_colored("These documents have some similarity. 🔗", self.YELLOW)
                print("They might touch on related topics.")
            else:
                self._print_colored("These documents are quite different. 🔀", self.RED)
                print("They appear to discuss different topics.")
                
        except Exception as e:
            self._print_colored(f"❌ Error comparing documents: {e}", self.RED)
            
    def search_similar(self):
        """Search for similar documents conversationally."""
        self._print_colored("\n🔎 Search for similar documents!", self.BLUE)
        
        print("\nDescribe what you're looking for:")
        query = input("Search query: ").strip()
        
        if not query:
            return
            
        print(f"\n⏳ Searching for documents similar to: '{query}'...")
        
        try:
            processor = self._get_processor()
            results = processor.search_similar(query, top_k=5, min_similarity=0.3)
            
            if not results:
                self._print_colored("\n😕 No similar documents found.", self.YELLOW)
                print("Try different keywords or add more documents.")
                return
                
            self._print_colored(f"\n📋 Found {len(results)} similar documents:", self.GREEN)
            print("-" * 50)
            
            for i, (doc_id, score) in enumerate(results, 1):
                # Get document info
                doc_info = self.storage.index.get(doc_id, {})
                preview = "No preview"
                if 'metadata' in doc_info and 'preview' in doc_info['metadata']:
                    preview = doc_info['metadata']['preview']
                    
                print(f"\n{i}. {self.BOLD}{doc_id}{self.ENDC} ({score:.2%} match)")
                print(f"   📝 {preview}")
                
        except Exception as e:
            self._print_colored(f"❌ Error searching: {e}", self.RED)
            
    def quick_compare(self):
        """Quick compare two pieces of text without saving."""
        self._print_colored("\n⚡ Quick comparison (without saving)!", self.BLUE)
        
        print("\nEnter the FIRST text (end with 'END' on a new line):")
        lines1 = []
        while True:
            line = input()
            if line.strip() == 'END':
                break
            lines1.append(line)
        text1 = '\n'.join(lines1)
        
        print("\nEnter the SECOND text (end with 'END' on a new line):")
        lines2 = []
        while True:
            line = input()
            if line.strip() == 'END':
                break
            lines2.append(line)
        text2 = '\n'.join(lines2)
        
        if not text1 or not text2:
            self._print_colored("Both texts are required.", self.YELLOW)
            return
            
        print("\n⏳ Comparing texts...")
        
        try:
            # Create temporary documents
            processor = self._get_processor()
            temp_id1 = f"temp_{hashlib.md5(text1.encode()).hexdigest()[:8]}"
            temp_id2 = f"temp_{hashlib.md5(text2.encode()).hexdigest()[:8]}"
            
            # Encrypt temporarily
            processor.encrypt_documents([text1, text2], doc_ids=[temp_id1, temp_id2])
            
            # Compare
            similarity = processor.compare_encrypted(temp_id1, temp_id2)
            
            # Show results
            self._print_colored(f"\n📊 Similarity Score: {similarity:.2%}", self.BOLD)
            
            if similarity > 0.8:
                self._print_colored("These texts are very similar! 🎯", self.GREEN)
            elif similarity > 0.6:
                self._print_colored("These texts are moderately similar. 🤝", self.BLUE)
            elif similarity > 0.4:
                self._print_colored("These texts have some similarity. 🔗", self.YELLOW)
            else:
                self._print_colored("These texts are quite different. 🔀", self.RED)
                
            # Clean up temporary documents
            self.storage.delete(temp_id1)
            self.storage.delete(temp_id2)
            
        except Exception as e:
            self._print_colored(f"❌ Error: {e}", self.RED)
            
    def show_stats(self):
        """Show system statistics in a friendly way."""
        self._print_colored("\n📊 System Statistics", self.BLUE)
        print("-" * 50)
        
        # Storage stats
        stats = self.storage.get_stats()
        
        print(f"\n📚 Documents: {stats['total_documents']}")
        if stats['total_documents'] > 0:
            print(f"💾 Total size: {stats['total_size_mb']:.2f} MB")
            print(f"📏 Average size: {stats['average_size_bytes']:.0f} bytes per document")
            
        # Key stats
        keys = self.key_manager.list_keys()
        print(f"\n🔐 Encryption keys: {len(keys)}")
        if keys:
            current = self.key_manager.get_current_key()
            print(f"🔑 Active key: {current}")
            
    def run(self):
        """Run the interactive chat interface."""
        self._print_colored("\n🔐 Welcome to FHE Document Comparison!", self.BOLD)
        print("Your documents are encrypted for privacy using advanced cryptography.\n")
        
        # Ensure keys are available
        if not self._ensure_keys():
            print("\nCome back when you're ready to set up encryption keys!")
            return
            
        # Main menu loop
        while True:
            print("\n" + "="*50)
            self._print_colored("What would you like to do?", self.BLUE)
            print("\n1. 📝 Add a document")
            print("2. 🔍 Compare two documents")
            print("3. 🔎 Search for similar documents")
            print("4. ⚡ Quick compare (without saving)")
            print("5. 📊 View statistics")
            print("6. 👋 Exit\n")
            
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == '1':
                self.add_document()
            elif choice == '2':
                self.compare_documents()
            elif choice == '3':
                self.search_similar()
            elif choice == '4':
                self.quick_compare()
            elif choice == '5':
                self.show_stats()
            elif choice == '6':
                self._print_colored("\n👋 Thanks for using FHE Document Comparison!", self.GREEN)
                break
            else:
                self._print_colored("Please enter a number from 1-6.", self.YELLOW)


def main():
    """Main entry point."""
    chat = FHEInteractiveChat()
    
    try:
        chat.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()