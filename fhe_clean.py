#!/usr/bin/env python3
"""Clean CLI interface for FHE document operations - minimal output."""

import argparse
import json
import sys
import os
import logging
from pathlib import Path
from typing import Optional, List

# Suppress all logging
logging.disable(logging.CRITICAL)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Suppress urllib3 warnings
import urllib3
urllib3.disable_warnings()

# Now import our modules
from key_management import FHEKeyManager
from encrypted_storage import EncryptedDocumentStore
from batch_operations import BatchProcessor, BatchConfig


class CleanFHECLI:
    """Clean CLI interface with minimal output."""
    
    def __init__(self):
        """Initialize with all output suppressed."""
        # Redirect stderr to devnull during initialization
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
        try:
            self.key_manager = FHEKeyManager()
            self.storage = EncryptedDocumentStore()
            self.processor = None
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr
            
    def _get_processor(self) -> BatchProcessor:
        """Get or create batch processor silently."""
        if self.processor is None:
            # Suppress output during processor creation
            old_stderr = sys.stderr
            old_stdout = sys.stdout
            sys.stderr = open(os.devnull, 'w')
            sys.stdout = open(os.devnull, 'w')
            
            try:
                self.processor = BatchProcessor(
                    key_manager=self.key_manager,
                    storage=self.storage,
                    config=BatchConfig(show_progress=False)
                )
            finally:
                sys.stderr.close()
                sys.stdout.close()
                sys.stderr = old_stderr
                sys.stdout = old_stdout
                
        return self.processor
        
    def cmd_init(self, args):
        """Initialize system (generate keys)."""
        if self.key_manager.get_current_key():
            print("✓ System already initialized")
            return
            
        print("Initializing... (30-60 seconds)")
        
        # Suppress output during key generation
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        sys.stderr = open(os.devnull, 'w')
        sys.stdout = open(os.devnull, 'w')
        
        try:
            # Mock password for simplicity
            import key_management
            key_management.getpass.getpass = lambda prompt: "default"
            
            self.key_manager.generate_keys()
            print("✓ Ready")
        except Exception as e:
            print(f"✗ Failed: {e}")
        finally:
            sys.stderr.close()
            sys.stdout.close()
            sys.stderr = old_stderr
            sys.stdout = old_stdout
            
    def cmd_add(self, args):
        """Add a document."""
        processor = self._get_processor()
        
        if processor.fhe_model is None:
            print("✗ Not initialized. Run: python fhe_clean.py init")
            return
            
        # Read text
        if args.file:
            with open(args.file, 'r') as f:
                text = f.read()
        else:
            text = args.text
            
        if not text:
            print("✗ No text provided")
            return
            
        # Create simple ID
        doc_id = args.id if args.id else f"doc_{len(self.storage.list_documents())+1}"
        
        # Suppress output during encryption
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        sys.stderr = open(os.devnull, 'w')
        sys.stdout = open(os.devnull, 'w')
        
        try:
            processor.encrypt_documents(
                [text],
                doc_ids=[doc_id],
                metadata=[{'preview': text[:50]}]
            )
            print(f"✓ {doc_id}")
        except Exception as e:
            print(f"✗ {e}")
        finally:
            sys.stderr.close()
            sys.stdout.close()
            sys.stderr = old_stderr
            sys.stdout = old_stdout
            
    def cmd_compare(self, args):
        """Compare two documents."""
        processor = self._get_processor()
        
        if processor.fhe_model is None:
            print("✗ Not initialized. Run: python fhe_clean.py init")
            return
            
        # Suppress output
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        sys.stderr = open(os.devnull, 'w')
        sys.stdout = open(os.devnull, 'w')
        
        try:
            similarity = processor.compare_encrypted(args.doc1, args.doc2)
            
            # Restore output
            sys.stderr.close()
            sys.stdout.close()
            sys.stderr = old_stderr
            sys.stdout = old_stdout
            
            # Simple output
            percent = int(similarity * 100)
            if percent >= 80:
                desc = "very similar"
            elif percent >= 60:
                desc = "similar"
            elif percent >= 40:
                desc = "somewhat similar"
            else:
                desc = "different"
                
            print(f"{percent}% - {desc}")
            
        except Exception as e:
            sys.stderr.close()
            sys.stdout.close()
            sys.stderr = old_stderr
            sys.stdout = old_stdout
            print(f"✗ {e}")
            
    def cmd_list(self, args):
        """List documents."""
        docs = self.storage.list_documents()
        
        if not docs:
            print("No documents")
            return
            
        for doc in docs:
            preview = doc.get('metadata', {}).get('preview', '')
            if preview:
                print(f"{doc['doc_id']}: {preview}")
            else:
                print(doc['doc_id'])
                
    def cmd_search(self, args):
        """Search for similar documents."""
        processor = self._get_processor()
        
        if processor.fhe_model is None:
            print("✗ Not initialized. Run: python fhe_clean.py init")
            return
            
        # Suppress output
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        sys.stderr = open(os.devnull, 'w')
        sys.stdout = open(os.devnull, 'w')
        
        try:
            results = processor.search_similar(args.query, top_k=args.top or 3)
            
            # Restore output
            sys.stderr.close()
            sys.stdout.close()
            sys.stderr = old_stderr
            sys.stdout = old_stdout
            
            if not results:
                print("No matches")
                return
                
            for doc_id, score in results:
                percent = int(score * 100)
                print(f"{doc_id}: {percent}%")
                
        except Exception as e:
            sys.stderr.close()
            sys.stdout.close()
            sys.stderr = old_stderr
            sys.stdout = old_stdout
            print(f"✗ {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="FHE Document Comparison (Clean)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fhe_clean.py init                    # Initialize (first time only)
  python fhe_clean.py add "Your text"         # Add document
  python fhe_clean.py add -f document.txt     # Add from file
  python fhe_clean.py list                    # List documents
  python fhe_clean.py compare doc1 doc2       # Compare documents
  python fhe_clean.py search "query text"     # Search documents
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Init
    subparsers.add_parser('init', help='Initialize system')
    
    # Add
    add_parser = subparsers.add_parser('add', help='Add document')
    add_parser.add_argument('text', nargs='?', help='Text to add')
    add_parser.add_argument('-f', '--file', help='Read from file')
    add_parser.add_argument('--id', help='Document ID')
    
    # List
    subparsers.add_parser('list', help='List documents')
    
    # Compare
    compare_parser = subparsers.add_parser('compare', help='Compare documents')
    compare_parser.add_argument('doc1', help='First document')
    compare_parser.add_argument('doc2', help='Second document')
    
    # Search
    search_parser = subparsers.add_parser('search', help='Search documents')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--top', type=int, help='Number of results')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    # Create CLI instance
    cli = CleanFHECLI()
    
    # Route commands
    command_map = {
        'init': cli.cmd_init,
        'add': cli.cmd_add,
        'list': cli.cmd_list,
        'compare': cli.cmd_compare,
        'search': cli.cmd_search
    }
    
    handler = command_map.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    # Suppress numpy warnings
    import numpy as np
    np.seterr(all='ignore')
    
    main()