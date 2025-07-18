#!/usr/bin/env python3
"""Production CLI interface for FHE document operations."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List
import logging

from key_management import FHEKeyManager
from encrypted_storage import EncryptedDocumentStore
from batch_operations import BatchProcessor, BatchConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FHEDocumentCLI:
    """Main CLI class for FHE document operations."""
    
    def __init__(self):
        """Initialize CLI components."""
        self.key_manager = FHEKeyManager()
        self.storage = EncryptedDocumentStore()
        self.processor = None
        
    def _get_processor(self) -> BatchProcessor:
        """Get or create batch processor."""
        if self.processor is None:
            self.processor = BatchProcessor(
                key_manager=self.key_manager,
                storage=self.storage,
                config=BatchConfig(show_progress=True)
            )
        return self.processor
        
    def cmd_keys(self, args):
        """Handle key management commands."""
        if args.action == 'generate':
            print("Generating new FHE keys...")
            key_info = self.key_manager.generate_keys(args.key_id)
            print(f"\nKeys generated successfully!")
            print(f"Key ID: {key_info['key_id']}")
            print(f"Created: {key_info['created']}")
            
        elif args.action == 'list':
            keys = self.key_manager.list_keys()
            current = self.key_manager.get_current_key()
            
            if not keys:
                print("No keys found. Generate keys with: fhe_cli keys generate")
                return
                
            print("Available keys:")
            for key_id, info in keys.items():
                marker = " (current)" if key_id == current else ""
                print(f"  - {key_id}{marker}")
                print(f"    Created: {info['created']}")
                print(f"    Size: {info['size_bytes'] / 1024 / 1024:.1f} MB")
                
        elif args.action == 'rotate':
            print("Rotating keys...")
            new_key = self.key_manager.rotate_keys(args.grace_days)
            print(f"New key generated: {new_key['key_id']}")
            print(f"Grace period: {args.grace_days} days")
            
    def cmd_encrypt(self, args):
        """Encrypt a single document."""
        processor = self._get_processor()
        
        if processor.fhe_model is None:
            print("Error: No compiled model found. Generate keys first:")
            print("  python fhe_cli.py keys generate")
            return
            
        # Read text
        if args.file:
            with open(args.file, 'r') as f:
                text = f.read()
        else:
            text = args.text
            
        # Prepare metadata
        metadata = {}
        if args.tags:
            metadata['tags'] = args.tags
        if args.metadata:
            metadata.update(json.loads(args.metadata))
            
        print(f"Encrypting document...")
        doc_ids = processor.encrypt_documents(
            [text],
            doc_ids=[args.id] if args.id else None,
            metadata=[metadata]
        )
        
        print(f"\nDocument encrypted successfully!")
        print(f"Document ID: {doc_ids[0]}")
        print(f"Size: {processor.storage.index[doc_ids[0]]['size_bytes']} bytes")
        
    def cmd_encrypt_batch(self, args):
        """Encrypt multiple documents."""
        processor = self._get_processor()
        
        if processor.fhe_model is None:
            print("Error: No compiled model found. Generate keys first:")
            print("  python fhe_cli.py keys generate")
            return
            
        # Read input file (JSON format expected)
        with open(args.input_file, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            print("Error: Input file must contain a JSON array of documents")
            return
            
        # Extract texts and metadata
        texts = []
        doc_ids = []
        metadata_list = []
        
        for item in data:
            if isinstance(item, str):
                texts.append(item)
                doc_ids.append(None)
                metadata_list.append({})
            elif isinstance(item, dict):
                texts.append(item.get('text', ''))
                doc_ids.append(item.get('id'))
                metadata_list.append(item.get('metadata', {}))
            else:
                print(f"Warning: Skipping invalid item: {item}")
                
        print(f"Encrypting {len(texts)} documents...")
        encrypted_ids = processor.encrypt_documents(texts, doc_ids, metadata_list)
        
        print(f"\nEncrypted {len(encrypted_ids)} documents successfully!")
        
        # Save results if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(encrypted_ids, f, indent=2)
            print(f"Document IDs saved to: {args.output_file}")
            
    def cmd_compare(self, args):
        """Compare two documents."""
        processor = self._get_processor()
        
        if processor.fhe_model is None:
            print("Error: No compiled model found. Generate keys first:")
            print("  python fhe_cli.py keys generate")
            return
            
        print(f"Comparing documents...")
        print(f"  Document 1: {args.doc1}")
        print(f"  Document 2: {args.doc2}")
        
        try:
            similarity = processor.compare_encrypted(args.doc1, args.doc2)
            print(f"\nSimilarity score: {similarity:.4f}")
            
            # Interpret score
            if similarity > 0.9:
                interpretation = "Very similar"
            elif similarity > 0.7:
                interpretation = "Similar"
            elif similarity > 0.5:
                interpretation = "Somewhat similar"
            else:
                interpretation = "Not very similar"
                
            print(f"Interpretation: {interpretation}")
            
        except Exception as e:
            print(f"Error: {e}")
            
    def cmd_search(self, args):
        """Search for similar documents."""
        processor = self._get_processor()
        
        if processor.fhe_model is None:
            print("Error: No compiled model found. Generate keys first:")
            print("  python fhe_cli.py keys generate")
            return
            
        print(f"Searching for documents similar to: '{args.query}'")
        print(f"Top {args.top_k} results with similarity >= {args.min_similarity}")
        
        results = processor.search_similar(
            args.query,
            top_k=args.top_k,
            min_similarity=args.min_similarity
        )
        
        if not results:
            print("\nNo similar documents found.")
            return
            
        print(f"\nFound {len(results)} similar documents:")
        for i, (doc_id, score) in enumerate(results, 1):
            doc_info = self.storage.index.get(doc_id, {})
            print(f"\n{i}. {doc_id} (similarity: {score:.4f})")
            if 'metadata' in doc_info and doc_info['metadata']:
                print(f"   Metadata: {doc_info['metadata']}")
                
    def cmd_stats(self, args):
        """Show system statistics."""
        print("FHE Document System Statistics")
        print("=" * 50)
        
        # Key stats
        keys = self.key_manager.list_keys()
        current_key = self.key_manager.get_current_key()
        print(f"\nKeys:")
        print(f"  Total keys: {len(keys)}")
        print(f"  Current key: {current_key}")
        
        # Storage stats
        storage_stats = self.storage.get_stats()
        print(f"\nStorage:")
        print(f"  Total documents: {storage_stats['total_documents']}")
        print(f"  Total size: {storage_stats['total_size_mb']:.2f} MB")
        if storage_stats['total_documents'] > 0:
            print(f"  Average size: {storage_stats['average_size_bytes']:.0f} bytes")
            
        # Memory stats (if processor exists)
        if self.processor is not None:
            memory_stats = self.processor.get_memory_stats()
            print(f"\nMemory:")
            print(f"  Current usage: {memory_stats['current_mb']:.1f} MB")
            print(f"  Used by app: {memory_stats['used_mb']:.1f} MB")
            
    def cmd_validate(self, args):
        """Validate document integrity."""
        print("Validating all documents...")
        
        validation = self.storage.validate_all()
        valid = validation['valid']
        invalid = validation['invalid']
        
        print(f"\nValidation Results:")
        print(f"  Valid documents: {len(valid)}")
        print(f"  Invalid documents: {len(invalid)}")
        
        if invalid:
            print(f"\nInvalid documents:")
            for doc_id in invalid:
                print(f"  - {doc_id}")
                
        if args.fix and invalid:
            print(f"\nRemoving {len(invalid)} invalid documents...")
            for doc_id in invalid:
                self.storage.delete(doc_id)
            print("Invalid documents removed.")
            
    def cmd_estimate(self, args):
        """Estimate resources for ICP deployment."""
        print("ICP Resource Estimation")
        print("=" * 50)
        
        # Document stats
        storage_stats = self.storage.get_stats()
        n_docs = storage_stats['total_documents']
        avg_size = storage_stats['average_size_bytes'] if n_docs > 0 else 1000
        
        print(f"\nCurrent System:")
        print(f"  Documents: {n_docs}")
        print(f"  Average size: {avg_size} bytes")
        
        # ICP constraints
        print(f"\nICP Constraints:")
        print(f"  Max message size: 2 MB")
        print(f"  Max memory: 4 GB")
        print(f"  Max instructions: 5 billion")
        
        # Estimates
        print(f"\nEstimates:")
        docs_per_message = int(2 * 1024 * 1024 / avg_size) if avg_size > 0 else 0
        print(f"  Documents per message: {docs_per_message}")
        
        # Cycle costs (rough estimates)
        encrypt_cycles = 1_000_000  # 1M cycles per encryption
        compare_cycles = 500_000    # 500K cycles per comparison
        
        print(f"  Encryption cost: ~{encrypt_cycles:,} cycles/document")
        print(f"  Comparison cost: ~{compare_cycles:,} cycles/operation")
        
        if n_docs > 0:
            total_encrypt_cycles = n_docs * encrypt_cycles
            print(f"  Total encryption cycles: ~{total_encrypt_cycles:,}")
            
        print(f"\nNote: These are rough estimates. Actual costs may vary.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FHE Document Encryption and Comparison CLI"
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Keys command
    keys_parser = subparsers.add_parser('keys', help='Manage FHE keys')
    keys_sub = keys_parser.add_subparsers(dest='action', help='Key actions')
    
    keys_gen = keys_sub.add_parser('generate', help='Generate new keys')
    keys_gen.add_argument('--key-id', help='Custom key ID')
    
    keys_sub.add_parser('list', help='List available keys')
    
    keys_rot = keys_sub.add_parser('rotate', help='Rotate keys')
    keys_rot.add_argument('--grace-days', type=int, default=7,
                         help='Grace period in days (default: 7)')
    
    # Encrypt command
    encrypt_parser = subparsers.add_parser('encrypt', help='Encrypt a document')
    encrypt_parser.add_argument('text', nargs='?', help='Text to encrypt')
    encrypt_parser.add_argument('--file', '-f', help='Read text from file')
    encrypt_parser.add_argument('--id', help='Document ID')
    encrypt_parser.add_argument('--tags', nargs='*', help='Document tags')
    encrypt_parser.add_argument('--metadata', help='JSON metadata')
    
    # Encrypt batch command
    batch_parser = subparsers.add_parser('encrypt-batch', 
                                        help='Encrypt multiple documents')
    batch_parser.add_argument('input_file', help='JSON file with documents')
    batch_parser.add_argument('--output-file', '-o', help='Save IDs to file')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two documents')
    compare_parser.add_argument('doc1', help='First document ID')
    compare_parser.add_argument('doc2', help='Second document ID')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for similar documents')
    search_parser.add_argument('query', help='Query text')
    search_parser.add_argument('--top-k', type=int, default=5,
                              help='Number of results (default: 5)')
    search_parser.add_argument('--min-similarity', type=float, default=0.5,
                              help='Minimum similarity (default: 0.5)')
    
    # Stats command
    subparsers.add_parser('stats', help='Show system statistics')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', 
                                          help='Validate document integrity')
    validate_parser.add_argument('--fix', action='store_true',
                               help='Remove invalid documents')
    
    # Estimate command
    subparsers.add_parser('estimate', help='Estimate ICP resources')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    # Create CLI instance
    cli = FHEDocumentCLI()
    
    # Route to appropriate command
    command_map = {
        'keys': cli.cmd_keys,
        'encrypt': cli.cmd_encrypt,
        'encrypt-batch': cli.cmd_encrypt_batch,
        'compare': cli.cmd_compare,
        'search': cli.cmd_search,
        'stats': cli.cmd_stats,
        'validate': cli.cmd_validate,
        'estimate': cli.cmd_estimate
    }
    
    handler = command_map.get(args.command)
    if handler:
        try:
            handler(args)
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
        except Exception as e:
            logger.error(f"Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()