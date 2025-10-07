from __future__ import annotations
from flask import Flask, render_template, request, jsonify
from typing import Dict, List
import json
import os
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from chains import Chains, load_chains_from_config
from assets import Assets, load_assets_from_config

app = Flask(__name__)
WATCHLIST_PATH = "./watchlist.json"

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """Load config file and return token map"""
    with open(config_path) as f:
        config = json.load(f)

    # Build token map for each chain
    token_map = {}
    for asset in config.get('assets', []):
        for variant in asset.get('variants', []):
            variant_name = variant['name']
            for chain_name in variant['tokens'].keys():
                if chain_name not in token_map:
                    token_map[chain_name] = []
                if variant_name not in token_map[chain_name]:
                    token_map[chain_name].append(variant_name)

    # Sort token lists
    for chain in token_map:
        token_map[chain].sort()

    return {
        'token_map': token_map,
        'chains': list(token_map.keys())
    }

def get_chain_tokens(config: Dict, chain: str) -> List[str]:
    """Get available tokens for a specific chain"""
    tokens = []
    for asset in config['assets']:
        for variant in asset['variants']:
            if chain in variant['tokens']:
                tokens.append(variant['name'])
    return sorted(tokens)

def load_watchlist() -> List[Dict]:
    """Load current watchlist pairs"""
    logger.debug(f"Loading watchlist from {WATCHLIST_PATH}")
    try:
        if os.path.exists(WATCHLIST_PATH):
            with open(WATCHLIST_PATH, 'r', encoding='utf-8') as f:
                pairs = json.load(f)
            logger.debug(f"Loaded {len(pairs)} pairs from watchlist")
            return pairs
        else:
            logger.debug("Watchlist file not found, returning empty list")
            return []
    except Exception as e:
        logger.error(f"Error loading watchlist: {e}")
        return []

def render_watchlist_page(config_path: str) -> str:
    """Render watchlist page with config data"""
    config = load_config(config_path)
    pairs = load_watchlist()
    
    # Build chain options HTML
    chain_options = ''
    for chain in sorted(config['chains']):
        chain_options += f'<option value="{chain}">{chain}</option>\n'

    # Build current pairs HTML
    current_pairs = ''
    for i, pair in enumerate(pairs):
        current_pairs += f'''
        <div class="pair">
            {pair['A_chain']}:{pair['A']} > {pair['B_chain']}:{pair['B']} ({pair['base']})
            <button onclick="deletePair({i})">Delete</button>
        </div>
        '''

    # Read template
    with open('templates/watchlist.html', 'r', encoding='utf-8') as f:
        template = f.read()

    # Convert token map to JSON string with proper escaping
    token_map_str = json.dumps(config['token_map']).replace("'", "\\'").replace('"', '\\"')

    # Replace placeholders
    html = template.replace('{{{TOKEN_MAP}}}', token_map_str)
    html = html.replace('{{CHAIN_OPTIONS}}', chain_options)
    html = html.replace('{{CURRENT_PAIRS}}', current_pairs)

    return html

@app.route('/')
def index():
    return render_watchlist_page('./config.json')

@app.route('/api/pairs', methods=['GET'])
def get_pairs():
    """Get all pairs from watchlist"""
    pairs = load_watchlist()
    return jsonify(pairs)

@app.route('/api/pairs', methods=['POST'])
def add_pair():
    """Add new pair to watchlist"""
    try:
        # Log request data
        logger.debug(f"Received POST request data: {request.data}")
        
        # Get current pairs
        pairs = load_watchlist()
        logger.debug(f"Current pairs count: {len(pairs)}")
        
        # Get and validate new pair data
        if not request.is_json:
            logger.error("Request is not JSON")
            return jsonify({
                'status': 'error',
                'message': 'Content-Type must be application/json'
            }), 400
            
        new_pair = request.json
        logger.debug(f"Parsed new pair data: {new_pair}")
        
        # Validate pair format
        required_fields = ['A_chain', 'B_chain', 'A', 'B', 'base']
        missing_fields = [field for field in required_fields if field not in new_pair]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {missing_fields}'
            }), 400
            
        # Validate data types
        try:
            # Normalize data
            normalized_pair = {
                'A_chain': str(new_pair['A_chain']).lower(),
                'B_chain': str(new_pair['B_chain']).lower(),
                'A': str(new_pair['A']).upper(),
                'B': str(new_pair['B']).upper(),
                'base': float(new_pair['base'])
            }
            logger.debug(f"Normalized pair data: {normalized_pair}")
            
            # Add new pair
            pairs.append(normalized_pair)
            
            # Save updated watchlist
            with open(WATCHLIST_PATH, 'w', encoding='utf-8') as f:
                json.dump(pairs, f, indent=2)
                logger.debug("Watchlist saved successfully")
            
            return jsonify({
                'status': 'success',
                'message': 'Pair added successfully'
            })
            
        except ValueError as e:
            logger.error(f"Data type validation error: {e}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid data types'
            }), 400
            
    except Exception as e:
        logger.error(f"Error adding pair: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/pairs/<int:index>', methods=['DELETE'])
def delete_pair(index):
    """Delete pair from watchlist by index"""
    try:
        pairs = load_watchlist()
        
        # Check if index is valid
        if index < 0 or index >= len(pairs):
            return jsonify({
                'status': 'error',
                'message': 'Invalid pair index'
            }), 400
            
        # Remove pair
        pairs.pop(index)
        
        # Save updated watchlist
        with open(WATCHLIST_PATH, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2)
            
        return jsonify({
            'status': 'success',
            'message': 'Pair deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"Error deleting pair: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

class WatchlistManager:
    def __init__(self, chains: Chains, assets: Assets):
        self.chains = chains
        self.assets = assets
        self.root = tk.Tk()
        self.root.title("Watchlist Manager")
        self.pairs: List[Dict] = []
        self._setup_ui()
        self.filename = "watchlist.json"

    def _setup_ui(self):
        # Chain selection
        chain_frame = ttk.LabelFrame(self.root, text="Chains", padding="5 5 5 5")
        chain_frame.pack(fill=tk.X, padx=5, pady=5)
        
        from_chain_label = ttk.Label(chain_frame, text="From Chain:")
        from_chain_label.grid(row=0, column=0, padx=5, pady=5)
        
        # Get chain names from Chains object
        chain_names = [chain.name for chain in self.chains]
        
        self.from_chain_cb = ttk.Combobox(chain_frame, values=chain_names)
        self.from_chain_cb.grid(row=0, column=1, padx=5, pady=5)
        
        to_chain_label = ttk.Label(chain_frame, text="To Chain:")
        to_chain_label.grid(row=0, column=2, padx=5, pady=5)
        
        self.to_chain_cb = ttk.Combobox(chain_frame, values=chain_names)
        self.to_chain_cb.grid(row=0, column=3, padx=5, pady=5)
        
        # Token selection
        token_frame = ttk.LabelFrame(self.root, text="Tokens", padding="5 5 5 5")
        token_frame.pack(fill=tk.X, padx=5, pady=5)
        
        token_a_label = ttk.Label(token_frame, text="Token A:")
        token_a_label.grid(row=0, column=0, padx=5, pady=5)
        
        # Get token symbols from Assets object
        token_symbols = [asset.name for asset in self.assets.assets]
        
        self.token_a_cb = ttk.Combobox(token_frame, values=token_symbols)
        self.token_a_cb.grid(row=0, column=1, padx=5, pady=5)
        
        token_b_label = ttk.Label(token_frame, text="Token B:")
        token_b_label.grid(row=0, column=2, padx=5, pady=5)
        
        self.token_b_cb = ttk.Combobox(token_frame, values=token_symbols)
        self.token_b_cb.grid(row=0, column=3, padx=5, pady=5)
        
        # Base amount
        base_frame = ttk.LabelFrame(self.root, text="Base Amount", padding="5 5 5 5")
        base_frame.pack(fill=tk.X, padx=5, pady=5)
        
        base_label = ttk.Label(base_frame, text="Base:")
        base_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.base_entry = ttk.Entry(base_frame)
        self.base_entry.insert(0, "100000.0")
        self.base_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Control buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        add_btn = ttk.Button(btn_frame, text="Add Pair", command=self._add_pair)
        add_btn.pack(side=tk.LEFT, padx=5)
        
        save_btn = ttk.Button(btn_frame, text="Save", command=self._save_watchlist)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # Pairs list
        list_frame = ttk.LabelFrame(self.root, text="Pairs", padding="5 5 5 5")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.pairs_list = tk.Listbox(list_frame)
        self.pairs_list.pack(fill=tk.BOTH, expand=True)

    def load(self, filename: str):
        """Load watchlist from file"""
        self.filename = filename
        try:
            with open(filename) as f:
                self.pairs = json.load(f)
                self._refresh_list()
        except FileNotFoundError:
            pass  # Ignore if file doesn't exist
        except Exception as e:
            self._show_error(f"Failed to load watchlist: {e}")

    def run(self):
        """Start the UI"""
        self.root.mainloop()

    def _add_pair(self):
        """Add pair to watchlist"""
        try:
            pair = {
                "A_chain": self.from_chain_cb.get(),
                "B_chain": self.to_chain_cb.get(),
                "A": self.token_a_cb.get(),
                "B": self.token_b_cb.get(),
                "base": float(self.base_entry.get())
            }
            self.pairs.append(pair)
            self._refresh_list()
        except ValueError as e:
            self._show_error(f"Invalid input: {e}")

    def _refresh_list(self):
        """Refresh pairs list display"""
        self.pairs_list.delete(0, tk.END)
        for p in self.pairs:
            self.pairs_list.insert(tk.END, 
                f"{p['A_chain']}:{p['A']} > {p['B_chain']}:{p['B']} ({p['base']})")

    def _save_watchlist(self):
        """Save watchlist to file"""
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.pairs, f, indent=2)
            self._show_info(f"Saved {len(self.pairs)} pairs to {self.filename}")
        except Exception as e:
            self._show_error(f"Failed to save: {e}")

    def _show_error(self, msg: str):
        """Show error message"""
        messagebox.showerror("Error", msg)

    def _show_info(self, msg: str):
        """Show info message"""
        messagebox.showinfo("Info", msg)

def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['web', 'gui'], default='web',
                       help='Run in web or GUI mode')
    parser.add_argument('--config', default='config.json',
                       help='Path to config file')
    parser.add_argument('--watchlist', default='watchlist.json',
                       help='Path to watchlist file')
    args = parser.parse_args()

    if args.mode == 'web':
        # Run Flask web server
        app.run(debug=True)
    else:
        # Run Tkinter GUI
        try:
            with open(args.config) as f:
                config = json.load(f)
            
            chains = load_chains_from_config(config)
            assets = load_assets_from_config(config, chains)
            
            manager = WatchlistManager(chains, assets)
            manager.load(args.watchlist)
            manager.run()
        except Exception as e:
            print(f"Error: {e}")
            return 1
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())