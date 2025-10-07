from __future__ import annotations
from flask import Flask, render_template, request, jsonify
from typing import Dict, List
import json
import os
import logging
import tkinter as tk
from tkinter import ttk
from chains import Chains, load_chains_from_config

app = Flask(__name__)
WATCHLIST_PATH = "./watchlist.json"

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """Load config.json and transform it into required format"""
    logger.debug(f"Loading config from {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # Transform chains list to dict format
        chains_dict = {}
        for chain in config['chains']:
            chains_dict[chain['name']] = {
                'name': chain['name'],
                'chainId': chain['chain_id'],
                'enabled': True  # Assume all chains are enabled
            }
            
        # Transform assets list to dict format    
        assets_dict = {}
        for asset in config['assets']:
            assets_dict[asset['symbol']] = {
                'symbol': asset['symbol'],
                'decimals': asset['decimals'],
                'address': asset['address']
            }
            
        return {
            'chains': chains_dict,
            'assets': assets_dict
        }
        
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

def get_chain_tokens(config: Dict, chain: str) -> List[str]:
    """Get available tokens for a specific chain"""
    tokens = []
    for symbol, asset in config['assets'].items():
        if chain in asset['address']:
            tokens.append(symbol)
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
    """Generate watchlist management page"""
    logger.debug("Starting page rendering")
    config = load_config(config_path)
    
    # Get enabled chains
    enabled_chains = sorted(config['chains'].keys())
    logger.debug(f"Enabled chains: {enabled_chains}")
    
    # Generate chain options HTML
    chain_options = '\n'.join([
        f'<option value="{chain}">{chain.upper()}</option>' 
        for chain in enabled_chains
    ])
    
    # Create token map for each chain
    token_map = {
        chain: get_chain_tokens(config, chain) 
        for chain in enabled_chains
    }
    
    # Load current watchlist
    current_pairs = load_watchlist()
    current_pairs_html = '\n'.join([
        f"""
        <div class="pair">
            {pair['A_chain'].upper()}:{pair['A']} → {pair['B_chain'].upper()}:{pair['B']}
            (Base: {pair['base']})
            <button onclick="deletePair({i})">Delete</button>
        </div>
        """
        for i, pair in enumerate(current_pairs)
    ]) if current_pairs else '<div class="pair">No pairs in watchlist</div>'
    
    logger.debug(f"Generated HTML for {len(current_pairs) if current_pairs else 0} pairs")
    
    # Load and render template
    try:
        with open('templates/watchlist.html', 'r', encoding='utf-8') as f:
            template = f.read()
            
        html = template.replace('{{CHAIN_OPTIONS}}', chain_options)
        html = html.replace('{{TOKEN_MAP}}', json.dumps(token_map))
        html = html.replace('{{CURRENT_PAIRS}}', current_pairs_html)  # Add this line
        return html
        
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        raise

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
    def __init__(self, chains: Chains):
        """Initialize with Chains instance"""
        self.chains = chains
        self.root = tk.Tk()
        self.root.title("Watchlist Manager")
        self.pairs: List[Dict] = []
        self.filename = ""
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface"""
        # Frame for chain selection
        frame = tk.Frame(self.root)
        frame.pack(padx=10, pady=10)
        
        tk.Label(frame, text="From Chain:").grid(row=0, column=0)
        tk.Label(frame, text="To Chain:").grid(row=1, column=0)
        
        # Comboboxes for chain selection
        self.from_chain_cb = ttk.Combobox(frame, state="readonly")
        self.from_chain_cb.grid(row=0, column=1)
        self.to_chain_cb = ttk.Combobox(frame, state="readonly")
        self.to_chain_cb.grid(row=1, column=1)
        
        # Update chain comboboxes to use chain names from Chains
        chain_names = [chain.name for chain in self.chains]
        self.from_chain_cb["values"] = chain_names
        self.to_chain_cb["values"] = chain_names
        
        # Buttons for adding, removing, and saving pairs
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Add Pair", command=self._add_pair).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Remove Pair", command=self._remove_pair).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Save Watchlist", command=self._save_watchlist).pack(side=tk.LEFT, padx=5)
        
        # Listbox to display watchlist pairs
        self.pair_listbox = tk.Listbox(self.root, width=50)
        self.pair_listbox.pack(padx=10, pady=10)
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def _add_pair(self):
        """Add pair to watchlist"""
        from_chain = self.from_chain_cb.get()
        to_chain = self.to_chain_cb.get()
        
        # Validate chains exist
        from_chain_obj = self.chains.get_chain(from_chain)
        to_chain_obj = self.chains.get_chain(to_chain)
        if not from_chain_obj or not to_chain_obj:
            self._show_error("Invalid chain selected")
            return
            
        # Create new pair entry
        new_pair = {
            "A_chain": from_chain,
            "B_chain": to_chain,
            "A": "",  # Placeholder for token A
            "B": "",  # Placeholder for token B
            "base": 0  # Placeholder for base amount
        }
        
        # Add to pairs and update listbox
        self.pairs.append(new_pair)
        self.pair_listbox.insert(tk.END, f"{from_chain} → {to_chain}")
        self._show_info(f"Added pair: {from_chain} → {to_chain}")
        
    def _remove_pair(self):
        """Remove selected pair from watchlist"""
        try:
            selected_index = self.pair_listbox.curselection()[0]
            self.pairs.pop(selected_index)
            self.pair_listbox.delete(selected_index)
            self._show_info("Removed selected pair")
        except Exception as e:
            self._show_error(f"Failed to remove pair: {e}")
            
    def _save_watchlist(self):
        """Save watchlist to file"""
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.pairs, f, indent=2)
            self._show_info(f"Saved {len(self.pairs)} pairs to {self.filename}")
        except Exception as e:
            self._show_error(f"Failed to save: {e}")
            
    def _show_info(self, message: str):
        """Show informational message in status bar"""
        self.status_bar.config(text=message, fg="green")
        
    def _show_error(self, message: str):
        """Show error message in status bar"""
        self.status_bar.config(text=message, fg="red")
        
    def load(self, watchlist_path: str):
        """Load watchlist from file"""
        self.filename = watchlist_path
        try:
            with open(watchlist_path, 'r') as f:
                self.pairs = json.load(f)
            # Update listbox
            self.pair_listbox.delete(0, tk.END)
            for pair in self.pairs:
                self.pair_listbox.insert(tk.END, f"{pair['A_chain']} → {pair['B_chain']}")
            self._show_info(f"Loaded {len(self.pairs)} pairs from {watchlist_path}")
        except Exception as e:
            self._show_error(f"Failed to load watchlist: {e}")
            
    def run(self):
        """Run the Tkinter main loop"""
        self.root.mainloop()

# Update the main function to pass Chains instance
def main(config_path: str, watchlist_path: str):
    """Launch watchlist manager"""
    try:
        # Load chains from config
        with open(config_path) as f:
            config = json.load(f)
        chains = load_chains_from_config(config)
        
        # Create manager with chains
        manager = WatchlistManager(chains)
        manager.load(watchlist_path)
        manager.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == '__main__':
    app.run(debug=True)
    
    import sys
    if len(sys.argv) != 3:
        print("Usage: watchlist_manager.py <config.json> <watchlist.json>")
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2]))