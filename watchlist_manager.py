from flask import Flask, render_template, request, jsonify
from typing import Dict, List
import json
import os
import logging

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

if __name__ == '__main__':
    app.run(debug=True)