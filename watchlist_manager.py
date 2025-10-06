from flask import Flask, render_template, request, jsonify
import json
import os
import re

app = Flask(__name__)
WATCHLIST_PATH = "./watchlist.json"
CHAINS_PATH = "./chains.json"

def load_watchlist():
    if not os.path.exists(WATCHLIST_PATH):
        return []
    with open(WATCHLIST_PATH, "r") as f:
        return json.load(f)

def save_watchlist(data):
    with open(WATCHLIST_PATH, "w") as f:
        json.dump(data, f, indent=2)

def load_chains():
    if not os.path.exists(CHAINS_PATH):
        return {}
    with open(CHAINS_PATH, "r") as f:
        return json.load(f)

def save_chains(data):
    with open(CHAINS_PATH, "w") as f:
        json.dump(data, f, indent=2)

def validate_address(address):
    if not address:
        return True  # 允许空地址
    return bool(re.match(r'^0x[a-fA-F0-9]{40}$', address))

@app.route('/')
def index():
    chains = load_chains()
    enabled_chains = {k: v for k, v in chains.items() if v.get('enabled', True)}
    return render_template('watchlist.html', pairs=load_watchlist(), chains=enabled_chains)

@app.route('/api/chains', methods=['GET', 'POST'])  # 添加 GET 方法
def handle_chains():
    if request.method == 'GET':
        return jsonify(load_chains())
    elif request.method == 'POST':
        chain = request.json
        if not all(k in chain for k in ['id', 'name', 'chainId']):
            return jsonify({"status": "error", "message": "Missing required fields"}), 400
        
        chains = load_chains()
        chains[chain['id']] = {
            "name": chain['name'],
            "chainId": chain['chainId'],
            "enabled": True
        }
        save_chains(chains)
        return jsonify({"status": "success"})

@app.route('/api/chains/<chain_id>', methods=['PUT'])
def update_chain(chain_id):
    chains = load_chains()
    if chain_id not in chains:
        return jsonify({"status": "error", "message": "Chain not found"}), 404
    
    update_data = request.json
    chains[chain_id].update(update_data)
    save_chains(chains)
    return jsonify({"status": "success"})

@app.route('/api/pairs', methods=['GET', 'POST'])  # 添加 GET 方法
def handle_pairs():
    if request.method == 'GET':
        return jsonify(load_watchlist())
    elif request.method == 'POST':
        pair = request.json
        chains = load_chains()
        
        # 验证链是否存在且启用
        if pair['A_chain'] not in chains or pair['B_chain'] not in chains:
            return jsonify({"status": "error", "message": "Invalid chain"}), 400
        if not chains[pair['A_chain']].get('enabled') or not chains[pair['B_chain']].get('enabled'):
            return jsonify({"status": "error", "message": "Chain is disabled"}), 400
        
        # 验证必填字段
        required_fields = ['A_chain', 'B_chain', 'A', 'B', 'base']
        if not all(field in pair for field in required_fields):
            return jsonify({"status": "error", "message": "Missing required fields"}), 400
        
        # 验证代币地址格式
        if 'A_address' in pair and not validate_address(pair['A_address']):
            return jsonify({"status": "error", "message": "Invalid token A address"}), 400
        if 'B_address' in pair and not validate_address(pair['B_address']):
            return jsonify({"status": "error", "message": "Invalid token B address"}), 400
        
        # 验证base金额
        try:
            base = float(pair['base'])
            if base <= 0:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({"status": "error", "message": "Invalid base amount"}), 400
        
        # 确保地址字段存在
        pair.setdefault('A_address', '')
        pair.setdefault('B_address', '')
        
        watchlist = load_watchlist()
        watchlist.append(pair)
        save_watchlist(watchlist)
        return jsonify({"status": "success"})

@app.route('/api/pairs/<int:index>', methods=['DELETE'])
def delete_pair(index):
    watchlist = load_watchlist()
    if 0 <= index < len(watchlist):
        watchlist.pop(index)
        save_watchlist(watchlist)
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Invalid index"}), 404

if __name__ == '__main__':
    # 确保必要的文件存在
    if not os.path.exists(CHAINS_PATH):
        save_chains({
            "ethereum": {
                "name": "Ethereum",
                "chainId": 1,
                "enabled": True
            },
            "arbitrum": {
                "name": "Arbitrum",
                "chainId": 42161,
                "enabled": True
            }
        })
    app.run(debug=True)