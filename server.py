from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from collections import defaultdict
import os

app = Flask(__name__)
CORS(app)

def simulate_dfa(states, alphabet, transitions, start, accept, input_string):
    steps = []
    current = start
    accept_set = set(accept)
    steps.append({"step":0,"state":current,"remaining":input_string,"symbol":None,"accepted":current in accept_set})
    for i, symbol in enumerate(input_string):
        if symbol not in alphabet:
            return {"accepted":False,"steps":steps,"error":f"Symbol '{symbol}' not in alphabet"}
        next_state = transitions.get(f"{current},{symbol}")
        if next_state is None:
            steps.append({"step":i+1,"state":"DEAD","remaining":input_string[i+1:],"symbol":symbol,"accepted":False})
            return {"accepted":False,"steps":steps}
        current = next_state
        steps.append({"step":i+1,"state":current,"remaining":input_string[i+1:],"symbol":symbol,"accepted":current in accept_set})
    return {"accepted":current in accept_set,"steps":steps}

def epsilon_closure(states_set, transitions):
    closure = set(states_set)
    stack = list(states_set)
    while stack:
        s = stack.pop()
        for ns in transitions.get(f"{s},ε", []):
            if ns not in closure:
                closure.add(ns); stack.append(ns)
    return frozenset(closure)

def nfa_move(states_set, symbol, transitions):
    result = set()
    for s in states_set:
        result.update(transitions.get(f"{s},{symbol}", []))
    return result

def simulate_nfa(states, alphabet, transitions, start, accept, input_string):
    accept_set = set(accept)
    current_states = epsilon_closure({start}, transitions)
    steps = [{"step":0,"states":sorted(list(current_states)),"remaining":input_string,"symbol":None,"accepted":bool(current_states & accept_set)}]
    for i, symbol in enumerate(input_string):
        if symbol not in alphabet:
            return {"accepted":False,"steps":steps,"error":f"Symbol '{symbol}' not in alphabet"}
        current_states = epsilon_closure(nfa_move(current_states, symbol, transitions), transitions)
        steps.append({"step":i+1,"states":sorted(list(current_states)),"remaining":input_string[i+1:],"symbol":symbol,"accepted":bool(current_states & accept_set)})
        if not current_states: break
    return {"accepted":bool(current_states & accept_set),"steps":steps}

def nfa_to_dfa(states, alphabet, transitions, start, accept):
    accept_set = set(accept)
    start_closure = epsilon_closure({start}, transitions)
    unmarked = [start_closure]
    dfa_states = {start_closure:"q0"}
    dfa_transitions = {}; dfa_accept = []; counter = [1]; conversion_steps = []
    while unmarked:
        current = unmarked.pop(0)
        current_name = dfa_states[current]
        step_info = {"dfa_state":current_name,"nfa_states":sorted(list(current)),"moves":{}}
        for symbol in alphabet:
            closure = epsilon_closure(nfa_move(current, symbol, transitions), transitions)
            if closure not in dfa_states:
                dfa_states[closure] = f"q{counter[0]}"; counter[0] += 1; unmarked.append(closure)
            target = dfa_states[closure]
            dfa_transitions[f"{current_name},{symbol}"] = target
            step_info["moves"][symbol] = target
        if current & accept_set: dfa_accept.append(current_name)
        conversion_steps.append(step_info)
    return {"states":list(dfa_states.values()),"alphabet":alphabet,"transitions":dfa_transitions,"start":dfa_states[start_closure],"accept":dfa_accept,"conversion_steps":conversion_steps,"state_mapping":{v:sorted(list(k)) for k,v in dfa_states.items()}}

def thompson(regex):
    n = [0]
    def ns(): s=f"s{n[0]}"; n[0]+=1; return s
    transitions = defaultdict(list)
    out = []
    for i, c in enumerate(regex):
        out.append(c)
        if i+1 < len(regex):
            nx = regex[i+1]
            if c not in '(|' and nx not in ')|*+?': out.append('.')
    regex = ''.join(out)
    prec = {'|':1,'.':2,'*':3,'+':3,'?':3}
    postfix, ops = [], []
    for c in regex:
        if c=='(': ops.append(c)
        elif c==')':
            while ops and ops[-1]!='(': postfix.append(ops.pop())
            if ops: ops.pop()
        elif c in prec:
            while ops and ops[-1]!='(' and prec.get(ops[-1],0)>=prec[c]: postfix.append(ops.pop())
            ops.append(c)
        else: postfix.append(c)
    while ops: postfix.append(ops.pop())
    stack=[]; all_states=set(); alpha=set()
    def np(): s,e=ns(),ns(); all_states.add(s); all_states.add(e); return s,e
    for c in postfix:
        if c=='.':
            if len(stack)<2: return {"error":"Invalid regex"}
            s2,e2=stack.pop(); s1,e1=stack.pop(); transitions[f"{e1},ε"].append(s2); stack.append((s1,e2))
        elif c=='|':
            if len(stack)<2: return {"error":"Invalid regex"}
            s2,e2=stack.pop(); s1,e1=stack.pop(); ns2,ne=np()
            transitions[f"{ns2},ε"]+=[s1,s2]; transitions[f"{e1},ε"].append(ne); transitions[f"{e2},ε"].append(ne); stack.append((ns2,ne))
        elif c=='*':
            if not stack: return {"error":"Invalid regex"}
            s1,e1=stack.pop(); ns2,ne=np()
            transitions[f"{ns2},ε"]+=[s1,ne]; transitions[f"{e1},ε"]+=[s1,ne]; stack.append((ns2,ne))
        elif c=='+':
            if not stack: return {"error":"Invalid regex"}
            s1,e1=stack.pop(); ns2,ne=np()
            transitions[f"{ns2},ε"].append(s1); transitions[f"{e1},ε"]+=[s1,ne]; stack.append((ns2,ne))
        elif c=='?':
            if not stack: return {"error":"Invalid regex"}
            s1,e1=stack.pop(); ns2,ne=np()
            transitions[f"{ns2},ε"]+=[s1,ne]; transitions[f"{e1},ε"].append(ne); stack.append((ns2,ne))
        else:
            ns2,ne=np(); transitions[f"{ns2},{c}"].append(ne); alpha.add(c); stack.append((ns2,ne))
    if not stack: return {"error":"Invalid regex"}
    start,end=stack[0]
    return {"states":sorted(list(all_states)),"alphabet":sorted(list(alpha)),"transitions":{k:v for k,v in transitions.items()},"start":start,"accept":[end]}

def minimize_dfa(states, alphabet, transitions, start, accept):
    accept_set=set(accept); non_accept=set(states)-accept_set
    partitions=[p for p in [accept_set,non_accept] if p]
    worklist=list(partitions)
    steps=[{"step":0,"partitions":[sorted(list(p)) for p in partitions]}]
    while worklist:
        splitter=worklist.pop(0)
        for sym in alphabet:
            preds={s for s in states if transitions.get(f"{s},{sym}") in splitter}
            new_parts=[]
            for part in partitions:
                inter=part&preds; diff=part-preds
                if inter and diff:
                    new_parts+=[inter,diff]
                    if part in worklist: worklist.remove(part); worklist+=[inter,diff]
                    else: worklist.append(inter if len(inter)<=len(diff) else diff)
                else: new_parts.append(part)
            partitions=new_parts
            steps.append({"step":len(steps),"partitions":[sorted(list(p)) for p in partitions],"symbol":sym})
    s2p={}
    for i,part in enumerate(partitions):
        for s in part: s2p[s]=f"m{i}"
    min_trans={}
    for s in states:
        for sym in alphabet:
            t=transitions.get(f"{s},{sym}")
            if t: min_trans[f"{s2p[s]},{sym}"]=s2p[t]
    return {"states":list(set(s2p.values())),"alphabet":alphabet,"transitions":min_trans,"start":s2p[start],"accept":list({s2p[s] for s in accept}),"minimization_steps":steps,"state_mapping":{f"m{i}":sorted(list(p)) for i,p in enumerate(partitions)}}

# ── ROUTES ──────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/health')
def api_health():
    return jsonify({"status": "ok"})

@app.route('/api/simulate/dfa', methods=['POST'])
def api_dfa():
    d = request.json
    return jsonify(simulate_dfa(d['states'],d['alphabet'],d['transitions'],d['start'],d['accept'],d['input']))

@app.route('/api/simulate/nfa', methods=['POST'])
def api_nfa():
    d = request.json
    return jsonify(simulate_nfa(d['states'],d['alphabet'],d['transitions'],d['start'],d['accept'],d['input']))

@app.route('/api/convert/nfa-to-dfa', methods=['POST'])
def api_n2d():
    d = request.json
    return jsonify(nfa_to_dfa(d['states'],d['alphabet'],d['transitions'],d['start'],d['accept']))

@app.route('/api/convert/regex-to-nfa', methods=['POST'])
def api_r2n():
    return jsonify(thompson(request.json['regex']))

@app.route('/api/minimize/dfa', methods=['POST'])
def api_min():
    d = request.json
    return jsonify(minimize_dfa(d['states'],d['alphabet'],d['transitions'],d['start'],d['accept']))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
