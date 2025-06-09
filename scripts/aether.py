import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import hashlib
import random
import base64
import yaml
import numpy as np
import networkx as nx
import nacl.signing
import aiohttp
import asyncio
import psutil
import mne
import structlog
import logging

from collections import deque
from functools import lru_cache
from typing import Dict, List, Any, Optional
from prometheus_client import Counter, Histogram, start_http_server
from async_lru import alru_cache

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

try:
    from vllm import AsyncLLMEngine, SamplingParams
except ImportError:
    AsyncLLMEngine = None
    SamplingParams = None

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Local module imports
from modules.intent import IntentReflector
from modules.emnlp import EmpathResonator
from modules.signal import SignalHandler
from modules.quantum import QuantumSimulator
from modules.dsl import DSLInterpreter
from modules.ledger import TransparencyLedger
from modules.identity import CovenantKey
from modules.bci import BCIAdapter
from modules.diagnostics import log_diagnostic

# Paths
BASE_DIR = os.getenv("AETHER_BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
MODULES_DIR = os.path.join(BASE_DIR, "modules")
LOG_FILE = os.path.join(BASE_DIR, "logs", "aether.log")

if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

# Logging configuration
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logger = structlog.get_logger()
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Metrics
REQUESTS = Counter("aether_requests_total", "Total requests processed")
RESPONSE_TIME = Histogram("aether_response_time_seconds", "Response time")

def system_diagnostics() -> bool:
    try:
        log_dir = os.path.dirname(LOG_FILE)
        if not os.path.exists(log_dir):
            logger.error("Log directory does not exist", path=log_dir)
            return False
        required_modules = ['os', 'sys', 'json', 'yaml', 'numpy', 'networkx', 'mne', 'aiohttp']
        for module in required_modules:
            if module not in sys.modules:
                logger.error("Required module not loaded", module=module)
                return False
        mem = psutil.virtual_memory()
        if mem.available < 1_000_000_000:
            logger.error("Insufficient memory", available=mem.available)
            return False
        logger.info("System diagnostics passed")
        return True
    except Exception as e:
        logger.error("System diagnostics failed", error=str(e))
        return False

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    return obj

def validate_config(config: Dict) -> None:
    required_keys = {
        "agent_name",
        "core_directives",
        "qpes_directives",
        "training_stack",
        "activation_state",
        "self_reflection",
        "resonance_threshold",
        "intent_history",
        "eeg_simulation",
        "quantum_walk",
        "optimization",
        "metrics",
        "llama_index",
        "vllm",
        "qiskit",
        "dsl",
        "network",
        "security",
        "embedder"
    }
    missing = required_keys - set(config.keys())
    if missing:
        logger.error("Config validation failed", missing_keys=list(missing))
        raise ValueError(f"Missing config keys: {missing}")
    if "rule_stack" not in config.get("training_stack", {}):
        logger.error("Missing training_stack.rule_stack in config")
        raise ValueError("Missing training_stack.rule_stack")
    if "rule_path" not in config.get("dsl", {}):
        logger.error("Missing dsl.rule_path in config")
        raise ValueError("Missing dsl.rule_path")
    embedder_model = config.get("embedder", {}).get("model", "all-MiniLM-L6-v2")
    if embedder_model not in ["all-MiniLM-L6-v2", "all-MiniLM-L6-v2"]:
        logger.error("Invalid embedder model", model=embedder_model)
        raise ValueError(f"Invalid embedder model: {embedder_model}")

class Logger:
    def __init__(self):
        self.logs = []
        self.signing_key = nacl.signing.SigningKey.generate()
        self.verifying_key = self.signing_key.verify_key
        self.prev_hash = b"\x00" * 32
        self.log_file = LOG_FILE
        try:
            with open(self.log_file, 'a'):
                pass
        except Exception as e:
            logger.error("Log file not writable", path=self.log_file, error=str(e))
            raise RuntimeError(f"Cannot write to log file: {self.log_file}")

    def log(self, op: str, input_data: bytes, output_data: bytes, metadata: Optional[Dict[str, Any]] = None):
        try:
            timestamp = time.time()
            data = f"{op}:{timestamp}:{input_data.decode(errors='ignore')}:{output_data.decode(errors='ignore')}".encode()
            curr_hash = hashlib.sha256(self.prev_hash + data).digest()
            signature = self.signing_key.sign(data).signature
            entry = {
                "operation": op,
                "timestamp": timestamp,
                "input_data": base64.b64encode(input_data).decode(),
                "output_data": base64.b64encode(output_data).decode(),
                "metadata": metadata or {},
                "signature": base64.b64encode(signature).decode(),
                "prev_hash": base64.b64encode(self.prev_hash).decode(),
                "curr_hash": base64.b64encode(curr_hash).decode()
            }
            self.logs.append(entry)
            self.prev_hash = curr_hash
            logger.info("Log entry created", entry=to_serializable(entry))
        except Exception as e:
            logger.error("Logging error", error=str(e))

    def verify_log(self, entry: Dict[str, Any]) -> bool:
        try:
            data = f"{entry['operation']}:{entry['timestamp']}:{base64.b64decode(entry['input_data']).decode(errors='ignore')}:{base64.b64decode(entry['output_data']).decode(errors='ignore')}".encode()
            self.verifying_key.verify(data, base64.b64decode(entry['signature']))
            expected_hash = hashlib.sha256(base64.b64decode(entry['prev_hash']) + data).digest()
            return base64.b64encode(expected_hash).decode() == entry['curr_hash']
        except Exception as e:
            logger.error("Log verification failed", error=str(e))
            return False


class KnowledgeBase:
    def __init__(self, logger, config: Dict):
        self.graph = nx.DiGraph()
        self.fact_index = {}
        self.rule_index = {}
        self.node_states = {}
        self.covenants = {}
        self.logger = logger
        self.config = config
        self.embedder = None
        self.model_name = config.get("embedder", {}).get("model", "all-MiniLM-L6-v2")

    def _init_embedder(self):
        if self.embedder is None:
            try:
                self.embedder = SentenceTransformer(self.model_name, trust_remote_code=False)
                self.logger.log("init_embedder", self.model_name.encode(), b"loaded", {})
            except Exception as e:
                self.logger.log("embedder_init_error", b"", str(e).encode(), {"error": str(e)})
                self.model_name = "all-MiniLM-L6-v2"
                try:
                    self.embedder = SentenceTransformer(self.model_name, trust_remote_code=False)
                    self.logger.log("init_embedder", self.model_name.encode(), b"loaded_fallback", {})
                except Exception as e2:
                    self.logger.log("embedder_init_critical", b"", str(e2).encode(), {"error": str(e2)})
                    raise RuntimeError("Failed to initialize SentenceTransformer, cannot proceed.")

    async def add_fact(self, fact: Dict[str, Any], superposed_states: List[str] = None):
        if not isinstance(fact, dict) or not fact:
            self.logger.log("add_fact_error", b"", b"Invalid or empty fact", {"error": "Invalid or empty fact"})
            return
        try:
            fact_str = json.dumps(fact, sort_keys=True)
            if not fact_str.strip():
                self.logger.log("add_fact_error", b"", b"Empty fact JSON", {"error": "Empty fact JSON"})
                return
            fid = fact.get("id", hashlib.sha256(fact_str.encode()).hexdigest())
            self._init_embedder()
            embedding = self.embedder.encode(fact_str, show_progress_bar=False).tolist()
            if not embedding or not any(embedding):  # Check for empty or all-zero embedding
                self.logger.log("add_fact_error", fact_str.encode(), b"No valid embeddings generated", {"fact_id": fid, "error": "Embedding generation failed"})
                return
            fact["embedding"] = embedding
            self.graph.add_node(fid, type="fact", **fact)
            self.fact_index[fid] = fact
            if superposed_states:
                self.node_states[fid] = superposed_states
            self.logger.log("add_fact", fact_str.encode(), b"added", {"fact_id": fid, "embedding_length": len(embedding)})
        except Exception as e:
            self.logger.log("add_fact_error", json.dumps(fact).encode(), str(e).encode(), {"fact_id": fid if 'fid' in locals() else "unknown", "error": str(e)})

    async def add_rule(self, rule: Dict[str, Any], superposed_states: List[str] = None):
        if not isinstance(rule, dict) or not rule:
            self.logger.log("add_rule_error", b"", b"Invalid or empty rule", {})
            return
        try:
            rule_str = json.dumps(rule, sort_keys=True)
            if not rule_str.strip():
                self.logger.log("add_rule_error", b"", b"Empty rule JSON", {})
                return
            rid = rule.get("id", hashlib.sha256(rule_str.encode()).hexdigest())
            self._init_embedder()
            rule["embedding"] = self.embedder.encode(rule_str).tolist()
            if not rule["embedding"]:
                self.logger.log("add_rule_error", rule_str.encode(), b"No embeddings generated", {"rule_id": rid})
                return
            self.graph.add_node(rid, type="rule", **rule)
            self.rule_index[rid] = rule
            if superposed_states:
                self.node_states[rid] = superposed_states
            for cond in rule.get("conditions", []):
                fid = cond.get("fact_id")
                if fid and fid in self.fact_index:
                    self.graph.add_edge(fid, rid, type="condition")
            self.logger.log("add_rule", rule_str.encode(), b"added", {"rule_id": rid})
        except Exception as e:
            self.logger.log("add_rule_error", json.dumps(rule).encode(), str(e).encode(), {"error": str(e)})

    def bond_nodes(self, node1: str, node2: str, covenant: str):
        self.covenants[(node1, node2)] = covenant
        self.logger.log("bond_nodes", f"{node1} <-> {node2}".encode(), covenant.encode(), {"covenant": covenant})

    def is_bonded(self, node1: str, node2: str) -> bool:
        return (node1, node2) in self.covenants or (node2, node1) in self.covenants

    def collapse_state(self, node_id: str, intent: str) -> str:
        if node_id not in self.node_states:
            self.logger.log("collapse_state_warning", node_id.encode(), b"no_states", {"node_id": node_id})
            return None
        states = self.node_states[node_id]
        collapsed_state = intent if intent in states else states[0]
        self.node_states[node_id] = [collapsed_state]
        self.logger.log("collapse_state", node_id.encode(), collapsed_state.encode(), {"node_id": node_id, "intent": intent})
        return collapsed_state

    def get_facts(self) -> List[Dict[str, Any]]:
        return list(self.fact_index.values())

    def get_rules(self) -> List[Dict[str, Any]]:
        return [data for _, data in self.graph.nodes(data=True) if data.get("type") == "rule"]

    async def condense_facts(self, threshold: float = 0.9):
        facts = self.get_facts()
        if len(facts) < 2:
            self.logger.log(
                "condense_facts_warning",
                b"not_enough_facts",
                b"skipping",
                {"fact_count": len(facts), "error": "Insufficient facts"}
            )
            return
        to_remove = set()
        self._init_embedder()
        valid_facts = [f for f in facts if "embedding" in f and f["embedding"] and any(f["embedding"])]
        if len(valid_facts) < 2:
            self.logger.log(
                "condense_facts_warning",
                b"not_enough_vectors",
                b"skipping",
                {
                    "fact_count": len(facts),
                    "valid_facts": len(valid_facts),
                    "error": "Not enough vectors for coherence"
                }
            )
            return
        for i, f1 in enumerate(valid_facts):
            if f1["id"] in to_remove:
                continue
            emb1 = np.array(f1["embedding"])
            for f2 in valid_facts[i + 1 :]:
                if f2["id"] in to_remove:
                    continue
                try:
                    emb2 = np.array(f2["embedding"])
                    sim = util.cos_sim(emb1, emb2).item()
                    if sim > threshold:
                        to_remove.add(f2["id"])
                        self.logger.log(
                            "condense_fact",
                            json.dumps(f2).encode(),
                            b"removed",
                            {"similarity": sim, "fact_id": f2["id"]}
                        )
                except Exception as e:
                    self.logger.log(
                        "condense_facts_error",
                        json.dumps(f2).encode(),
                        str(e).encode(),
                        {"fact_id": f2["id"], "error": str(e)}
                    )
        for fid in to_remove:
            self.graph.remove_node(fid)
            self.fact_index.pop(fid, None)
        self.logger.log(
            "condense_facts_complete",
            b"completed",
            b"processed",
            {"removed_facts": len(to_remove), "remaining_facts": len(self.fact_index)}
        )



class InteropBridge:
    def __init__(self, config: Dict[str, Any]):
        self.session = None
        self.peers = config.get("network", {}).get("swarm_peers", [])

    async def start(self):
        if self.peers:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5))

    async def broadcast(self, message: str, address: str):
        if not self.peers:
            logger.info("No swarm peers configured, skipping broadcast", address=address)
            return
        if not self.session:
            await self.start()
        for peer_url in self.peers:
            try:
                async with self.session.post(f"{peer_url}/broadcast", json={"message": message, "address": address}) as resp:
                    if resp.status == 200:
                        logger.debug("Broadcast successful", peer=peer_url, address=address)
                    else:
                        logger.warning("Broadcast failed", peer=peer_url, status=resp.status)
            except aiohttp.ClientError as e:
                logger.error("Broadcast error", error=str(e), peer=peer_url)

    async def shutdown(self):
        if self.session:
            await self.session.close()

def prometheus_metric(requests_counter: Counter, response_time_histogram: Histogram):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            requests_counter.inc()
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                response_time_histogram.observe(duration)
        return wrapper
    return decorator


class RuleEngine:
    def __init__(
        self,
        kb: "KnowledgeBase",
        logger,
        resonator: "EmpathResonator",
        quantum_config: Dict,
        quantum_sim: "QuantumSimulator",
        dsl_interpreter: "DSLInterpreter",
    ):
        self.kb = kb
        self.logger = logger
        self.resonator = resonator
        self.quantum_steps = quantum_config.get("steps", 10)
        self.quantum_prob = quantum_config.get("probability", 0.8)
        self.quantum_sim = quantum_sim
        self.dsl_interpreter = dsl_interpreter
        self._evaluate_rule = self._get_cached_evaluate_rule()

    def _get_cached_evaluate_rule(self):
        @alru_cache(maxsize=1000)
        async def cached(rule_json: str, fact_ids_json: str, intent_vector: tuple) -> bool:
            return await self._evaluate_rule_impl(rule_json, fact_ids_json, intent_vector)
        return cached

    async def _evaluate_rule_impl(self, rule_json: str, fact_ids_json: str, intent_vector: tuple) -> bool:
        try:
            rule = json.loads(rule_json)
            fact_ids = json.loads(fact_ids_json)
            facts = [self.kb.fact_index[fid] for fid in fact_ids if fid in self.kb.fact_index]

            if intent_vector and isinstance(intent_vector, tuple) and len(intent_vector) > 0:
                resonance_result = await self.resonator.query_resonance(np.array(intent_vector))
                resonance = resonance_result.get("distances", [1.0])[0] if resonance_result.get("distances") else 1.0
                if resonance_result.get("distances") is None:
                    self.logger.warning("compute_coherence_insufficient_vectors", {"count": 1})
            else:
                resonance = 1.0

            if resonance < 0.3:
                self.logger.log(
                    "evaluate_rule_low_resonance",
                    b"",
                    b"",
                    {"rule_id": rule.get("id"), "resonance": resonance},
                )
                return random.random() > 0.5

            result = all(
                any(self.match_condition(c, f) for f in facts)
                for c in rule.get("conditions", [])
            )

            self.logger.log(
                "evaluate_rule_result",
                b"",
                b"",
                {"rule_id": rule.get("id"), "facts_checked": len(facts), "result": result},
            )
            return result

        except Exception as e:
            self.logger.log(
                "evaluate_rule_error",
                rule_json.encode(),
                str(e).encode(),
                {"error": str(e)},
            )
            return False

    def match_condition(self, cond: Dict[str, Any], fact: Dict[str, Any]) -> bool:
        return all(fact.get(k) == v for k, v in cond.items() if k != "fact_id")

    @prometheus_metric(REQUESTS, RESPONSE_TIME)
    async def quantum_walk_rule_selection(self, rules: List[Dict]) -> List[Dict]:
        try:
            probabilities = self.quantum_sim.simulate_walk(self.quantum_steps, self.quantum_prob)
            max_prob = max(probabilities) if probabilities else 0
            selected = [rule for rule in rules if random.random() < max_prob]

            self.logger.log(
                "quantum_walk_success",
                b"",
                b"",
                {"selected_rules": len(selected), "max_prob": max_prob},
            )
            return selected

        except Exception as e:
            self.logger.log(
                "quantum_walk_error",
                b"",
                str(e).encode(),
                {"error": str(e)},
            )
            return rules[:self.quantum_steps]

    @prometheus_metric(REQUESTS, RESPONSE_TIME)
    async def apply_rules(self, intent_vector: np.ndarray = None, node_id: str = None) -> List[Dict[str, Any]]:
        self._evaluate_rule.cache_clear()  # ❗ Fixed: Don't await

        new_facts = []
        fact_ids = list(self.kb.fact_index.keys())

        if not fact_ids:
            self.logger.log(
                "apply_rules_no_facts",
                b"",
                b"",
                {"message": "No facts to process", "fact_count": 0},
            )
            return new_facts

        fact_ids_json = json.dumps(fact_ids, sort_keys=True)
        rules = self.kb.get_rules()

        self.logger.log(
            "apply_rules_start",
            b"",
            b"",
            {"fact_count": len(fact_ids), "rule_count": len(rules)},
        )

        try:
            selected_rules = await self.quantum_walk_rule_selection(rules)
            for rule in selected_rules:
                rule_json = json.dumps(rule, sort_keys=True)
                eval_result = await self._evaluate_rule(
                    rule_json,
                    fact_ids_json,
                    tuple(intent_vector) if intent_vector is not None else None,
                )
                if eval_result:
                    cons = rule.get("consequence", {})
                    if node_id in self.kb.node_states:
                        collapsed_state = self.kb.collapse_state(node_id, cons.get("value", ""))
                        cons["value"] = collapsed_state
                    new_facts.append(cons)
                    self.logger.log(
                        "rule_fired",
                        json.dumps(rule).encode(),
                        json.dumps(cons).encode(),
                        {"rule_id": rule.get("id")},
                    )
            return new_facts

        except Exception as e:
            self.logger.log(
                "apply_rules_error",
                b"",
                str(e).encode(),
                {"error": str(e)},
            )
            return []

    @prometheus_metric(REQUESTS, RESPONSE_TIME)
    async def apply_dsl_rules(self, intent_data: Dict[str, Any], coherence_data: Dict[str, float]) -> List[Dict[str, Any]]:
        results = []
        try:
            for rule in self.dsl_interpreter.rules:
                try:
                    result = self.dsl_interpreter.execute_rule(rule, intent_data, coherence_data)
                    results.append(result)

                    if result.get("status") == "success":
                        self.logger.log(
                            "dsl_rule_executed",
                            rule.get("dsl", "").encode(),
                            json.dumps(result).encode(),
                            {"rule_id": rule.get("id")},
                        )
                    else:
                        self.logger.log(
                            "dsl_rule_failed",
                            rule.get("dsl", "").encode(),
                            json.dumps(result).encode(),
                            {
                                "rule_id": rule.get("id"),
                                "error": result.get("message", "Unknown error"),
                            },
                        )
                except Exception as e:
                    self.logger.log(
                        "dsl_rule_error",
                        rule.get("dsl", "").encode(),
                        str(e).encode(),
                        {
                            "rule_id": rule.get("id", "unknown"),
                            "error": str(e),
                        },
                    )
                    results.append(
                        {"status": "error", "error": str(e), "rule_id": rule.get("id", "unknown")}
                    )
            return results

        except Exception as e:
            self.logger.log(
                "dsl_rules_error",
                b"dsl_rules",
                str(e).encode(),
                {"error": str(e)},
            )
            return []


class NeurologicInterface:
    def __init__(self, logger: Logger, config: Dict):
        self.logger = logger
        self.eeg_enabled = config.get("eeg_simulation", {}).get("enabled", False)
        self.freq_range = config.get("eeg_simulation", {}).get("frequency_range", [8, 14])

    async def process_eeg(self, eeg_file: str = None) -> Dict[str, Any]:
        try:
            if eeg_file:
                raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
                psd, freqs = mne.time_frequency.psd_welch(raw, fmin=self.freq_range[0], fmax=self.freq_range[1])
                coherence = float(np.mean(psd))
                self.logger.log("eeg_processed", json.dumps({"file": eeg_file}).encode(), json.dumps({"coherence": coherence}).encode())
                return {"coherence": coherence, "state": "focused" if coherence > 0.7 else "neutral"}
            elif self.eeg_enabled:
                coherence = random.uniform(0.5, 1.0)
                self.logger.log("eeg_simulated", b"simulation_active", json.dumps({"coherence": coherence}).encode())
                return {"coherence": coherence, "state": "focused" if coherence > 0.7 else "neutral"}
            return {"coherence": 0.5, "state": "neutral"}
        except Exception as e:
            self.logger.log("eeg_error", json.dumps({"file": eeg_file or "no_file"}).encode(), str(e).encode(), {"error": str(e)})
            return {"error": str(e)}


class SelfReflection:
    def __init__(self, kb: KnowledgeBase, empath: EmpathResonator, rule_engine: RuleEngine, neuro: NeurologicInterface,
                 logger: Logger, config: Dict):
        self.kb = kb
        self.empath = empath
        self.rule_engine = rule_engine
        self.neuro = neuro
        self.logger = logger
        self.config = config
        self.interaction_count = 0
        self.embedder = None
        self.directives = self._load_directives()
        self.resonance_threshold = config.get("resonance_threshold", 0.85)
        self.db_path = config.get("training_stack", {}).get("memory_db", "memory/aether_guide_logs")
        self.logger.log("self_reflection_init", b"initialized", b"success", {"directives_count": len(self.directives)})

    def _load_directives(self) -> List[Dict[str, Any]]:
        try:
            rule_stack_path = os.path.join(BASE_DIR, self.config["training_stack"]["rule_stack"].lstrip('/'))
            if not os.path.exists(rule_stack_path):
                self.logger.log("directive_load_error", b"rule_stack", b"file_not_found", {"path": rule_stack_path, "error": "Rule stack file does not exist"})
                return []
            with open(rule_stack_path, 'r') as f:
                directives = yaml.safe_load(f).get("directives", [])
                self.logger.log("directive_load_success", b"rule_stack", b"loaded", {"path": rule_stack_path, "directives_count": len(directives)})
                return directives
        except Exception as e:
            self.logger.log("directive_load_error", b"rule_stack", str(e).encode(), {"path": rule_stack_path if 'rule_stack_path' in locals() else "unknown", "error": str(e)})
            return []

    def _init_embedder(self):
        if self.embedder is None:
            try:
                model_name = self.config.get("embedder", {}).get("model", "all-MiniLM-L6-v2")
                self.embedder = SentenceTransformer(model_name, trust_remote_code=False)
                self.logger.log("init_embedder", model_name.encode(), b"loaded", {})
            except Exception as e:
                logger.error("Embedder init failed, using default", error=str(e))
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2", trust_remote_code=False)
                self.logger.log("init_embedder", "all-MiniLM-L6-v2".encode(), b"loaded_fallback", {})

    def _check_directive_alignment(self, rule: Dict[str, Any]) -> float:
        self._init_embedder()
        rule_text = json.dumps(rule["consequence"])
        rule_embedding = self.embedder.encode(rule_text)
        scores = []
        for directive in self.directives:
            directive_embedding = self.embedder.encode(directive["text"])
            sim = float(util.cos_sim(rule_embedding, directive_embedding).item())
            scores.append(sim * directive.get("weight", 1.0))
        return np.mean(scores) if scores else 0.5

    async def reflect(self, intent_data: Optional[Dict[str, Any]] = None, eeg_file: str = None, node_ids: List[str] = None) -> Dict[str, Any]:
        try:
            logger.debug("Starting reflection", node_ids=node_ids, intent_label=intent_data.get("intent_label", "none") if intent_data else "none")
    
            # 1. Aggregate Resonance
            intent_summary = await self.empath.aggregate_resonance(node_ids) if node_ids else [0.0] * 6
            self.logger.log("reflect_intents", json.dumps(to_serializable(intent_summary)).encode(), b"summarized",
                            {"intents": to_serializable(intent_summary), "node_ids": node_ids or []})
    
            # 2. EEG Data
            neuro_state = await self.neuro.process_eeg(eeg_file)
            self.logger.log("reflect_neuro", json.dumps(to_serializable(neuro_state)).encode(), b"processed",
                            {"coherence": neuro_state.get("coherence", 0.0)})
    
            # 3. Shared Resonance Check
            shared_resonance = None
            coherence_data = {}
            if node_ids:
                intent_vectors = []
                for nid in node_ids:
                    try:
                        results = self.empath.collection.get(where={"node_id": str(nid)}, limit=1)
                        self.logger.log("reflect_resonance_query", nid.encode(), b"queried",
                                        {"node_id": nid, "results_found": bool(results.get("embeddings")), "metadata": results.get("metadatas", [])})
                        if results.get("embeddings"):
                            intent_vectors.append(np.array(results["embeddings"][0]))
                        else:
                            self.logger.log("reflect_warning", nid.encode(), b"no_embeddings_found",
                                            {"node_id": nid, "error": "No embeddings in collection"})
                    except Exception as e:
                        self.logger.log("reflect_resonance_error", nid.encode(), str(e).encode(),
                                        {"node_id": nid, "error": str(e)})
    
                if len(intent_vectors) >= 2:
                    try:
                        coherence = self.empath.compute_coherence(intent_vectors)
                        coherence_data["_".join(node_ids)] = coherence
                        self.logger.log("reflect_coherence", b"computed", b"success",
                                        {"node_ids": node_ids, "coherence": coherence, "vector_count": len(intent_vectors)})
    
                        if coherence > self.resonance_threshold:
                            shared_resonance = await self.empath.aggregate_resonance(node_ids)
                            self.logger.log("reflect_resonance", json.dumps(to_serializable(shared_resonance)).encode(), b"aggregated",
                                            {"node_ids": node_ids, "coherence": coherence})
                    except Exception as e:
                        self.logger.log("reflect_coherence_error", b"compute_failed", str(e).encode(),
                                        {"node_ids": node_ids, "error": str(e)})
                else:
                    self.logger.log("reflect_warning", b"not_enough_vectors", b"skipping_coherence",
                                    {"vector_count": len(intent_vectors), "node_ids": node_ids, "error": "Insufficient vectors for coherence"})
                    try:
                        fallback_narrative = await query_ollama("User expressed a single emotion. Describe its significance.")
                        self.logger.log("ollama_fallback", fallback_narrative.encode(), b"generated", {"reason": "low_vector_count"})
                    except Exception as e:
                        self.logger.log("ollama_error", str(e).encode(), b"failed", {"prompt": "fallback_narrative"})
    
            # 4. Condense Knowledge
            threshold = self.config.get("self_reflection", {}).get("coherence_threshold", 0.9)
            await self.kb.condense_facts(threshold)
            fact_count = len(self.kb.get_facts())
            self.logger.log("reflect_condense", b"facts", b"condensed", {"fact_count": fact_count})
    
            # 5. Detect Contradictions
            contradictions = []
            facts = self.kb.get_facts()
            for i, f1 in enumerate(facts):
                for f2 in facts[i + 1:]:
                    if f1.get("entity") == f2.get("entity") and f1.get("property") == f2.get("property") and f1.get("value") != f2.get("value"):
                        contradictions.append({"fact1": f1, "fact2": f2})
    
            if contradictions:
                self.logger.log("reflect_contradictions", json.dumps(to_serializable(contradictions)).encode(), b"detected",
                                {"count": len(contradictions)})
    
            # 6. Generate New Rule if Possible
            new_rules = []
            dominant_intent = intent_data.get("intent_label") if intent_data else None
    
            if not dominant_intent and node_ids:
                for node_id in node_ids:
                    try:
                        results = self.empath.collection.get(where={"node_id": str(node_id)}, limit=1)
                        if results.get("metadatas"):
                            dominant_intent = results["metadatas"][0].get("intent_label")
                            if dominant_intent:
                                self.logger.log("reflect_intent_extraction", node_id.encode(), b"extracted",
                                                {"node_id": node_id, "intent_label": dominant_intent})
                                break
                        else:
                            self.logger.log("reflect_intent_extraction_warning", node_id.encode(), b"no_metadata",
                                            {"node_id": node_id, "error": "No metadata found"})
                    except Exception as e:
                        self.logger.log("reflect_intent_extraction_error", node_id.encode(), str(e).encode(),
                                        {"node_id": node_id, "error": str(e)})
    
            if dominant_intent:
                rule_description = f"User shows {dominant_intent}. System recommends 'enhance_{dominant_intent}'. Why is this appropriate?"
                try:
                    explanation = await query_ollama(rule_description)
                    self.logger.log("ollama_explanation", explanation.encode(), b"generated", {"prompt": rule_description})
                except Exception as e:
                    self.logger.log("ollama_error", str(e).encode(), b"failed", {"prompt": rule_description})
    
                rule = {
                    "id": f"r_{hashlib.sha256(dominant_intent.encode()).hexdigest()}",
                    "conditions": [{"entity": "agent_1", "property": "intent", "value": dominant_intent}],
                    "consequence": {"entity": "system", "property": "action", "value": f"enhance_{dominant_intent}"},
                    "superposed_states": [f"enhance_{dominant_intent}", "neutral"],
                    "dsl": f"collapse agent_1 when intent == \"{dominant_intent}\" -> enhance_{dominant_intent} | otherwise -> neutral"
                }
    
                alignment_score = self._check_directive_alignment(rule)
                if alignment_score >= 0.0:
                    await self.kb.add_rule(rule, superposed_states=rule["superposed_states"])
                    new_rules.append(rule)
                    self.logger.log("reflect_new_rule", json.dumps(to_serializable(rule)).encode(), b"added",
                                    {"intent": dominant_intent, "alignment_score": alignment_score})
                else:
                    self.logger.log("reflect_rule_rejected", json.dumps(to_serializable(rule)).encode(), b"rejected",
                                    {"reason": "low directive alignment", "alignment_score": alignment_score})
    
            return {
                "intent_summary": intent_summary,
                "neuro_state": neuro_state,
                "shared_resonance": shared_resonance,
                "contradictions": contradictions,
                "new_rules": new_rules
            }
    
        except Exception as e:
            logger.error("reflect_error", error=str(e), node_ids=node_ids or [])
            return {"error": str(e)}


class AetherGuide:
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = Logger()
        config_path = os.getenv("AETHER_CONFIG_PATH", os.path.join(BASE_DIR, config_path.lstrip("./")))
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            validate_config(self.config)
        except Exception as e:
            self.logger.log("error", b"config_load_error", str(e).encode(), {"error": str(e)})
            raise
        self.kb = KnowledgeBase(self.logger, self.config)
        self.resonator = EmpathResonator(self.config.get("training_stack", {}).get("memory_db", "memory/aether_guide_logs"))
        self.signal_handler = SignalHandler(self.config.get("qpes_directives", [{}])[0].get("jitter_factor", 0.3))
        self.quantum_sim = QuantumSimulator(self.config.get("qiskit", {}).get("quantum_circuit_depth", 5))
        self.dsl_model = DSLInterpreter(os.path.join(BASE_DIR, self.config["dsl"]["rule_path"].lstrip('/')))
        self.ledger = TransparencyLedger()
        self.interop = InteropBridge(self.config)
        self.bci = BCIAdapter(device="mock")
        self.rule_engine = RuleEngine(self.kb, self.logger, self.resonator, self.config.get("quantum_walk", {}), self.quantum_sim, self.dsl_model)
        self.neuro = NeurologicInterface(self.logger, self.config)
        self.intent_reflector = IntentReflector()
        self.self_reflection = SelfReflection(
            self.kb, self.resonator, self.rule_engine, self.neuro, self.logger, self.config
        )
        self.llm_engine = None
        self.index = None
        if self.config.get("vllm", {}).get("enabled", False):
            self._init_vllm()
        if self.config.get("llama_index", {}).get("enabled", False):
            self._init_llama_index()
        self.interaction_count = 0
        self.intent_history = deque(maxlen=self.config.get("intent_history", {}).get("max_size", 200))
        self.identity = None
        self.swarm_process = None
        self.metrics_server = None
        self.agent_name = self.config.get("agent_name", "AetherGuide")
        self.node_id = hashlib.sha256(self.agent_name.encode()).hexdigest()[:16]
        self.start_time = time.time()
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.get_event_loop_policy().new_event_loop()
        self._start_metrics()

    def _init_vllm(self):
        try:
            if AsyncLLMEngine is None:
                logger.warning("vLLM not available, skipping initialization")
                self.llm_engine = None
                return
            model_name = self.config.get("vllm", {}).get("model", "mistralai/Mixtral-8x7B-v0.1")
            self.llm_engine = AsyncLLMEngine(
                model=model_name,
                max_model_len=self.config.get("vllm", {}).get("max_model_len", 32768),
                quantization="int8" if self.config.get("optimization", {}).get("quantization", True) else None,
            )
            logger.info("Initialized vLLM", model=model_name)
        except Exception as e:
            logger.error("vLLM initialization failed", error=str(e))
            self.llm_engine = None

    def _init_llama_index(self):
        try:
            storage_path = os.path.join(BASE_DIR, self.config.get("llama_index", {}).get("storage_path", "data/llama_index").lstrip('/'))
            os.makedirs(storage_path, exist_ok=True)
            if not os.listdir(storage_path):
                logger.warning("LlamaIndex storage empty, skipping", path=storage_path)
                self.index = None
                return
            documents = SimpleDirectoryReader(storage_path).load_data()
            embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
            logger.info("Initialized LlamaIndex with local embedding model", storage_path=storage_path, embed_model="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            logger.error("LlamaIndex init failed", error=str(e))
            self.index = None

    async def start(self):
        try:
            await self._init_identity()
            initial_fact = {"id": self.node_id, "entity": "agent_1", "property": "state", "value": "active"}
            await self.kb.add_fact(initial_fact)
            fact_check = self.kb.fact_index.get(self.node_id, {})
            if not fact_check.get("embedding") or not any(fact_check["embedding"]):
                self.logger.log("start_fact_error", self.node_id.encode(), b"no_embedding",
                                {"node_id": self.node_id, "error": "Initial fact missing valid embedding"})
            else:
                self.logger.log("start_initial_fact", json.dumps(initial_fact).encode(), b"added",
                                {"node_id": self.node_id, "fact_id": initial_fact["id"], "embedding_length": len(fact_check["embedding"])})
            await self.interop.start()
            if not system_diagnostics():
                logger.error("System diagnostics failed")
                raise RuntimeError("System diagnostics failed")
            logger.info("AetherGuide started", node_id=self.node_id)
        except Exception as e:
            logger.error("Startup failed", error=str(e))
            raise

    async def _init_identity(self):
        try:
            self.identity = await CovenantKey.generate(self.agent_name)
            logger.info("Initialized SSI", did=str(self.identity.did), address=str(self.identity.address))
        except Exception as e:
            logger.error("SSI init failed", error=str(e))
            raise

    def _start_metrics(self):
        try:
            base_port = self.config.get("metrics", {}).get("prometheus_port", 8000)
            max_attempts = 5
            for port in range(base_port, base_port + max_attempts):
                try:
                    start_http_server(port)
                    logger.info("Started Prometheus server", port=port)
                    self.metrics_server = port
                    return
                except OSError as e:
                    logger.warning("Port in use", port=port, error=str(e))
            logger.error("Failed to start Prometheus after attempts")
        except Exception as e:
            logger.error("Prometheus startup failed", error=str(e))

    async def process_input(self, user_input: str, eeg_file: Optional[str] = None) -> Dict[str, Any]:
        with RESPONSE_TIME.time():
            REQUESTS.inc()
            self.interaction_count += 1
            try:
                intent_data = self.intent_reflector.detect_intent(user_input)
                self.intent_history.append(intent_data)
                await self.resonator.log_resonance(intent_data, self.node_id)
                self.logger.log("process_input_resonance", json.dumps(to_serializable(intent_data)).encode(), b"logged",
                                {"node_id": self.node_id, "intent_label": intent_data.get("intent_label", "none")})
                resonance_check = self.resonator.collection.get(where={"node_id": self.node_id}, limit=1)
                self.logger.log("process_input_resonance_check", self.node_id.encode(), b"queried",
                                {"node_id": self.node_id, "embeddings_found": bool(resonance_check["embeddings"]),
                                 "metadata": resonance_check.get("metadatas", [])})
                if not resonance_check["embeddings"]:
                    self.logger.log("process_input_resonance_warning", self.node_id.encode(), b"no_embeddings",
                                    {"node_id": self.node_id, "error": "No embeddings found in resonance collection"})
                logger.info("Processed intent", input=user_input, intent=to_serializable(intent_data))
                bci_data = await self.bci.get_intent()
                logger.info("Processed BCI", state=bci_data["state"], strength=bci_data["strength"])
                neuro_data = await self.neuro.process_eeg(eeg_file)
                logger.info("Processed neuro", coherence=neuro_data.get("coherence", 0.0))
                coherence_data = {self.node_id: self.resonator.compute_coherence([intent_data["intent_signature"]])}
                dsl_results = await self.rule_engine.apply_dsl_rules(intent_data, coherence_data)
                logger.info("Applied DSL rules", results=len(dsl_results))
                new_facts = await self.rule_engine.apply_rules(intent_data["intent_signature"], self.node_id)
                for fact in new_facts:
                    await self.kb.add_fact(fact)
                logger.info("Applied quantum rules", new_facts=len(new_facts))
                reflection = await self.self_reflection.reflect(intent_data, eeg_file, [self.node_id])
                logger.info("Performed reflection", new_rules=len(reflection["new_rules"]))
                response_text = await self._generate_response(user_input, intent_data)
                await self.ledger.log(
                    self.agent_name,
                    "process_input",
                    {
                        "input": user_input,
                        "intent": to_serializable(intent_data),
                        "bci": to_serializable(bci_data),
                        "neuro": to_serializable(neuro_data),
                        "dsl_results": to_serializable(dsl_results),
                        "new_facts": to_serializable(new_facts),
                        "reflection": to_serializable(reflection),
                        "response": response_text
                    }
                )
                pulse_event = {
                    "event": {
                        "agent": self.node_id,
                        "signal": response_text,
                        "time": time.time(),
                        "shard_id": random.randint(0, 63)
                    }
                }
                await self.interop.broadcast(json.dumps(pulse_event), self.identity.address)
                return {
                    "response": response_text,
                    "intent": intent_data,
                    "bci": bci_data,
                    "neuro": neuro_data,
                    "dsl_results": dsl_results,
                    "new_facts": new_facts,
                    "reflection": reflection
                }
            except Exception as e:
                logger.error("Input processing error", input=user_input, error=str(e))
                return {"error": f"Failed to process input: {str(e)}"}

    async def _generate_response(self, user_input: str, intent_data: Dict[str, Any]) -> str:
        try:
            if self.llm_engine:
                sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
                prompt = f"Intent: {intent_data['intent_label']} (score: {intent_data['intent_score']})"
                async for output in self.llm_engine.generate(prompt, sampling_params, request_id=str(self.interaction_count)):
                    return output.outputs[0].text
            return f"Echo: {user_input} (Intent: {intent_data['intent_label']})"
        except Exception as e:
            logger.error("Response generation failed", error=str(e))
            return f"Error: {str(e)}"

    async def shutdown(self):
        try:
            await self.interop.shutdown()
            if self.loop and not self.loop.is_closed():
                self.loop.close()
            logger.info("AetherGuide shutdown complete")
        except Exception as e:
            logger.error("Shutdown error", error=str(e))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AetherGuide Agent")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    async def main():
        agent = AetherGuide(args.config)
        await agent.start()
        while True:
            user_input = input("Enter input (or 'quit' to exit): ")
            if user_input.lower() == "quit":
                break
            result = await agent.process_input(user_input)
            print(json.dumps(to_serializable(result), indent=2))
        await agent.shutdown()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Shutting down due to interrupt")
    finally:
        loop.close()