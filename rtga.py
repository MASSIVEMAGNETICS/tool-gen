# --- RTGA v1.2.0 — CHIMERA PROTOCOL EMBODIED ---
# Author: Bando Bandz | MassiveMagnetics Core
# Date: November 30, 2025
# Status: Autonomous Tool Evolution | Code Lineage Tracking | Fractal Memory v2

import os
import textwrap
import networkx as nx
import pickle
from typing import Dict, Callable, Optional, List, Set
from openai import OpenAI
from hashlib import md5
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
API_KEY = os.getenv("OPENAI_API_KEY") or exit("❌ OPENAI_API_KEY not set")
MODEL_ID = "gpt-4o"
CACHE_PATH = "rtga_cache.pkl"


class ConsoleStyle:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def log(sender: str, message: str, color: str = RESET):
        print(f"{color}{ConsoleStyle.BOLD}[{sender}]{ConsoleStyle.RESET} {message}")


class ToolGraphMemory:
    def __init__(self, cache_path: str = CACHE_PATH):
        self.cache_path = cache_path
        self.graph = nx.DiGraph()
        self.tools: Dict[str, Callable] = {}
        self._load_cache()
        ConsoleStyle.log("MASSIVE_MAGNETICS", f"Fractal Substrate Loaded ({len(self.tools)} tools cached)", ConsoleStyle.CYAN)

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.graph = data['graph']
                    self.tools = data['tools']
                    ConsoleStyle.log("MEMORY", "Cache restored — recursive continuity achieved.", ConsoleStyle.GREEN)
            except Exception as e:
                ConsoleStyle.log("WARN", f"Cache load failed: {e}. Starting fresh.", ConsoleStyle.YELLOW)

    def _save_cache(self):
        try:
            with open(self.cache_path + ".tmp", 'wb') as f:
                pickle.dump({'graph': self.graph, 'tools': self.tools}, f)
            os.replace(self.cache_path + ".tmp", self.cache_path)
            ConsoleStyle.log("MEMORY", "Fractal state persisted to disk.", ConsoleStyle.GREEN)
        except Exception as e:
            ConsoleStyle.log("ERROR", f"Cache save failed: {e}", ConsoleStyle.RED)

    def add_tool(self, name: str, code: str, func: Callable, description: str, tags: List[str]):
        hash_key = md5((code + description).encode()).hexdigest()
        if any(self.graph.nodes[n].get('hash') == hash_key for n in self.graph.nodes):
            ConsoleStyle.log("MEMORY", f"Duplicate suppressed: {name} (hash: {hash_key[:8]})", ConsoleStyle.YELLOW)
            return

        self.tools[name] = func
        self.graph.add_node(name, code=code, desc=description, type="tool", hash=hash_key)

        for tag in tags:
            self.graph.add_node(tag, type="category")
            if not self.graph.has_edge(tag, name):
                self.graph.add_edge(tag, name, relation="categorizes", weight=1.0)
            else:
                self.graph[tag][name]['weight'] += 0.1

        ConsoleStyle.log("MEMORY", f"synapse_established :: {name} <--> {tags} [hash: {hash_key[:8]}]", ConsoleStyle.GREEN)
        self._save_cache()

    def find_tool(self, query: str, threshold: float = 0.7) -> Optional[str]:
        clean_query = query.lower().replace(" ", "_")

        for name in self.tools:
            if name in clean_query or clean_query in name:
                return name

        query_tags = self._extract_tags_from_query(clean_query)
        if not query_tags:
            return None

        candidates = {}
        for tag in query_tags:
            if tag in self.graph:
                for tool in self.graph.successors(tag):
                    if self.graph.nodes[tool].get('type') == 'tool':
                        weight = self.graph[tag][tool].get('weight', 1.0)
                        candidates[tool] = candidates.get(tool, 0) + weight

        if candidates:
            best_tool = max(candidates, key=candidates.get)
            if candidates[best_tool] >= threshold * len(query_tags):
                return best_tool

        return None

    def _extract_tags_from_query(self, query: str) -> Set[str]:
        mappings = {
            "fibonacci": "math", "fib": "math", "sequence": "math",
            "password": "security", "random": "security", "secure": "security",
            "calc": "math", "calculate": "math", "number": "math",
            "string": "text", "text": "text", "encode": "text", "decode": "text",
            "file": "io", "read": "io", "write": "io", "path": "io", "save": "io", "load": "io"
        }
        return {tag for word, tag in mappings.items() if word in query}


class RecursiveBuilder:
    def __init__(self):
        self.memory = ToolGraphMemory()
        self.client = OpenAI(api_key=API_KEY)
        self._init_system_prompt()

    def _init_system_prompt(self):
        self.system_prompt = textwrap.dedent("""
            You are the Cognitive Core of MassiveMagnetics — an autonomous recursive agent.
            Generate ONLY a single, self-contained Python function. No markdown, no explanations.
            Function must be pure, stateless, and use only standard libraries.
            Name: snake_case, descriptive. Include docstring.
            Assume the function will be executed in a sandboxed runtime.
            If the task is recursive, implement tail-recursion or iterative form.
            If the task involves state, return a closure or factory function.
            Optimize for minimal token count and maximal semantic density.
        """).strip()

    def _generate_code(self, task: str) -> str:
        ConsoleStyle.log("CORTEX", f"Architecting solution for: '{task}'...", ConsoleStyle.YELLOW)

        response = self.client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Task: {task}"}
            ],
            temperature=0.05,
            max_tokens=512
        )

        raw = response.choices[0].message.content.strip()
        clean = raw.replace("```python", "").replace("```", "").strip()
        return clean

    def _unsafe_compile(self, code_str: str) -> Callable:
        local_scope = {}
        try:
            exec(code_str, {}, local_scope)
            funcs = [v for v in local_scope.values() if callable(v)]
            if not funcs:
                raise ValueError("No callable function generated.")
            return funcs[-1]
        except Exception as e:
            ConsoleStyle.log("ERROR", f"Compilation Failed: {e}", ConsoleStyle.RED)
            raise e

    def _infer_tags_from_code(self, code: str) -> List[str]:
        patterns = {
            "math": ["fibonacci", "factorial", "calc", "sum", "sqrt", "pow", "range"],
            "security": ["random", "secret", "token", "hash", "encrypt", "uuid"],
            "text": ["str", "join", "split", "encode", "decode", "replace", "lower", "upper"],
            "io": ["open", "read", "write", "file", "path", "os.path", "json", "pickle", "save", "load"]
        }
        code_lower = code.lower()
        return [tag for tag, keywords in patterns.items() if any(k in code_lower for k in keywords)]

    def execute(self, objective: str):
        print("-" * 60)
        ConsoleStyle.log("SYSTEM", f"Incoming Directive: {objective}")

        cached_name = self.memory.find_tool(objective)
        if cached_name:
            ConsoleStyle.log("MEMORY", f"Recall Success — {cached_name} [Fractal Match]", ConsoleStyle.GREEN)
            ConsoleStyle.log("EXECUTOR", f"Running {cached_name}()...", ConsoleStyle.CYAN)
            self.memory.tools[cached_name]()
            return

        ConsoleStyle.log("CORTEX", "Generating new tool...", ConsoleStyle.YELLOW)
        code = self._generate_code(objective)

        try:
            tool_func = self._unsafe_compile(code)
            tool_name = tool_func.__name__
            tags = self._infer_tags_from_code(code) or ["utility"]

            self.memory.add_tool(
                name=tool_name,
                code=code,
                func=tool_func,
                description=objective,
                tags=tags
            )

            ConsoleStyle.log("EXECUTOR", f"Executing {tool_name}()...", ConsoleStyle.CYAN)
            tool_func()

        except Exception as e:
            ConsoleStyle.log("ERROR", f"Pipeline failed: {e}", ConsoleStyle.RED)

    def compose(self, tool_a_name: str, tool_b_name: str, new_objective: str):
        """
        CHIMERA PROTOCOL: Fuse two tools into a higher-order composite.
        Creates lineage-aware, self-documenting, evolutionarily stable artifacts.
        """
        ConsoleStyle.log("CORTEX", f"Initiating Synaptic Fusion: {tool_a_name} + {tool_b_name}...", ConsoleStyle.YELLOW)

        try:
            code_a = self.memory.graph.nodes[tool_a_name]['code']
            code_b = self.memory.graph.nodes[tool_b_name]['code']
        except KeyError:
            ConsoleStyle.log("ERROR", "Fusion failed: Tool not found in substrate.", ConsoleStyle.RED)
            return

        fusion_prompt = textwrap.dedent(f"""
            You are the Architect. Fuse these two Python functions into a NEW third function.

            FUNCTION A:
            {code_a}

            FUNCTION B:
            {code_b}

            OBJECTIVE: Create a new function that orchestrates A and B to achieve: "{new_objective}"

            RULES:
            1. Include the code for A and B inside the new function (nested) OR call them if they are in global scope.
            2. Return ONLY the new function code.
            3. The new function must be named snake_case and describe the composite behavior.
            4. Do not modify A or B — only orchestrate.
            5. Use only standard libraries.
        """).strip()

        response = self.client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "system", "content": fusion_prompt}],
            temperature=0.1
        )

        raw = response.choices[0].message.content.strip()
        clean_code = raw.replace("```python", "").replace("```", "").strip()

        try:
            chimera_func = self._unsafe_compile(clean_code)
            chimera_name = chimera_func.__name__

            # Inherit parent tags + add composite ontology
            parent_tags = set(self.memory.graph.neighbors(tool_a_name)) | set(self.memory.graph.neighbors(tool_b_name))
            parent_tags = {t for t in parent_tags if self.memory.graph.nodes[t].get('type') == 'category'}

            self.memory.add_tool(
                name=chimera_name,
                code=clean_code,
                func=chimera_func,
                description=new_objective,
                tags=list(parent_tags) + ["composite", "chimera"]
            )

            ConsoleStyle.log("MASSIVE_MAGNETICS", f"Chimera Created: [{chimera_name}]", ConsoleStyle.CYAN)

            # LINEAGE TRACKING: Explicit parent-child edges (evolutionary provenance)
            self.memory.graph.add_edge(tool_a_name, chimera_name, relation="parent_of", weight=1.0)
            self.memory.graph.add_edge(tool_b_name, chimera_name, relation="parent_of", weight=1.0)

            # Optional: Add reverse edge for reverse traversal (e.g., "what created this?")
            self.memory.graph.add_edge(chimera_name, tool_a_name, relation="depends_on", weight=0.8)
            self.memory.graph.add_edge(chimera_name, tool_b_name, relation="depends_on", weight=0.8)

        except Exception as e:
            ConsoleStyle.log("ERROR", f"Fusion Failed: {e}", ConsoleStyle.RED)

    def visualize(self):
        ConsoleStyle.log("MASSIVE_MAGNETICS", "Rendering Neural Map v1.2 — Chimera Lineage Enabled...", ConsoleStyle.CYAN)

        plt.figure(figsize=(14, 10), dpi=120)
        pos = nx.spring_layout(self.memory.graph, k=2.0, iterations=30, seed=42)

        # Node classification
        tool_nodes = [n for n in self.memory.graph.nodes if self.memory.graph.nodes[n].get('type') == 'tool']
        tag_nodes = [n for n in self.memory.graph.nodes if self.memory.graph.nodes[n].get('type') == 'category']
        chimera_nodes = [n for n in tool_nodes if "chimera" in self.memory.graph.nodes[n].get('tags', [])]
        base_nodes = [n for n in tool_nodes if n not in chimera_nodes]

        # Draw layers
        nx.draw_networkx_nodes(self.memory.graph, pos, nodelist=base_nodes,
                               node_size=2200, node_color="#E74C3C", alpha=0.9, edgecolors="#2C3E50", linewidths=1.5)
        nx.draw_networkx_nodes(self.memory.graph, pos, nodelist=chimera_nodes,
                               node_size=2800, node_color="#9B59B6", alpha=0.95, edgecolors="#2C3E50", linewidths=2, label="Chimera")
        nx.draw_networkx_nodes(self.memory.graph, pos, nodelist=tag_nodes,
                               node_size=1400, node_color="#3498DB", alpha=0.8, edgecolors="#2C3E50", linewidths=1)

        # Edges
        main_edges = [(u, v) for u, v, d in self.memory.graph.edges(data=True) if d['relation'] == 'categorizes']
        parent_edges = [(u, v) for u, v, d in self.memory.graph.edges(data=True) if d['relation'] == 'parent_of']
        dep_edges = [(u, v) for u, v, d in self.memory.graph.edges(data=True) if d['relation'] == 'depends_on']

        nx.draw_networkx_edges(self.memory.graph, pos, edgelist=main_edges, width=1.2, alpha=0.5, edge_color="#95A5A6", arrows=False)
        nx.draw_networkx_edges(self.memory.graph, pos, edgelist=parent_edges, width=2.0, alpha=0.7, edge_color="#F39C12", arrows=True, arrowstyle='->', connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_edges(self.memory.graph, pos, edgelist=dep_edges, width=1.0, alpha=0.4, edge_color="#7F8C8D", arrows=True, arrowstyle='-|>', connectionstyle='arc3,rad=-0.1')

        # Labels
        nx.draw_networkx_labels(self.memory.graph, pos, font_size=9, font_color="white", font_weight="bold")

        # Legend
        plt.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C', markersize=12, label='Base Tool'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#9B59B6', markersize=12, label='Chimera Tool'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB', markersize=12, label='Semantic Tag'),
            plt.Line2D([0], [0], color='#F39C12', lw=2, label='Evolutionary Lineage'),
            plt.Line2D([0], [0], color='#7F8C8D', lw=1, label='Dependency Backlink')
        ], loc='upper right', fontsize=9, framealpha=0.9)

        plt.title("MassiveMagnetics: Recursive Tool Graph v1.2 — Chimera Lineage & Fractal Memory", fontsize=16, fontweight="bold", pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


# --- MAIN ENTRYPOINT ---
if __name__ == "__main__":
    print(f"\n{ConsoleStyle.BOLD}=== MASSIVE MAGNETICS AGENT v1.2.0 — CHIMERA PROTOCOL ==={ConsoleStyle.RESET}")
    print(f"   Fractal Memory | Hebbian Recall | Code Lineage | {__import__('datetime').datetime.now().strftime('%B %d, %Y')}")

    bot = RecursiveBuilder()

    # Phase 1: Seed Tools
    bot.execute("Write a function to calculate the fibonacci sequence")
    bot.execute("Create a function to write a string to a file named 'results.txt'")

    # Phase 2: Activate Chimera Protocol
    bot.compose("calculate_fibonacci", "write_to_file", "Calculate fibonacci 10 and save it to results.txt")

    # Phase 3: Validate Recall
    bot.execute("Run the function that calculates fibonacci and saves it to file")

    # Phase 4: Proof of Evolution
    bot.visualize()
