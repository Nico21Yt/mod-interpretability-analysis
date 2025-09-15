#!/usr/bin/env python3
"""
Circuit Tracing Runner
Responsible for executing the complete EAP analysis workflow
"""

import os
import json
import datetime
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import networkx as nx
import pygraphviz as pgv

from .eap_analyzer import EAPAnalyzer


class CircuitTracingRunner:
    """Circuit Tracing Runner"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._setup_device()
        
    def _setup_device(self):
        """Setup computing device"""
        print(f"--- Current CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES','')}' ---")
        
        # Check CUDA availability
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        
        # Handle device configuration
        device_config = self.config.get("device", "auto")
        if device_config == "auto":
            # Auto-detect: if CUDA is available and device count > 0, use cuda:0
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
        else:
            # Manually specified device
            device = torch.device(device_config)
        
        if device.type == "cuda":
            try:
                current_idx = torch.cuda.current_device()
                print(f"Using cuda:{current_idx} / name='{torch.cuda.get_device_name(current_idx)}'")
            except Exception as e:
                print(f"Error getting CUDA device info: {e}")
                print("Falling back to CPU")
                device = torch.device("cpu")
        else:
            print("Falling back to CPU")
            
        return device
    
    def parse_layer_string(self, layer_str: str) -> List[int]:
        """Parse layer string configuration"""
        layers = set()
        parts = layer_str.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                if start > end:
                    raise ValueError("Range start value cannot be greater than end value")
                layers.update(range(start, end + 1))
            else:
                layers.add(int(part))
        return sorted(list(layers))
    
    def get_node_attributes(self, node_name):
        """Get node attributes"""
        if "tok_embeds" in node_name:
            return 'embedding', -1, '#FFDDC1'
        if node_name.startswith('a'):
            return 'attention', int(node_name.split('.')[0][1:]), '#C1D4FF'
        if node_name.startswith('m'):
            return 'mlp', int(node_name[1:]), '#D4FFC1'
        return 'unknown', 99, '#E0E0E0'
    
    def draw_circuit(self, scores_file_path: str, output_file_path: str) -> None:
        """Draw circuit diagram"""
        drawing_config = self.config.get("drawing_config", {})
        mode = drawing_config.get("graph_mode", "all").lower()

        print(f"\n--- Starting circuit diagram drawing (mode: {mode}) ---")
        print(f"Loading scores from {scores_file_path}")
        
        with open(scores_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        pruned_circuit = data.get("pruned_circuit")
        if not pruned_circuit:
            print("Error: 'pruned_circuit' key not found. Skipping drawing.")
            return

        # Filter edges
        if mode == 'positive':
            edges_to_draw = [e for e in pruned_circuit if e.get("score", 0) > 0]
            title = "Modular Arithmetic EAP Circuit Analysis (Positive Edges Only)"
        elif mode == 'negative':
            edges_to_draw = [e for e in pruned_circuit if e.get("score", 0) < 0]
            title = "Modular Arithmetic EAP Circuit Analysis (Negative Edges Only)"
        else:
            edges_to_draw = pruned_circuit
            title = "Modular Arithmetic EAP Circuit Analysis (All Edges)"

        # Limit quantity
        top_n = drawing_config.get("top_edges", None)
        if isinstance(top_n, int) and top_n > 0:
            edges_to_draw = sorted(edges_to_draw, key=lambda x: abs(x["score"]), reverse=True)[:top_n]

        if not edges_to_draw:
            print(f"No edges found in mode '{mode}'. Nothing to draw.")
            return

        # Build graph
        G = nx.DiGraph()
        all_scores = [edge['score'] for edge in edges_to_draw]
        max_abs_score = max(abs(s) for s in all_scores) if all_scores else 1.0

        print(f"Found {len(edges_to_draw)} edges in mode '{mode}'. Building graph...")
        for edge_data in edges_to_draw:
            try:
                source, dest = edge_data["edge"].split(" -> ")
                score = edge_data["score"]
                color = "#FF6B6B" if score > 0 else "#4D96FF"
                penwidth = f"{1 + (abs(score) / max_abs_score) * 4.0:.2f}"
                G.add_edge(source, dest, score=score, color=color, penwidth=penwidth)
            except (ValueError, KeyError):
                print(f"Warning: Skipping malformed edge data: {edge_data}")
                continue

        # Node attributes
        for node in G.nodes():
            node_type, layer, color = self.get_node_attributes(node)
            G.nodes[node].update({
                'type': node_type,
                'layer': layer,
                'style': 'filled,rounded',
                'fillcolor': color,
                'shape': 'box',
                'fixedsize': 'true',
                'width': '2.0',
                'height': '0.8',
                'fontsize': drawing_config.get("node_fontsize", "14"),
                'fontname': 'Arial Bold'
            })

        A = nx.nx_agraph.to_agraph(G)
        A.graph_attr.update(
            rankdir='TB', 
            splines='spline',
            nodesep=drawing_config.get("nodesep", "1.0"),
            ranksep=drawing_config.get("ranksep", "3.0"),
            size=drawing_config.get("size", "30,50!"),
            dpi=drawing_config.get("dpi", "300"),
            label=title,
            labelfontsize="16",
            labelfontname="Arial Bold",
            bgcolor="white",
            pad="1.0"
        )
        A.node_attr.update(
            fontname='Arial Bold', 
            fontsize=drawing_config.get("node_fontsize", "14")
        )
        A.edge_attr.update(
            fontname='Arial', 
            fontsize=drawing_config.get("fontsize", "12"),
            arrowsize="0.8"
        )

        # Layered layout
        print("Arranging layout by layers...")
        layers = sorted(list(set(d['layer'] for _, d in G.nodes(data=True))))
        for layer in layers:
            nodes_in_layer = [n for n, d in G.nodes(data=True) if d['layer'] == layer]
            if nodes_in_layer:
                subgraph = A.add_subgraph(nodes_in_layer, name=f'cluster_{layer if layer >= 0 else "input"}')
                subgraph.graph_attr.update(
                    label=f'Layer {layer}' if layer >= 0 else 'Input Embedding',
                    style='filled', 
                    color='#F5F5F5',
                    fontsize="12",
                    fontname="Arial Bold"
                )

        print(f"Drawing graph and saving to: {output_file_path}")
        try:
            A.draw(output_file_path, format='png', prog='dot')
            print("Modular arithmetic circuit diagram saved successfully!")
        except Exception as e:
            print(f"Error during drawing: {e}")
    
    def load_data(self, prompts_file: str) -> tuple:
        """Load data"""
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)
        print(f"[Data] Modular arithmetic prompts loaded: {len(prompt_data)} items")
        
        questions = [item['clean_prompt'] for item in prompt_data]
        corrupted_prompts = [item['corrupted_prompt'] for item in prompt_data]
        
        return questions, corrupted_prompts
    
    def run_analysis(self, prompts_file: str) -> str:
        """
        Run complete EAP analysis
        
        Args:
            prompts_file: prompts data file path
            
        Returns:
            Result output directory path
        """
        print("=== Starting Circuit Tracing Analysis ===")
        
        # Parse layer configuration
        try:
            layers_list = self.parse_layer_string(self.config["layers_to_analyze_str"])
            self.config["layers_to_analyze"] = layers_list
            if "debug_layer_for_head_check" not in self.config:
                self.config["debug_layer_for_head_check"] = layers_list[-1]
            print(f"[Config] Layers to analyze: {layers_list}")
        except ValueError as e:
            print(f"Error parsing layer string: {e}")
            return None

        # Load model
        print(f"Loading model: {self.config['model_name']}...")
        model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            torch_dtype=torch.float16
        ).to(self.device)

        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        if self.config.get('use_checkpointing', False):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

        # Create analyzer
        analyzer = EAPAnalyzer(model, tokenizer, self.config)

        # Load data
        questions, corrupted_prompts = self.load_data(prompts_file)
        
        # Get sample count from configuration
        num_samples = self.config.get("num_samples", 5)
        max_available = min(len(questions), len(corrupted_prompts))
        if num_samples > max_available:
            print(f"Warning: Requested {num_samples} samples, but only {max_available} available. Using {max_available}.")
            num_samples = max_available
        
        print(f"\nStarting modular arithmetic EAP analysis with {num_samples} samples...")
        attribution_scores, generation_results = analyzer.analyze(
            questions[:num_samples], corrupted_prompts[:num_samples]
        )

        print("\n--- Modular arithmetic EAP analysis completed ---")
        if not attribution_scores:
            print("Analysis returned no scores. Please check settings.")
            return None

        # Create output directory
        circuit_results_base = self.config.get("circuit_results_dir", "circuit_results")
        os.makedirs(circuit_results_base, exist_ok=True)
        
        # Get next available index
        existing_indices = []
        if os.path.exists(circuit_results_base):
            for item in os.listdir(circuit_results_base):
                if os.path.isdir(os.path.join(circuit_results_base, item)) and item.isdigit():
                    existing_indices.append(int(item))
        
        next_index = max(existing_indices) + 1 if existing_indices else 1
        run_output_dir = os.path.join(circuit_results_base, str(next_index))
        os.makedirs(run_output_dir, exist_ok=True)
        print(f"\nSaving results to index directory: {run_output_dir} (index: {next_index})")

        # Save complete graph information
        all_edges_list = sorted(
            [{"edge": f"{src} -> {dest}", "score": score}
             for (src, dest), score in attribution_scores.items()],
            key=lambda item: abs(item["score"]), reverse=True
        )

        all_nodes = set()
        for edge_data in all_edges_list:
            try:
                source, dest = edge_data["edge"].split(" -> ")
                all_nodes.add(source)
                all_nodes.add(dest)
            except Exception:
                continue

        full_graph_info = {
            "task": "modular_arithmetic",
            "metric": "answer_logit_difference",
            "nodes": [{
                "id": node,
                "type": self.get_node_attributes(node)[0],
                "layer": self.get_node_attributes(node)[1]
            } for node in sorted(list(all_nodes), key=lambda x: (self.get_node_attributes(x)[1], x))],
            "edges": all_edges_list
        }
        full_info_path = os.path.join(run_output_dir, "full_graph_info.json")
        with open(full_info_path, 'w', encoding='utf-8') as f:
            json.dump(full_graph_info, f, indent=4, ensure_ascii=False)
        print(f"Complete graph information saved to: {full_info_path}")

        # Save pruned subgraph
        pruning_config = self.config.get("pruning_config", {"mode": "threshold", "threshold": 1.0})
        pruning_mode = pruning_config.get("mode", "threshold")
        
        if pruning_mode == "threshold":
            threshold = pruning_config.get("threshold", 1.0)
            pruned_circuit_edges = [edge for edge in all_edges_list if abs(edge["score"]) >= threshold]
            print(f"\nUsing threshold mode: Found {len(pruned_circuit_edges)} edges with |score| >= {threshold}.")
        elif pruning_mode == "percentage":
            percentage = pruning_config.get("percentage", 10.0)
            num_edges_to_keep = max(1, int(len(all_edges_list) * percentage / 100.0))
            pruned_circuit_edges = all_edges_list[:num_edges_to_keep]
            print(f"\nUsing percentage mode: Keeping top {percentage}% ({len(pruned_circuit_edges)}) edges out of {len(all_edges_list)} total.")
        else:
            threshold = 1.0
            pruned_circuit_edges = [edge for edge in all_edges_list if abs(edge["score"]) >= threshold]
            print(f"\nUnknown mode '{pruning_mode}', using default threshold mode: Found {len(pruned_circuit_edges)} edges with |score| >= {threshold}.")

        if pruned_circuit_edges:
            pruned_graph_info = {"pruned_circuit": pruned_circuit_edges}
            pruned_info_path = os.path.join(run_output_dir, "pruned_graph_info.json")
            with open(pruned_info_path, 'w', encoding='utf-8') as f:
                json.dump(pruned_graph_info, f, indent=4, ensure_ascii=False)
            print(f"Pruned graph information saved to: {pruned_info_path}")

            # Draw graph
            if self.config.get("drawing_config", {}).get("draw_graph", False):
                print("\n--- Drawing modular arithmetic circuit diagram ---")
                temp_for_drawing = {"pruned_circuit": pruned_circuit_edges}
                temp_info_path = os.path.join(run_output_dir, "temp_for_drawing.json")
                with open(temp_info_path, 'w', encoding='utf-8') as f:
                    json.dump(temp_for_drawing, f, indent=4, ensure_ascii=False)

                circuit_diagram_path = os.path.join(run_output_dir, "modular_arithmetic_circuit_diagram.png")
                self.draw_circuit(temp_info_path, circuit_diagram_path)
                os.remove(temp_info_path)
        else:
            if pruning_mode == "threshold":
                threshold = pruning_config.get("threshold", 1.0)
                print(f"No edges satisfy score threshold (>= {threshold}).")
            else:
                print("No edges found for pruning.")

        # Save generation results and configuration
        with open(os.path.join(run_output_dir, "generation_results.json"), 'w', encoding='utf-8') as f:
            json.dump(generation_results, f, indent=4, ensure_ascii=False)
        export_config = {k: v for k, v in self.config.items() if k != 'layers_to_analyze'}
        with open(os.path.join(run_output_dir, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(export_config, f, indent=4, ensure_ascii=False)
        print("Generation results and configuration saved.")

        print(f"\nModular arithmetic EAP analysis completed!")
        print(f"Results saved in: {run_output_dir}")
        
        return run_output_dir
