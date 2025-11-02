"""
Simplified inference and evaluation for climbing route generation.
"""
import torch
import numpy as np
import re
from typing import List, Dict, Tuple
from transformers import PreTrainedTokenizerFast, LogitsProcessor, LogitsProcessorList
from .gpt import load_model
from .data_processing import HOLD_ID, HOLDCOORDINATES


class RouteConstraintProcessor(LogitsProcessor):
    """Enforce route constraints during generation."""
    
    def __init__(self, tokenizer, min_holds=5, max_holds=15):
        self.tokenizer = tokenizer
        self.min_holds = min_holds
        self.max_holds = max_holds
        
        vocab = tokenizer.get_vocab()
        self.start_tokens = [tid for token, tid in vocab.items() if token.startswith('start')]
        self.finish_tokens = [tid for token, tid in vocab.items() if token.startswith('finish')]
        self.hand_tokens = [tid for token, tid in vocab.items() if token.startswith('hand')]
        self.feet_tokens = [tid for token, tid in vocab.items() if token.startswith('feet')]
        self.all_hold_tokens = self.start_tokens + self.hand_tokens + self.finish_tokens + self.feet_tokens
        self.eos_token_id = tokenizer.eos_token_id
        
        print(f"Constraint processor initialized:")
        print(f"  Start tokens: {len(self.start_tokens)}")
        print(f"  Finish tokens: {len(self.finish_tokens)}")
        print(f"  Hand tokens: {len(self.hand_tokens)}")
        print(f"  Feet tokens: {len(self.feet_tokens)}")
        print(f"  EOS token: {self.eos_token_id}")
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for batch_idx in range(input_ids.shape[0]):
            decoded = self.tokenizer.decode(input_ids[batch_idx], skip_special_tokens=True)
            
            # Count holds by looking at actual tokens
            parts = decoded.split()
            if len(parts) < 2:
                continue
                
            # Skip angle and grade tokens
            holds = [p for p in parts[2:] if any(p.startswith(role) for role in ['start', 'hand', 'finish', 'feet'])]
            
            start_count = sum(1 for h in holds if h.startswith('start'))
            finish_count = sum(1 for h in holds if h.startswith('finish'))
            hand_count = sum(1 for h in holds if h.startswith('hand'))
            feet_count = sum(1 for h in holds if h.startswith('feet'))
            hold_count = len(holds)
            
            # Block EOS until minimum requirements met
            if hold_count < self.min_holds or start_count < 1 or finish_count < 1:
                scores[batch_idx, self.eos_token_id] = -float('inf')
            
            # Limit start holds to exactly 1-2
            if start_count >= 2:
                for token_id in self.start_tokens:
                    scores[batch_idx, token_id] = -float('inf')
            
            # Limit finish holds to exactly 1-2
            if finish_count >= 2:
                for token_id in self.finish_tokens:
                    scores[batch_idx, token_id] = -float('inf')
            
            # Force EOS if max holds reached
            if hold_count >= self.max_holds:
                for token_id in self.all_hold_tokens:
                    scores[batch_idx, token_id] = -float('inf')
                scores[batch_idx, self.eos_token_id] = 100.0  # Strong boost to EOS
        
        return scores


def parse_route(route_str: str) -> Dict:
    """Parse route string with spaces or underscores."""
    route_str = ' '.join(route_str.split())
    
    # Try space-separated format first
    pattern = r"angle(\d+)\s+grade(\d+)\s+(.*)"
    match = re.match(pattern, route_str)
    
    if not match:
        # Fallback to underscore format
        pattern = r"angle(\d+)_grade(\d+)_(.*)"
        match = re.match(pattern, route_str)
        if not match:
            return None
        angle, grade, holds_str = match.groups()
        holds = holds_str.split("_")
    else:
        angle, grade, holds_str = match.groups()
        holds = holds_str.split()
    
    hold_data = []
    for h in holds:
        h = h.strip()
        if not h:
            continue
        for role in ['start', 'hand', 'finish', 'feet']:
            if h.startswith(role):
                try:
                    hold_id = int(h.replace(role, ''))
                    hold_data.append({'role': role, 'id': hold_id})
                    break
                except ValueError:
                    continue
    
    return {
        'angle': int(angle),
        'grade': int(grade),
        'holds': hold_data,
        'raw_string': route_str
    }


def validate_route(route: Dict) -> Tuple[bool, List[str]]:
    """Check if route meets basic requirements."""
    issues = []
    
    if route is None:
        return False, ["Invalid format"]
    
    roles = [h['role'] for h in route['holds']]
    start_count = roles.count('start')
    finish_count = roles.count('finish')
    hand_count = roles.count('hand')
    feet_count = roles.count('feet')
    
    # Check start/finish requirements
    if not (1 <= start_count <= 2):
        issues.append(f"Invalid start holds: {start_count} (need 1-2)")
    if not (1 <= finish_count <= 2):
        issues.append(f"Invalid finish holds: {finish_count} (need 1-2)")
    
    # Check total holds
    total_holds = len(route['holds'])
    if total_holds < 5:
        issues.append(f"Too few holds: {total_holds} (need 5+)")
    if total_holds > 18:
        issues.append(f"Too many holds: {total_holds} (max 18)")
    
    # Check for at least some hand or feet holds
    if hand_count == 0 and feet_count == 0:
        issues.append("Must have at least one hand or feet hold")
    
    # Check for valid hold IDs
    invalid_ids = [h['id'] for h in route['holds'] if h['id'] not in HOLD_ID]
    if invalid_ids:
        issues.append(f"Invalid hold IDs: {len(invalid_ids)} holds")
    
    return len(issues) == 0, issues


def compute_metrics(route: Dict) -> Dict:
    """Calculate avg_reach and density metrics."""
    is_valid, issues = validate_route(route)
    
    if not is_valid:
        return {'valid': False, 'issues': issues}
    
    valid_holds = [h for h in route['holds'] if h['id'] in HOLD_ID]
    
    if len(valid_holds) < 2:
        return {'valid': True, 'avg_reach': 0, 'density': 0}
    
    # Get coordinates
    coords = np.array([HOLDCOORDINATES[HOLD_ID.index(h['id'])] for h in valid_holds])
    
    # Calculate reach distances
    reaches = [np.linalg.norm(coords[i+1] - coords[i]) for i in range(len(coords) - 1)]
    
    # Calculate area
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    area = x_range * y_range
    
    return {
        'valid': True,
        'issues': [],
        'num_holds': len(valid_holds),
        'avg_reach': np.mean(reaches),
        'density': len(valid_holds) / (area + 1),
    }


def evaluate_batch(routes: List[str]) -> Dict:
    """Evaluate multiple routes."""
    metrics_list = [compute_metrics(parse_route(r)) for r in routes]
    valid_metrics = [m for m in metrics_list if m['valid']]
    invalid_metrics = [m for m in metrics_list if not m['valid']]
    
    stats = {
        'total': len(routes),
        'valid': len(valid_metrics),
        'valid_rate': len(valid_metrics) / len(routes) if routes else 0,
    }
    
    # Show common issues
    if invalid_metrics:
        all_issues = []
        for m in invalid_metrics:
            all_issues.extend(m.get('issues', []))
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        stats['common_issues'] = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    if valid_metrics:
        stats['avg_holds'] = np.mean([m['num_holds'] for m in valid_metrics])
        stats['avg_reach_mean'] = np.mean([m['avg_reach'] for m in valid_metrics])
        stats['density_mean'] = np.mean([m['density'] for m in valid_metrics])
    
    # Add metrics to each route for ranking
    stats['route_metrics'] = metrics_list
    
    return stats


def generate_routes(
    model_path: str,
    prompt: str = "angle40_grade18",
    num_routes: int = 20,
    use_constraints: bool = True,
    temperature: float = 0.9,
    max_new_tokens: int = 50,
    ) -> Tuple[List[str], Dict]:
    """Generate and evaluate climbing routes."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    
    prompt_list = prompt.split('_')
    angle, grade, holds = None, None, []

    for p in prompt_list:
        if p.startswith('angle'):
            angle = int(p.replace('angle', ''))
            if not (20 <= angle <= 60):
                raise ValueError(f"Invalid angle: {angle} (must be 20-60)")
        elif p.startswith('grade'):
            grade = int(p.replace('grade', ''))
            if not (12 <= grade <= 28):
                raise ValueError(f"Invalid grade: {grade} (must be 12-28)")
        elif any(p.startswith(role) for role in ['start', 'hand', 'finish', 'feet']):
            holds.append(p)
        else:
            raise ValueError(f"Invalid prompt part: {p}")


    holds_str = " ".join(holds) + " " if holds else ""
    prompt = f"{tokenizer.bos_token}angle{angle} grade{grade} {holds_str}"
    
    logits_processor = None
    if use_constraints:
        logits_processor = LogitsProcessorList([
            RouteConstraintProcessor(tokenizer, min_holds=5, max_holds=12)
        ])
    
    model_inputs = tokenizer([prompt] * num_routes, return_tensors="pt", 
                            add_special_tokens=False).to(device)
    
    with torch.no_grad():
        output_ids = model.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=logits_processor,
            repetition_penalty=1.2,
        )
    
    routes = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    
    # Debug: print first route details
    if routes:
        print(f"\nFirst generated route: {routes[0][:150]}...")
        parsed = parse_route(routes[0])
        if parsed:
            roles = [h['role'] for h in parsed['holds']]
            print(f"  Total holds: {len(parsed['holds'])}")
            print(f"  Start: {roles.count('start')}, Finish: {roles.count('finish')}")
            print(f"  Hand: {roles.count('hand')}, Feet: {roles.count('feet')}")
    
    stats = evaluate_batch(routes)
    
    return routes, stats


def main():
    model_path = 'models/climb_gpt/run_20251007_195358'
    
    print("Generating routes...")
    routes, stats = generate_routes(
        model_path=model_path,
        prompt = "angle40_grade18_start1149", 
        use_constraints=True,
    )
    
    print(f"\nValid: {stats['valid']}/{stats['total']} ({stats['valid_rate']:.1%})")
    if stats.get('common_issues'):
        print(f"Common issues:")
        for issue, count in stats['common_issues']:
            print(f"  - {issue}: {count}x")
    
    if stats['valid'] > 0:
        print(f"\nMetrics (valid routes only):")
        print(f"  Avg holds: {stats['avg_holds']:.1f}")
        print(f"  Avg reach: {stats['avg_reach_mean']:.1f}")
        print(f"  Density: {stats['density_mean']:.3f}")
    
    # Rank routes by quality score
    route_scores = []
    for i, (route, metrics) in enumerate(zip(routes, stats['route_metrics'])):
        if metrics['valid']:
            # Quality score: balance between reach distance and density
            score = metrics['avg_reach'] * metrics['density']
            route_scores.append((score, i, route, metrics))
    
    route_scores.sort(reverse=True, key=lambda x: x[0])
    
    from .visualization import Visualization
    viz = Visualization()
    
    print(f"\nTop 5 routes (ranked by quality score):")
    for rank, (score, idx, route, metrics) in enumerate(route_scores[:5], 1):
        print(f"{rank}. Score={score:.2f} | {route}")
       
        viz.plot_boulder(route)


if __name__ == "__main__":
    main()