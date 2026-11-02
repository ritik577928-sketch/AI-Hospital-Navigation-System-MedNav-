# Hospital Navigation System
### AI-Powered Smart Hospital Pathfinding

---

## Quick Start

```bash
pip install flask
python app.py
```
Then open: http://127.0.0.1:5000

---

## Algorithms Implemented

| Algorithm | Type | Optimal? |
|-----------|------|----------|
| BFS | Uninformed | ✗ (min hops) |
| DFS | Uninformed | ✗ |
| UCS | Informed | ✓ |
| A* | Informed | ✓ |
| GBFS | Greedy | ✗ |
| Minimax α-β | Adversarial | ✓ (game-optimal) |
| CSP | Scheduling | ✓ |

## Features
- Interactive SVG hospital floor plan (2 floors)
- Step-by-step traversal animation on map
- Wheelchair token animation along found path
- Traversal log synced with animation
- CSP department scheduling with MRV + LCV heuristics
- Minimax adversarial pathfinding with Alpha-Beta pruning
- PEAS model display
- Department info on node click

## Team
- Ritik Kumar

Course: MDSA (25MT1306E) | Batch 12
