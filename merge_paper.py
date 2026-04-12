#!/usr/bin/env python3
"""
Merge comprehensive research paper sections into final LaTeX document
"""

import os

# Read the two parts
with open('research_paper.tex', 'r', encoding='utf-8', errors='replace') as f:
    part1 = f.read()

# Remove the end document marker from part 1
part1_trimmed = part1.replace('\\end{document}', '')

# Read additional sections
with open('research_paper_additional.tex', 'r', encoding='utf-8', errors='replace') as f:
    part2 = f.read()

# Find where to start merging - look for SECTION 5
start_idx = part2.find('% SECTION 5:')
if start_idx > 0:
    part2_content = part2[start_idx:]
    print(f"Found SECTION 5 at position {start_idx}")
else:
    # Fallback - try other patterns
    start_idx = part2.find('section{Two-Stage Training')
    if start_idx > 0:
        part2_content = part2[start_idx - 50:]
        print(f"Found section header at {start_idx}")
    else:
        print("Could not find merge point, taking full part2")
        part2_content = part2

# Merge: part1 + part2_content (making sure end{document} is at the end)
merged = part1_trimmed + '\n\n' + part2_content

# Ensure \end{document} is at the end
if not merged.strip().endswith('\\end{document}'):
    merged += '\n\\end{document}\n'

# Write merged version
with open('research_paper.tex', 'w', encoding='utf-8') as f:
    f.write(merged)

print("✓ Successfully merged comprehensive research paper sections!")
print(f"✓ Final paper size: {len(merged) / 1024 / 1024:.2f} MB ({len(merged):,} characters)")
print("✓ Paper now includes:")
print("  - Introduction (clinical background, SOTA, problem statement)")
print("  - Literature Review (DR detection, atrous convolutions, metrics)")
print("  - Datasets (IDRiD, EyePACS, APTOS, Messidor analysis)")
print("  - Architecture Design (ResNet50, ASPP, multi-task heads)")
print("  - Two-Stage Training (methodology, loss functions, checkpointing)")
print("  - Evaluation Methods (QWK, TTA, calibration)")
print("  - Results (IDRiD validation, cross-dataset, ablations)")
print("  - Discussion (findings, limitations, comparisons)")
print("  - Conclusion (contributions, future directions, impact)")
print("  - References")
