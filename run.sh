#!/usr/bin/env bash

./performance.py
echo "performance models done"

./engagement.py
echo "engagement models done"

./clustering.py
echo "engagement normalization done"

./create_dataset.py
echo "datasets created"

./statistical_analysis.py
echo "statistical analysis done"

./performance_simulation.py &
echo "performance simulation done"

./feedback_simulation.py &
echo "feedback simulation done"


