export PYTHONPATH=$(pwd):$PYTHONPATH

# Flatland Cyclic Colors
echo "Running Flatland Cyclic Colors..."
python scripts/results/action_clustering.py flc
python scripts/results/test_algo.py flc
python scripts/results/compute_dislib.py flc

# Flatland Permutation Colors
echo "Running Flatland Permutation Colors..."
python scripts/results/action_clustering.py flp
python scripts/results/test_algo.py flp
python scripts/results/compute_dislib.py flp

# COIL2
echo "Running COIL2..."
python scripts/results/action_clustering.py coil2
python scripts/results/test_algo.py coil2
python scripts/results/longterm_prediction.py coil2
python scripts/results/compute_dislib.py coil2

# COIL3
echo "Running COIL3..."
python scripts/results/action_clustering.py coil3
python scripts/results/test_algo.py coil3
python scripts/results/longterm_prediction.py coil3
python scripts/results/compute_dislib.py coil3

# 3DShapes
echo "Running 3DShapes..."
python scripts/results/action_clustering.py shapes
python scripts/results/test_algo.py shapes
python scripts/results/compute_dislib.py shapes

# MPI3D
echo "Running MPI3D..."
python scripts/results/test_algo.py mpi3d
python scripts/results/compute_dislib.py mpi3d
python scripts/results/test_algo.py mpi3d_noisy
python scripts/results/compute_dislib.py mpi3d_noisy

# COIL2 entangled action
echo "Running COIL2 entangled action..."
python scripts/results/compute_dislib.py coil2_entangled

# COIL random actions clustering
python scripts/results/action_clustering.py coil_randomaction

# COIL iid generalisation
echo "Running COIL iid generalisation..."
python scripts/results/generalisation.py coil2_iid
python scripts/results/generalisation.py coil3_iid

# COIL ood generalisation
echo "Running COIL iid generalisation..."
python scripts/results/generalisation.py coil2_ood
python scripts/results/generalisation.py coil3_ood