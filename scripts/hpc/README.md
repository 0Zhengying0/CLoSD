# HPC jobs

These scripts run CLoSD and DiP on the local Slurm cluster. Submit them from
the repository with `sbatch scripts/hpc/<job>.slurm`.

Before using them on another cluster, update the `ROOT` and `CONTAINER` paths
and the `#SBATCH` account, partition, GPU, and log paths. Checkpoints, caches,
containers, and training output stay outside this repository.
