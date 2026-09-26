export ZHENGYING_ROOT=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying

export PATH=$ROOT/envs/closd/bin:$PATH
export HF_HOME=$ZHENGYING_ROOT/cache/huggingface
export TORCH_HOME=$ZHENGYING_ROOT/cache/torch
export XDG_CACHE_HOME=$ZHENGYING_ROOT/cache/xdg
export PIP_CACHE_DIR=$ZHENGYING_ROOT/cache/pip
export TMPDIR=$ZHENGYING_ROOT/tmp

export APPTAINER_CACHEDIR=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/cache/apptainer
export APPTAINER_TMPDIR=/mnt/proj1/eu-26-37/it4i-dianyu/Zhengying/tmp
export TMPDIR=$ZHENGYING_ROOT/tmp

export ISAACGYM_BINDINGS=$ZHENGYING_ROOT/isaacgym/python/isaacgym/_bindings/linux-x86_64
export LD_LIBRARY_PATH=$ISAACGYM_BINDINGS:$ZHENGYING_ROOT/envs/closd/lib:${LD_LIBRARY_PATH:-}