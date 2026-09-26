"""Import checks in separate interpreters (Isaac Gym requires import before torch)."""
import subprocess
import sys
for code in [
    'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))',
    'from isaacgym import gymapi, gymtorch; print("Isaac Gym + gymtorch import OK")',
    'import closd.run; print("CLoSD import OK")',
]:
    subprocess.check_call([sys.executable, '-c', code])
subprocess.check_call(['ninja', '--version'])
