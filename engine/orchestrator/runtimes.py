import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger("VisionForge.Runtimes")

class RuntimeBuilder:
    @staticmethod
    def build_command(runtime: str, config: dict, script_path: str, args: list, workspace_root: str):
        """
        Builds the shell command to execute the runner script in the requested runtime.
        
        Args:
            runtime: 'local', 'conda', 'docker'
            config: dict with keys like 'conda_env', 'docker_image'
            script_path: Absolute path to the python runner script
            args: List of arguments for the script
            workspace_root: Absolute path to the workspace root (for Docker mounting)
        
        Returns:
            (cmd_list, cwd, env)
        """
        
        # 1. LOCAL
        if runtime == "local":
            python_exec = sys.executable
            cmd = [python_exec, script_path] + args
            cwd = os.path.dirname(script_path)
            return cmd, cwd, None

        # 2. ANACONDA
        elif runtime == "conda":
            env_name = config.get("conda_env", "base")
            # We use 'conda run' to execute in the environment without activating shell
            # Requires conda to be in PATH
            cmd = ["conda", "run", "-n", env_name, "--no-capture-output", "python", script_path] + args
            cwd = os.path.dirname(script_path)
            return cmd, cwd, None

        # 3. DOCKER
        elif runtime == "docker":
            image = config.get("docker_image", "ultralytics/ultralytics:latest")
            
            # Mounts
            # 1. Workspace -> /workspace
            # 2. Project Root (Engine) -> /app
            
            # We assume script_path is inside the project root.
            # Let's find project root relative to script. 
            # script is in engine/runners/..., project root is 3 levels up usually?
            # Safe bet: Mount the parent of 'engine' as /app
            
            engine_parent = Path(script_path).parent.parent.parent # runners -> engine -> root
            local_workspace = Path(workspace_root)
            
            # Path Translation Helpers
            def translate_path(local_path):
                p = str(local_path)
                if str(local_workspace) in p:
                    return p.replace(str(local_workspace), "/workspace")
                # If script path or other project paths
                if str(engine_parent) in p:
                    return p.replace(str(engine_parent), "/app")
                return p # Return original if we can't map (might fail in docker)
            
            # Translate Args
            docker_args = []
            for arg in args:
                docker_args.append(translate_path(arg))
                
            container_script_path = translate_path(script_path)
            
            docker_cmd = [
                "docker", "run", "--rm",
                "--ipc=host", # Shared memory for PyTorch
                "--gpus", "all", # GPU support
                "-v", f"{local_workspace}:/workspace",
                "-v", f"{engine_parent}:/app",
                "-w", os.path.dirname(container_script_path), # Workdir
                image,
                "python", container_script_path
            ] + docker_args
            
            # Return CWD as local engine parent just to launch the docker client 
            return docker_cmd, str(engine_parent), None

        else:
            raise ValueError(f"Unknown runtime: {runtime}")
