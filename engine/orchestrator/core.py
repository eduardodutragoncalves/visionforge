import asyncio
import json
import logging
import os
import sys
from typing import Dict, Optional
from datetime import datetime

from engine.core.events import manager
from engine.io import paths, manifests
from engine.orchestrator.post_processor import handle_training_completion
from engine.services.gpu_stats import get_gpu_stats

logger = logging.getLogger("VisionForge.Orchestrator")

class Orchestrator:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.active_jobs: Dict[str, Dict] = {}
        self.is_processing = False
        self.workspace_root = str(paths.get_workspace_root())
        self.max_concurrent_jobs = self._resolve_max_concurrent_jobs()
        self.semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
        self._dispatcher_task: Optional[asyncio.Task] = None

    def _resolve_max_concurrent_jobs(self) -> int:
        env_value = os.environ.get("VISIONFORGE_MAX_CONCURRENT_RUNS")
        if env_value:
            try:
                return max(1, int(env_value))
            except ValueError:
                return 1
        try:
            gpu_count = len(get_gpu_stats())
        except Exception:
            gpu_count = 0
        return max(1, gpu_count or 1)

    async def start(self):
        """Starts the worker loop"""
        logger.info("Orchestrator worker started.")
        self.is_processing = True
        self._dispatcher_task = asyncio.create_task(self._dispatcher())

    async def stop(self):
        self.is_processing = False
        try:
            self.queue.put_nowait(None)
        except Exception:
            pass
        if self._dispatcher_task:
            self._dispatcher_task.cancel()

    async def submit_job(self, project_id: str, run_id: str, config: dict):
        """Adds a job to the queue and updates status"""
        
        # Ensure run directory and set status to 'queued'
        run_dir = os.path.join(self.workspace_root, "projects", project_id, "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        run_json = os.path.join(run_dir, "run.json")
        self._update_run_status(run_json, "queued")

        job = {
            "project_id": project_id,
            "run_id": run_id,
            "config": config,
            "queued_at": datetime.now().isoformat()
        }
        await self.queue.put(job)
        logger.info(f"Job queued: {run_id} for project {project_id}")
        
        # Broadcast queue update
        await manager.broadcast("orchestrator/queue_updated", {
            "queue_size": self.queue.qsize(),
            "new_job": job
        })

    async def _dispatcher(self):
        """Dispatch jobs with bounded concurrency."""
        while self.is_processing:
            job = await self.queue.get()
            if job is None:
                self.queue.task_done()
                break
            asyncio.create_task(self._run_job_guarded(job))

    async def _run_job_guarded(self, job: Dict):
        run_id = job.get("run_id")
        async with self.semaphore:
            if run_id:
                self.active_jobs[run_id] = job
            try:
                await self._run_job(job)
            except Exception as e:
                logger.error(f"Job failed: {e}")
                await self._handle_job_failure(job, str(e))
            finally:
                if run_id and run_id in self.active_jobs:
                    self.active_jobs.pop(run_id, None)
                self.queue.task_done()

    async def _run_job(self, job):
        project_id = job["project_id"]
        run_id = job["run_id"]
        config = job["config"]
        framework = config.get("framework", "dummy")
        
        logger.info(f"Starting job {run_id} ({framework})")
        
        # Setup paths
        run_dir = os.path.join(self.workspace_root, "projects", project_id, "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        logs_path = os.path.join(run_dir, "logs.txt")
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
        run_json_path = os.path.join(run_dir, "run.json")

        # Update run.json status to running
        self._update_run_status(run_json_path, "running")
        
        # Broadcast start
        await manager.broadcast("train/started", {
            "run_id": run_id,
            "project_id": project_id,
            "total_epochs": config.get("epochs", 10),
            "config": config
        })

        # Runtime configuration
        runtime = config.get("runtime", "local")
        runtime_config = config.get("runtime_config", {})
        
        script_path = ""
        script_args = []
        
        from engine.orchestrator.runtimes import RuntimeBuilder

        if framework == "ultralytics":
             script_path = os.path.join(os.path.dirname(__file__), "../runners/yolo_train.py")
             
             # Locate dataset (MVP: First dataset in project.json)
             project_dir = os.path.join(self.workspace_root, "projects", project_id)
             project_json_path = os.path.join(project_dir, "project.json")
             
             dataset_path = None
             dataset_task = "detect"
             dataset_manifest_path = paths.get_dataset_manifest_path(project_id)
             if dataset_manifest_path.exists():
                 try:
                     dataset = manifests.load_manifest(dataset_manifest_path)
                     dataset_task = dataset.get("task") or dataset_task
                     dataset_path = dataset.get("root_path") or str(paths.get_project_dataset_dir(project_id))
                 except Exception:
                     dataset_path = None

             if os.path.exists(project_json_path):
                 with open(project_json_path, 'r') as f:
                     p_data = json.load(f)
                     datasets = p_data.get("datasets", [])
                     if datasets:
                         dataset_path = datasets[0]
            
             if not dataset_path:
                 self._update_run_status(run_json_path, "failed")
                 await manager.broadcast("train/logs", {"level": "ERROR", "message": "No dataset found in project"})
                 return

             script_args = [
                 "--run-id", run_id,
                 "--project-id", project_id,
                 "--dataset-path", dataset_path,
                 "--dataset-manifest", str(dataset_manifest_path),
                 "--task", dataset_task,
                 "--run-dir", run_dir,
                 "--epochs", str(config.get("epochs", 10)),
                 "--batch", str(config.get("batch_size", 16)),
                 "--model", config.get("base_model", "yolov8n.pt")
             ]

             gpu_devices = config.get("gpu_devices")
             if isinstance(gpu_devices, list) and gpu_devices:
                 device_str = ",".join(str(device) for device in gpu_devices)
                 script_args.extend(["--device", device_str])
             
        else: # Dummy
            script_path = os.path.join(os.path.dirname(__file__), "../runners/dummy_runner.py")
            script_args = ["--run-id", run_id, "--epochs", str(config.get("epochs", 10))]
        
        # Build Command via RuntimeBuilder
        try:
            cmd, cwd, env = RuntimeBuilder.build_command(
                runtime, 
                runtime_config, 
                os.path.abspath(script_path), 
                script_args, 
                self.workspace_root
            )
            logger.info(f"Command[{runtime}]: {cmd}")
        except Exception as e:
             self._update_run_status(run_json_path, "failed")
             await manager.broadcast("train/logs", {"level": "ERROR", "message": f"Runtime Error: {e}"})
             return

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )

        # File handles
        with open(logs_path, "a") as f_log, open(metrics_path, "a") as f_metrics:
            
            async def read_stream(stream, is_stderr=False):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    
                    decoded_line = line.decode('utf-8', errors='replace').strip()
                    if not decoded_line:
                        continue
                    
                    # Persist to logs.txt
                    f_log.write(f"[{datetime.now().isoformat()}] {decoded_line}\n")
                    f_log.flush()

                    if decoded_line.startswith("VF_EVENT:"):
                        try:
                            # Parse Event
                            json_str = decoded_line[len("VF_EVENT:"):].strip()
                            event_data = json.loads(json_str)
                            event_type = event_data.get("type")
                            payload = event_data.get("payload") or {}
                            if isinstance(payload, dict):
                                payload.setdefault("run_id", run_id)
                                payload.setdefault("project_id", project_id)

                            # Persist metrics specifically
                            if event_type == "train/metrics":
                                f_metrics.write(json.dumps(payload) + "\n")
                                f_metrics.flush()
                            
                            # Broadcast to UI
                            await manager.broadcast(event_type, payload)
                            
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse event: {decoded_line}")
                    else:
                        # Broadcast inputs from stdout (YOLO logs) to UI logs too
                        if not is_stderr:
                             # Limit chatter? For now send all
                             await manager.broadcast("train/logs", {
                                "run_id": run_id, 
                                "message": decoded_line,
                                "level": "INFO"
                             })
                        else:
                             await manager.broadcast("train/logs", {
                                "run_id": run_id, 
                                "message": decoded_line,
                                "level": "ERROR"
                             })

            await asyncio.gather(
                read_stream(process.stdout),
                read_stream(process.stderr, is_stderr=True)
            )

            exit_code = await process.wait()
            if exit_code == 0:
                self._update_run_status(run_json_path, "completed")
                
                # Retrieve final metrics
                final_metrics = {}
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            try:
                                last_metric = json.loads(lines[-1])
                                final_metrics["final_map"] = last_metric.get("map50", 0)
                                final_metrics["accuracy"] = last_metric.get("accuracy", 0)
                                final_metrics["loss"] = last_metric.get("loss", 0)
                            except: 
                                pass

                # Execute Post Processor
                try:
                    handle_training_completion(project_id, run_id, final_metrics, config)
                except Exception as e:
                    logger.error(f"Post-processing failed: {e}")

                await manager.broadcast("train/completed", {"run_id": run_id, "exit_code": 0})
            else:
                self._update_run_status(run_json_path, "failed")
                await manager.broadcast("train/failed", {"run_id": run_id, "exit_code": exit_code})

    def _update_run_status(self, file_path: str, status: str):
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
        else:
            data = {}
        
        data["status"] = status
        data["updated_at"] = datetime.now().isoformat()
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    async def _handle_job_failure(self, job, error_msg):
        # Fallback to ensure run.json is updated if python crash
        pass

# Singleton
orchestrator = Orchestrator()
