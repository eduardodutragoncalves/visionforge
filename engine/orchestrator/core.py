import asyncio
import json
import logging
import os
import re
import sys
from typing import Dict, Optional, Set
from datetime import datetime

from engine.core.events import manager
from engine.io import paths, manifests
from engine.orchestrator.post_processor import handle_training_completion
from engine.orchestrator.runtimes import RuntimeBuilder
from engine.services.gpu_stats import get_gpu_stats

logger = logging.getLogger("VisionForge.Orchestrator")

class Orchestrator:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.active_jobs: Dict[str, Dict] = {}
        self.queued_run_ids: Set[str] = set()
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
        self.reconcile_stale_runs()
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
        job_type = (config or {}).get("type", "train")
        if job_type != "export":
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
        self.queued_run_ids.add(run_id)
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
                self.queued_run_ids.discard(run_id)
                self.active_jobs[run_id] = job
            try:
                await self._run_job(job)
            except Exception as e:
                logger.error(f"Job failed: {e}")
                await self._handle_job_failure(job, str(e))
            finally:
                if run_id and run_id in self.active_jobs:
                    self.active_jobs.pop(run_id, None)
                if run_id:
                    self.queued_run_ids.discard(run_id)
                self.queue.task_done()

    async def _run_job(self, job):
        project_id = job["project_id"]
        run_id = job["run_id"]
        config = job["config"]
        job_type = config.get("type", "train")
        if job_type == "export":
            await self._run_export_job(project_id, run_id, config)
            return

        framework = config.get("framework", "dummy")
        
        logger.info(f"Starting job {run_id} ({framework})")
        
        # Setup paths
        run_dir = os.path.join(self.workspace_root, "projects", project_id, "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        logs_path = os.path.join(run_dir, "logs.txt")
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
        run_json_path = os.path.join(run_dir, "run.json")
        run_name = None
        if os.path.exists(run_json_path):
            try:
                with open(run_json_path, "r") as f:
                    run_data = json.load(f)
                    run_name = run_data.get("name") or run_data.get("run_name")
            except Exception:
                run_name = None

        # Update run.json status to running
        self._update_run_status(run_json_path, "running")
        
        # Broadcast start
        await manager.broadcast("train/started", {
            "run_id": run_id,
            "project_id": project_id,
            "total_epochs": config.get("epochs", 10),
            "config": config,
            "run_name": run_name
        })

        # Runtime configuration
        runtime = config.get("runtime", "local")
        runtime_config = config.get("runtime_config", {})
        
        script_path = ""
        script_args = []
        
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

             split_train = config.get("split_train")
             split_val = config.get("split_val")
             split_test = config.get("split_test")
             if split_train is not None:
                 script_args.extend(["--split-train", str(split_train)])
             if split_val is not None:
                 script_args.extend(["--split-val", str(split_val)])
             if split_test is not None:
                 script_args.extend(["--split-test", str(split_test)])

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

        # In some multi-GPU Ultralytics modes, python callbacks that emit VF_EVENT metrics
        # may not run in the parent process. Keep a log-based fallback parser for progress.
        saw_vf_metrics = False
        fallback_last_epoch = 0
        fallback_last_loss: Optional[float] = None
        fallback_last_accuracy: Optional[float] = None

        ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
        epoch_re = re.compile(r"\b(\d{1,4})\s*/\s*(\d{1,4})\b")
        loss_re = re.compile(r"\b\d+(?:\.\d+)?G\s+([0-9]*\.?[0-9]+)\b")
        acc_re = re.compile(r"\ball\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\b")

        # File handles
        with open(logs_path, "a") as f_log, open(metrics_path, "a") as f_metrics:
            
            async def read_stream(stream, is_stderr=False):
                nonlocal saw_vf_metrics, fallback_last_epoch, fallback_last_loss, fallback_last_accuracy
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
                                saw_vf_metrics = True
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

                             # Fallback progress parser for Ultralytics logs without VF_EVENT metrics.
                             if framework == "ultralytics" and not saw_vf_metrics:
                                clean_line = ansi_re.sub("", decoded_line).replace("\r", " ")
                                acc_match = acc_re.search(clean_line)
                                if acc_match:
                                    try:
                                        fallback_last_accuracy = float(acc_match.group(1))
                                    except Exception:
                                        pass

                                epoch_match = epoch_re.search(clean_line)
                                if epoch_match:
                                    try:
                                        epoch = int(epoch_match.group(1))
                                        total_epochs = int(epoch_match.group(2))
                                    except Exception:
                                        epoch = 0
                                        total_epochs = 0

                                    if epoch > fallback_last_epoch and total_epochs > 0:
                                        loss_match = loss_re.search(clean_line)
                                        if loss_match:
                                            try:
                                                fallback_last_loss = float(loss_match.group(1))
                                            except Exception:
                                                pass

                                        parsed_metric = {
                                            "run_id": run_id,
                                            "project_id": project_id,
                                            "epoch": epoch,
                                            "total_epochs": total_epochs,
                                            "loss": fallback_last_loss if fallback_last_loss is not None else 0.0,
                                            "accuracy": fallback_last_accuracy if fallback_last_accuracy is not None else 0.0
                                        }
                                        fallback_last_epoch = epoch

                                        f_metrics.write(json.dumps(parsed_metric) + "\n")
                                        f_metrics.flush()
                                        await manager.broadcast("train/metrics", parsed_metric)
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

                self._update_run_status(run_json_path, "completed", {
                    "completed_at": datetime.now().isoformat(),
                    "final_metrics": final_metrics
                })

                # Execute Post Processor
                try:
                    handle_training_completion(project_id, run_id, final_metrics, config)
                except Exception as e:
                    logger.error(f"Post-processing failed: {e}")

                await manager.broadcast("train/completed", {"run_id": run_id, "exit_code": 0})
            else:
                self._update_run_status(run_json_path, "failed", {
                    "failed_at": datetime.now().isoformat(),
                    "error": f"exit_code={exit_code}"
                })
                await manager.broadcast("train/failed", {"run_id": run_id, "exit_code": exit_code})

    async def _run_export_job(self, project_id: str, run_id: str, config: Dict):
        variant_id = config.get("variant_id")
        target_id = config.get("target_id", "edge_generic")
        export_format = config.get("format", "onnx")
        runtime = config.get("runtime", "local")
        runtime_config = config.get("runtime_config", {}) or {}

        if not variant_id:
            await manager.broadcast("export/failed", {
                "run_id": run_id,
                "project_id": project_id,
                "error": "Missing variant_id"
            })
            return

        variant_dir = paths.get_variant_dir(project_id, variant_id)
        model_path = variant_dir / "model.pt"
        if not variant_dir.exists() or not model_path.exists():
            await manager.broadcast("export/failed", {
                "run_id": run_id,
                "project_id": project_id,
                "variant_id": variant_id,
                "target_id": target_id,
                "error": f"Variant/model not found: {variant_dir}"
            })
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_dir = variant_dir / "deploy" / target_id / f"pkg_{timestamp}"
        artifacts_dir = package_dir / "artifacts"
        logs_path = package_dir / "export.log"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        script_path = os.path.join(os.path.dirname(__file__), "../runners/yolo_export.py")
        script_args = [
            "--variant-path", str(variant_dir),
            "--variant-id", variant_id,
            "--project-id", project_id,
            "--target-id", target_id,
            "--format", export_format,
            "--output-dir", str(artifacts_dir),
            "--package-dir", str(package_dir)
        ]

        try:
            cmd, cwd, env = RuntimeBuilder.build_command(
                runtime,
                runtime_config,
                os.path.abspath(script_path),
                script_args,
                self.workspace_root
            )
        except Exception as e:
            await manager.broadcast("export/failed", {
                "run_id": run_id,
                "project_id": project_id,
                "variant_id": variant_id,
                "target_id": target_id,
                "error": f"Runtime Error: {e}"
            })
            return

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )

        with open(logs_path, "a") as f_log:
            async def read_stream(stream, is_stderr=False):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded_line = line.decode("utf-8", errors="replace").strip()
                    if not decoded_line:
                        continue

                    f_log.write(f"[{datetime.now().isoformat()}] {decoded_line}\n")
                    f_log.flush()

                    if decoded_line.startswith("VF_EVENT:"):
                        try:
                            json_str = decoded_line[len("VF_EVENT:"):].strip()
                            event_data = json.loads(json_str)
                            event_type = event_data.get("type")
                            payload = event_data.get("payload") or {}
                            if isinstance(payload, dict):
                                payload.setdefault("run_id", run_id)
                                payload.setdefault("project_id", project_id)
                                payload.setdefault("variant_id", variant_id)
                                payload.setdefault("target_id", target_id)
                            await manager.broadcast(event_type, payload)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse export event: {decoded_line}")
                        continue

                    await manager.broadcast("export/progress", {
                        "run_id": run_id,
                        "project_id": project_id,
                        "variant_id": variant_id,
                        "target_id": target_id,
                        "message": decoded_line,
                        "level": "ERROR" if is_stderr else "INFO"
                    })

            await asyncio.gather(
                read_stream(process.stdout),
                read_stream(process.stderr, is_stderr=True)
            )

        exit_code = await process.wait()
        if exit_code != 0:
            await manager.broadcast("export/failed", {
                "run_id": run_id,
                "project_id": project_id,
                "variant_id": variant_id,
                "target_id": target_id,
                "error": f"exit_code={exit_code}"
            })
            return

        deployment_manifest_path = package_dir / "manifest.json"
        if not deployment_manifest_path.exists():
            artifact_path = artifacts_dir / f"model.{export_format}"
            deployment_manifest = {
                "id": f"deploy_{timestamp}",
                "project_id": project_id,
                "variant_id": variant_id,
                "target_id": target_id,
                "format": export_format,
                "artifact_path": str(artifact_path),
                "created_at": datetime.now().isoformat(),
                "status": "completed"
            }
            manifests.save_manifest(deployment_manifest_path, deployment_manifest, "deployment")

    def _update_run_status(self, file_path: str, status: str, updates: Optional[Dict] = None):
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
        else:
            data = {}
        
        data["status"] = status
        data["updated_at"] = datetime.now().isoformat()
        if updates:
            data.update(updates)
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    async def _handle_job_failure(self, job, error_msg):
        # Fallback to ensure run.json is updated if python crash
        try:
            project_id = job.get("project_id")
            run_id = job.get("run_id")
            if not project_id or not run_id:
                return
            run_dir = os.path.join(self.workspace_root, "projects", project_id, "runs", run_id)
            run_json_path = os.path.join(run_dir, "run.json")
            self._update_run_status(run_json_path, "failed", {
                "failed_at": datetime.now().isoformat(),
                "error": error_msg
            })
        except Exception:
            pass

    def reconcile_stale_runs(self, project_id: Optional[str] = None) -> int:
        """
        Marks orphan runs as failed.
        A run is orphan when persisted as queued/running but not tracked
        in memory as queued/active by the current orchestrator instance.
        """
        now = datetime.now().isoformat()
        known_live = set(self.active_jobs.keys()) | set(self.queued_run_ids)
        projects_root = os.path.join(self.workspace_root, "projects")
        if not os.path.isdir(projects_root):
            return 0

        project_ids = [project_id] if project_id else [
            name for name in os.listdir(projects_root)
            if os.path.isdir(os.path.join(projects_root, name))
        ]

        repaired = 0
        for pid in project_ids:
            runs_root = os.path.join(projects_root, pid, "runs")
            if not os.path.isdir(runs_root):
                continue
            for run_dir_name in os.listdir(runs_root):
                run_json_path = os.path.join(runs_root, run_dir_name, "run.json")
                if not os.path.isfile(run_json_path):
                    continue
                try:
                    with open(run_json_path, "r") as f:
                        data = json.load(f)
                except Exception:
                    continue

                status = str(data.get("status", "")).lower()
                if status not in {"queued", "running"}:
                    continue

                run_id = data.get("id") or run_dir_name
                if run_id in known_live:
                    continue

                previous = status
                data["status"] = "failed"
                data["failed_at"] = now
                if not data.get("error"):
                    data["error"] = f"Recovered stale run after engine restart ({previous})."
                data["updated_at"] = now
                try:
                    with open(run_json_path, "w") as f:
                        json.dump(data, f, indent=2)
                    repaired += 1
                except Exception:
                    continue

        if repaired > 0:
            logger.warning(f"Reconciled {repaired} stale runs.")
        return repaired

# Singleton
orchestrator = Orchestrator()
