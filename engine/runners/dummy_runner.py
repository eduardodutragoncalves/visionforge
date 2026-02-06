import time
import sys
import json
import random
import argparse

def emit_event(event_type, payload):
    envelope = {
        "type": event_type,
        "payload": payload,
        "timestamp": time.time()
    }
    # Flush to ensure immediate delivery
    print(f"VF_EVENT: {json.dumps(envelope)}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--run-id", type=str, required=True)
    args = parser.parse_args()

    emit_event("train/started", {"run_id": args.run_id, "config": vars(args)})
    
    # Simulate setup
    print("Loading datasets...", flush=True) # Raw log
    time.sleep(1)
    print("Model initialized.", flush=True)

    for epoch in range(1, args.epochs + 1):
        # Simulate training time
        time.sleep(1.0) 
        
        # Calculate mock metrics
        loss = 1.0 / (0.1 * epoch + 1) + random.uniform(-0.05, 0.05)
        accuracy = 1.0 - loss if loss < 1 else 0.0
        
        # Emit metrics
        metrics = {
            "epoch": epoch,
            "total_epochs": args.epochs,
            "loss": round(loss, 4),
            "accuracy": round(max(0, accuracy), 4),
            "lr": 0.001
        }
        emit_event("train/metrics", metrics)
        
        # Emit log
        log_msg = f"Epoch {epoch}/{args.epochs} - loss: {loss:.4f} - acc: {accuracy:.4f}"
        print(log_msg, flush=True) # Will be captured as raw log
        emit_event("train/logs", {"message": log_msg, "level": "INFO"})

    emit_event("train/completed", {"run_id": args.run_id, "final_accuracy": round(accuracy, 4)})

if __name__ == "__main__":
    main()
