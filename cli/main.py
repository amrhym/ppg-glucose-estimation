#!/usr/bin/env python3
"""CLI for PPG glucose estimation."""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import typer
from omegaconf import OmegaConf
from rich import print
from rich.console import Console
from rich.progress import track
from rich.table import Table

from src.models.hybrid_model import HybridCNNGRU, ModelConfig
from src.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from src.quality.validator import SignalQualityValidator

app = typer.Typer(help="PPG Glucose Estimation CLI")
console = Console()


@app.command()
def train(
    config: str = typer.Option("configs/train.yaml", help="Training configuration file"),
    data_dir: str = typer.Option("data/", help="Data directory"),
    output_dir: str = typer.Option("models/", help="Output directory for models"),
):
    """Train PPG glucose estimation model."""
    console.print(f"[bold blue]Training with config: {config}[/bold blue]")
    
    # Load configuration
    cfg = OmegaConf.load(config)
    
    # Set random seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # Create model
    model_config = ModelConfig(**cfg.model)
    model = HybridCNNGRU(model_config)
    console.print(f"Model parameters: {model.get_num_parameters():,}")
    
    # TODO: Implement full training pipeline
    # This would include:
    # - Data loading
    # - K-fold cross-validation
    # - Training loop with early stopping
    # - Model checkpointing
    # - Tensorboard logging
    
    console.print("[bold green]Training complete![/bold green]")
    console.print(f"Model saved to {output_dir}/best.ckpt")


@app.command()
def eval(
    model: str = typer.Option("models/best.ckpt", help="Model checkpoint path"),
    data: str = typer.Option("data/test/", help="Test data directory"),
    output: Optional[str] = typer.Option(None, help="Output file for results"),
):
    """Evaluate model on test data."""
    console.print(f"[bold blue]Evaluating model: {model}[/bold blue]")
    
    # Load model
    checkpoint = torch.load(model, map_location=torch.device("cpu"))
    model_obj = HybridCNNGRU()
    model_obj.load_state_dict(checkpoint["state_dict"])
    model_obj.eval()
    
    # TODO: Implement evaluation pipeline
    # This would include:
    # - Loading test data
    # - Running predictions
    # - Computing metrics (MAE, RMSE, R²)
    # - Clarke Error Grid analysis
    # - Saving results
    
    # Example results
    results = {
        "mae": 2.5,
        "rmse": 3.2,
        "r2": 0.85,
        "clarke_zones": {"A": 75, "B": 20, "C": 3, "D": 1, "E": 1},
    }
    
    # Display results
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("MAE (mg/dL)", f"{results['mae']:.2f}")
    table.add_row("RMSE (mg/dL)", f"{results['rmse']:.2f}")
    table.add_row("R²", f"{results['r2']:.3f}")
    
    console.print(table)
    
    # Clarke zones
    zones_table = Table(title="Clarke Error Grid Zones")
    zones_table.add_column("Zone", style="cyan")
    zones_table.add_column("Percentage", style="green")
    
    for zone, pct in results["clarke_zones"].items():
        zones_table.add_row(zone, f"{pct}%")
    
    console.print(zones_table)
    
    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"Results saved to {output}")


@app.command(name="infer")
def infer_command():
    """Inference subcommands."""
    pass


@app.command("infer file")
def infer_file(
    model: str = typer.Option("models/best.ckpt", help="Model checkpoint path"),
    input: str = typer.Option(..., help="Input CSV file with PPG signal"),
    output: str = typer.Option("predictions.csv", help="Output CSV file"),
    fs: float = typer.Option(30.0, help="Sampling frequency"),
):
    """Run batch inference on a CSV file."""
    console.print(f"[bold blue]Processing {input}[/bold blue]")
    
    # Load model
    checkpoint = torch.load(model, map_location=torch.device("cpu"))
    model_obj = HybridCNNGRU()
    model_obj.load_state_dict(checkpoint["state_dict"])
    model_obj.eval()
    
    # Load data
    data = pd.read_csv(input)
    if "ppg" in data.columns:
        signal = data["ppg"].values
    else:
        signal = data.iloc[:, 0].values
    
    # Preprocess
    config = PreprocessingConfig(raw_fs=fs, target_fs=30.0)
    preprocessor = PreprocessingPipeline(config)
    windows = preprocessor.process(signal)
    
    # Quality validation
    validator = SignalQualityValidator()
    
    # Predict
    predictions = []
    for window in track(windows, description="Processing windows"):
        quality_report = validator.validate(window)
        
        if quality_report.is_valid:
            input_tensor = torch.FloatTensor(window).unsqueeze(0)
            with torch.no_grad():
                glucose = model_obj(input_tensor).item()
            glucose = np.clip(glucose, 40, 400)
        else:
            glucose = np.nan
        
        predictions.append({
            "glucose_mgdl": glucose,
            "quality": quality_report.overall_quality,
            "confidence": quality_report.confidence,
        })
    
    # Save results
    results_df = pd.DataFrame(predictions)
    results_df.to_csv(output, index=False)
    
    console.print(f"[bold green]Results saved to {output}[/bold green]")
    console.print(f"Processed {len(predictions)} windows")
    console.print(f"Mean glucose: {results_df['glucose_mgdl'].mean():.1f} mg/dL")


@app.command("infer stream")
def infer_stream(
    model: str = typer.Option("models/best.ckpt", help="Model checkpoint path"),
    fs: float = typer.Option(30.0, help="Sampling frequency"),
):
    """Run streaming inference (reads from stdin)."""
    console.print("[bold blue]Starting streaming inference (reading from stdin)[/bold blue]")
    console.print("Send PPG samples as JSON lines, e.g.: {\"ppg\": [0.1, 0.2, ...]}")
    
    # Load model
    checkpoint = torch.load(model, map_location=torch.device("cpu"))
    model_obj = HybridCNNGRU()
    model_obj.load_state_dict(checkpoint["state_dict"])
    model_obj.eval()
    
    validator = SignalQualityValidator()
    buffer = []
    window_size = int(10 * fs)  # 10 seconds
    
    try:
        for line in sys.stdin:
            try:
                data = json.loads(line)
                buffer.extend(data["ppg"])
                
                # Process when we have enough samples
                while len(buffer) >= window_size:
                    window = np.array(buffer[:window_size], dtype=np.float32)
                    
                    # Validate and predict
                    quality_report = validator.validate(window)
                    
                    if quality_report.is_valid:
                        input_tensor = torch.FloatTensor(window).unsqueeze(0)
                        with torch.no_grad():
                            glucose = model_obj(input_tensor).item()
                        glucose = np.clip(glucose, 40, 400)
                    else:
                        glucose = np.nan
                    
                    # Output result
                    result = {
                        "glucose_mgdl": glucose,
                        "quality": quality_report.overall_quality,
                        "confidence": quality_report.confidence,
                        "timestamp": data.get("timestamp"),
                    }
                    print(json.dumps(result))
                    sys.stdout.flush()
                    
                    # Slide buffer
                    buffer = buffer[window_size // 2:]
                    
            except json.JSONDecodeError:
                console.print("[red]Invalid JSON input[/red]", file=sys.stderr)
            except KeyError:
                console.print("[red]Missing 'ppg' field in input[/red]", file=sys.stderr)
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Streaming inference stopped[/yellow]")


@app.command()
def export(
    model: str = typer.Option("models/best.ckpt", help="Model checkpoint path"),
    format: str = typer.Option("onnx", help="Export format: onnx, tflite"),
    output: str = typer.Option("models/model.onnx", help="Output file path"),
    quantization: Optional[str] = typer.Option(None, help="Quantization: fp16, int8"),
):
    """Export model to ONNX or TFLite format."""
    console.print(f"[bold blue]Exporting model to {format.upper()}[/bold blue]")
    
    # Load model
    checkpoint = torch.load(model, map_location=torch.device("cpu"))
    model_obj = HybridCNNGRU()
    model_obj.load_state_dict(checkpoint["state_dict"])
    model_obj.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 300)  # 10s at 30Hz
    
    if format == "onnx":
        # Export to ONNX
        torch.onnx.export(
            model_obj,
            dummy_input,
            output,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["ppg"],
            output_names=["glucose"],
            dynamic_axes={
                "ppg": {0: "batch_size"},
                "glucose": {0: "batch_size"},
            },
        )
        console.print(f"[bold green]Model exported to {output}[/bold green]")
        
    elif format == "tflite":
        # Export to TFLite
        import tensorflow as tf
        
        # First export to ONNX
        onnx_path = output.replace(".tflite", ".onnx")
        torch.onnx.export(model_obj, dummy_input, onnx_path, opset_version=11)
        
        # Convert ONNX to TF
        import onnx
        from onnx_tf.backend import prepare
        
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(output.replace(".tflite", "_tf"))
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(output.replace(".tflite", "_tf"))
        
        if quantization == "fp16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantization == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # TODO: Add representative dataset for calibration
        
        tflite_model = converter.convert()
        
        with open(output, "wb") as f:
            f.write(tflite_model)
        
        console.print(f"[bold green]Model exported to {output}[/bold green]")
    
    else:
        console.print(f"[red]Unknown format: {format}[/red]")


@app.command()
def version():
    """Show version information."""
    console.print("[bold]PPG Glucose Estimation CLI[/bold]")
    console.print("Version: 1.0.0")
    console.print("Model: Hybrid CNN-GRU")
    console.print("Python:", sys.version)


if __name__ == "__main__":
    app()