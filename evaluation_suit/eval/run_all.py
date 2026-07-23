"""
Eval Suite Orchestrator — runs tasks 01→06 in order.

Resumable: checks existing results and skips completed (model, seed) pairs.
Writes results/manifest.json summarizing what ran and what was skipped.

Usage:
    # Run everything
    python -m evaluation_suit.eval.run_all

    # Run specific models only
    python -m evaluation_suit.eval.run_all --models gamba gsg

    # Run specific tasks only
    python -m evaluation_suit.eval.run_all --tasks 01 02 03

    # Dry run (show what would run without running)
    python -m evaluation_suit.eval.run_all --dry-run

    # Save checkpoints
    python evaluation_suit/eval/run_all.py --save-checkpoints
"""

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation_suit.eval.common.io_utils import write_json, read_json

console = Console()

SEEDS = [0, 1, 2]
ALL_MODELS = ["gamba", "gsg", "banglabert"]
GENERATIVE_MODELS = ["gamba", "gsg"]


def run_task_01(models, seeds, dry_run=False, save_checkpoints=False):
    """01_sentiment — SentNoB"""
    from evaluation_suit.eval.sentiment.run import train_and_evaluate

    results = []
    for model in models:
        for seed in seeds:
            desc = f"sentiment/{model}/seed={seed}"
            if dry_run:
                console.log(f"[bold yellow][DRY RUN][/bold yellow] Would run: {desc}")
                continue
            console.print(Panel(f"[bold cyan]Running:[/bold cyan] {desc}", border_style="cyan"))
            try:
                r = train_and_evaluate(model_key=model, seed=seed, save_checkpoint=save_checkpoints)
                results.append({"task": "sentiment", "model": model, "seed": seed,
                                "status": "completed" if r else "skipped"})
            except Exception as e:
                console.print(f"[bold red]✗ FAILED:[/bold red] {e}")
                traceback.print_exc()
                results.append({"task": "sentiment", "model": model, "seed": seed,
                                "status": "failed", "error": str(e)})
    return results


def run_task_02(models, seeds, dry_run=False, save_checkpoints=False):
    """02_ner — WikiAnn"""
    from evaluation_suit.eval.ner.run import train_and_evaluate

    results = []
    for dataset in ["wikiann"]:
        for model in models:
            for seed in seeds:
                desc = f"ner/{model}/{dataset}/seed={seed}"
                if dry_run:
                    console.log(f"[bold yellow][DRY RUN][/bold yellow] Would run: {desc}")
                    continue
                console.print(Panel(f"[bold cyan]Running:[/bold cyan] {desc}", border_style="cyan"))
                try:
                    r = train_and_evaluate(model_key=model, dataset_name=dataset, seed=seed, save_checkpoint=save_checkpoints)
                    results.append({"task": "ner", "model": model, "dataset": dataset,
                                    "seed": seed, "status": "completed" if r else "skipped"})
                except Exception as e:
                    console.print(f"[bold red]✗ FAILED:[/bold red] {e}")
                    traceback.print_exc()
                    results.append({"task": "ner", "model": model, "dataset": dataset,
                                    "seed": seed, "status": "failed", "error": str(e)})
    return results


def run_task_03(models, seeds, dry_run=False, save_checkpoints=False):
    """03_nli — XNLI + BanglaParaphrase"""
    from evaluation_suit.eval.nli.run import train_and_evaluate

    results = []
    for dataset in ["xnli", "paraphrase"]:
        for model in models:
            for seed in seeds:
                desc = f"nli/{model}/{dataset}/seed={seed}"
                if dry_run:
                    console.log(f"[bold yellow][DRY RUN][/bold yellow] Would run: {desc}")
                    continue
                console.print(Panel(f"[bold cyan]Running:[/bold cyan] {desc}", border_style="cyan"))
                try:
                    r = train_and_evaluate(model_key=model, dataset_name=dataset, seed=seed, save_checkpoint=save_checkpoints)
                    results.append({"task": "nli", "model": model, "dataset": dataset,
                                    "seed": seed, "status": "completed" if r else "skipped"})
                except Exception as e:
                    console.print(f"[bold red]✗ FAILED:[/bold red] {e}")
                    traceback.print_exc()
                    results.append({"task": "nli", "model": model, "dataset": dataset,
                                    "seed": seed, "status": "failed", "error": str(e)})
    return results


def run_task_04(models, dry_run=False):
    """04_mt — FLORES-200 (contamination check + generation)"""
    from evaluation_suit.eval.mt.check_contamination import check_contamination

    results = []
    gen_models = [m for m in models if m in GENERATIVE_MODELS]

    if not gen_models:
        console.log("[bold yellow][04_mt][/bold yellow] No generative models selected. Skipping.")
        return [{"task": "04_mt", "status": "skipped", "reason": "No generative models"}]

    # Step 1: Contamination check
    if dry_run:
        console.log("[bold yellow][DRY RUN][/bold yellow] Would run: 04_mt/check_contamination")
        for m in gen_models:
            console.log(f"[bold yellow][DRY RUN][/bold yellow] Would run: 04_mt/{m}/generate")
        return []

    console.print(Panel(f"[bold cyan]Running:[/bold cyan] 04_mt/check_contamination", border_style="cyan"))
    try:
        report = check_contamination()
        results.append({"task": "04_mt", "step": "contamination_check",
                        "status": "completed", "proceed": report.get("proceed", False)})
    except Exception as e:
        console.print(f"[bold red]✗ Contamination check FAILED:[/bold red] {e}")
        traceback.print_exc()
        results.append({"task": "04_mt", "step": "contamination_check",
                        "status": "failed", "error": str(e)})
        return results

    if not report.get("proceed", False):
        console.print("[bold yellow]⚠ MT eval GATED by contamination. Skipping generation.[/bold yellow]")
        results.append({"task": "04_mt", "step": "generate",
                        "status": "gated", "reason": report.get("reason", "")})
        return results

    # Step 2: Generation
    from evaluation_suit.eval.mt.generate import run_mt_eval

    for model in gen_models:
        desc = f"04_mt/{model}/generate"
        console.print(Panel(f"[bold cyan]Running:[/bold cyan] {desc}", border_style="cyan"))
        try:
            r = run_mt_eval(model_key=model)
            results.append({"task": "04_mt", "model": model, "status": "completed"})
        except Exception as e:
            console.print(f"[bold red]✗ FAILED:[/bold red] {e}")
            traceback.print_exc()
            results.append({"task": "04_mt", "model": model,
                            "status": "failed", "error": str(e)})
    return results


def run_task_05(models, dry_run=False):
    """05_long_context — NIAH"""
    results = []
    gen_models = [m for m in models if m in GENERATIVE_MODELS]

    if not gen_models:
        return [{"task": "05_long_context", "status": "skipped", "reason": "No generative models"}]

    if dry_run:
        console.log("[bold yellow][DRY RUN][/bold yellow] Would run: 05_long_context/build_niah")
        for m in gen_models:
            console.log(f"[bold yellow][DRY RUN][/bold yellow] Would run: 05_long_context/{m}/run")
        return []

    # Build NIAH dataset (uses gamba tokenizer)
    from evaluation_suit.eval.long_context.build_niah import build_niah_dataset
    from transformers import AutoTokenizer

    niah_data = "evaluation_suit/results/05_long_context/niah_data/niah_samples.jsonl"
    if not Path(niah_data).exists():
        console.print(Panel(f"[bold cyan]Building NIAH dataset[/bold cyan]", border_style="cyan"))
        tokenizer = AutoTokenizer.from_pretrained(
            "ahmed-farhanur-rashid/bangla-gamba", trust_remote_code=True,
        )
        build_niah_dataset(tokenizer)

    from evaluation_suit.eval.long_context.run import run_niah_eval

    for model in gen_models:
        desc = f"05_long_context/{model}"
        console.print(Panel(f"[bold cyan]Running:[/bold cyan] {desc}", border_style="cyan"))
        try:
            r = run_niah_eval(model_key=model)
            results.append({"task": "05_long_context", "model": model, "status": "completed"})
        except Exception as e:
            console.print(f"[bold red]✗ FAILED:[/bold red] {e}")
            traceback.print_exc()
            results.append({"task": "05_long_context", "model": model,
                            "status": "failed", "error": str(e)})
    return results


def run_task_06(models, dry_run=False):
    """06_summarization — XL-Sum"""
    from evaluation_suit.eval.summarization.generate import run_summarization_eval

    results = []
    gen_models = [m for m in models if m in GENERATIVE_MODELS]

    if not gen_models:
        return [{"task": "06_summarization", "status": "skipped", "reason": "No generative models"}]

    for model in gen_models:
        desc = f"06_summarization/{model}"
        if dry_run:
            console.log(f"[bold yellow][DRY RUN][/bold yellow] Would run: {desc}")
            continue
            
        console.print(Panel(f"[bold cyan]Running:[/bold cyan] {desc}", border_style="cyan"))
        try:
            r = run_summarization_eval(model_key=model)
            results.append({"task": "06_summarization", "model": model, "status": "completed"})
        except Exception as e:
            console.print(f"[bold red]✗ FAILED:[/bold red] {e}")
            traceback.print_exc()
            results.append({"task": "06_summarization", "model": model,
                            "status": "failed", "error": str(e)})
    return results


TASK_RUNNERS = {
    "01": ("sentiment", run_task_01),
    "02": ("ner", run_task_02),
    "03": ("nli", run_task_03),
    "04": ("mt", run_task_04),
    "05": ("long_context", run_task_05),
    "06": ("summarization", run_task_06),
}


def main():
    parser = argparse.ArgumentParser(description="Run Eval Suite (all tasks)")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS, help="Models to evaluate")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_RUNNERS.keys()),
                        choices=list(TASK_RUNNERS.keys()), help="Tasks to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without running")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Save fine-tuned head weights to disk for HF upload")
    args = parser.parse_args()

    start_time = time.time()
    
    config_text = Text()
    config_text.append("Models: ", style="bold")
    config_text.append(f"{args.models}\n")
    config_text.append("Tasks: ", style="bold")
    config_text.append(f"{args.tasks}\n")
    config_text.append("Seeds: ", style="bold")
    config_text.append(f"{args.seeds}\n")
    config_text.append("Dry run: ", style="bold")
    config_text.append(f"{args.dry_run}\n")
    config_text.append("Save Checkpoints: ", style="bold")
    config_text.append(f"{args.save_checkpoints}")
    
    console.print(Panel(config_text, title="[bold magenta]Bangla LM Eval Suite[/bold magenta]", border_style="magenta"))

    all_results = []

    for task_key in sorted(args.tasks):
        task_name, runner = TASK_RUNNERS[task_key]
        console.print(f"\n[bold reverse] TASK: {task_name} [/bold reverse]")

        # Tasks 01-03 take models + seeds + save_checkpoints; 04-06 take models only
        if task_key in ("01", "02", "03"):
            results = runner(args.models, args.seeds, args.dry_run, args.save_checkpoints)
        else:
            results = runner(args.models, args.dry_run)

        all_results.extend(results)

    # Write manifest
    elapsed = time.time() - start_time
    completed_count = sum(1 for r in all_results if r.get("status") == "completed")
    skipped_count = sum(1 for r in all_results if r.get("status") == "skipped")
    failed_count = sum(1 for r in all_results if r.get("status") == "failed")
    gated_count = sum(1 for r in all_results if r.get("status") == "gated")
    
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "models": args.models,
        "tasks": args.tasks,
        "seeds": args.seeds,
        "dry_run": args.dry_run,
        "task_results": all_results,
        "summary": {
            "completed": completed_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "gated": gated_count,
        },
    }

    manifest_path = "evaluation_suit/results/manifest.json"
    if not args.dry_run:
        write_json(manifest_path, manifest)
        
        table = Table(title="[bold green]Suite Complete[/bold green]")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        table.add_row("Elapsed Time", f"{elapsed:.0f}s")
        table.add_row("Manifest", f"{manifest_path}")
        table.add_row("Completed", f"[green]{completed_count}[/green]")
        table.add_row("Skipped", f"[yellow]{skipped_count}[/yellow]")
        table.add_row("Failed", f"[red]{failed_count}[/red]")
        table.add_row("Gated", f"[blue]{gated_count}[/blue]")
        
        console.print("\n")
        console.print(table)
    else:
        console.log(f"[bold yellow][DRY RUN][/bold yellow] Would write manifest to {manifest_path}")


if __name__ == "__main__":
    main()
