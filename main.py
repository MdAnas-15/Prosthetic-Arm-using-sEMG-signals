"""Interactive launcher and backward-compatible CLI for the prosthetic-arm project."""
from __future__ import annotations
import argparse
import threading
import tkinter as tk
from tkinter import messagebox


def launch_menu():
    root = tk.Tk()
    root.title("Prosthetic Arm | Showcase")
    root.geometry("620x470")
    root.minsize(560, 420)
    root.configure(bg="#101827")
    tk.Label(root, text="PROSTHETIC ARM", font=("Segoe UI", 24, "bold"),
             fg="#f4f7fb", bg="#101827").pack(pady=(28, 4))
    tk.Label(root, text="Select the demonstration mode", font=("Segoe UI", 11),
             fg="#aebbd0", bg="#101827").pack(pady=(0, 24))
    status = tk.StringVar(value="Ready")
    buttons = []

    def restore():
        for button in buttons:
            button.configure(state="normal")
        status.set("Ready")
        root.deiconify()

    def run_task(label, task):
        for button in buttons:
            button.configure(state="disabled")
        status.set(label + " — running")
        root.withdraw()

        def worker():
            try:
                task()
            except Exception as exc:
                error_text = str(exc)
                root.after(0, lambda error=error_text:
                           messagebox.showerror("Demo error", error))
            finally:
                root.after(0, restore)

        threading.Thread(target=worker, daemon=True).start()

    def semg():
        from integration.bridge import run_semg_output
        run_task("sEMG signals", run_semg_output)

    def camera():
        from prosthetic_sim.camera_control import run_camera_demo
        run_task("Camera simulation", run_camera_demo)

    def combined():
        from integration.bridge import replay_processed
        run_task(
            "Combined sEMG + simulation",
            lambda: replay_processed(
                "data/processed/db5_9class.npz",
                "artifacts/cnn_gru_db5.pt",
                gui=True,
                limit=1000,
                threshold=0.80,
                command_hold=1.0,
                inter_movement_gap=1.5,
                stable_votes=5,
            ),
        )

    specs = [
        ("1   sEMG SIGNAL OUTPUT", "Predictions and confidence only — no 3D arm.", semg),
        ("2   CAMERA-BASED SIMULATION", "Move your hand in front of the webcam.", camera),
        ("3   COMBINED sEMG + SIMULATION", "Drive the 3D arm from held-out sEMG windows.", combined),
    ]
    for label, subtitle, callback in specs:
        card = tk.Frame(root, bg="#1d2a3d", padx=18, pady=12)
        card.pack(fill="x", padx=42, pady=7)
        button = tk.Button(card, text=label, command=callback,
                           font=("Segoe UI", 12, "bold"), bg="#315b83",
                           fg="white", activebackground="#4279aa",
                           activeforeground="white", relief="flat",
                           cursor="hand2", padx=12, pady=9)
        button.pack(fill="x")
        buttons.append(button)
        tk.Label(card, text=subtitle, font=("Segoe UI", 9),
                 bg="#1d2a3d", fg="#b8c6d9").pack(pady=(7, 0))
    tk.Label(root, textvariable=status, font=("Segoe UI", 10),
             fg="#80c7a5", bg="#101827").pack(pady=14)
    tk.Label(root, text="Research prototype • Press Q in the camera window to stop a demo",
             font=("Segoe UI", 8), fg="#7d8ba0", bg="#101827").pack(side="bottom", pady=12)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Prosthetic arm demonstration")
    sub = parser.add_subparsers(dest="mode")
    d = sub.add_parser("demo", help="Run the scripted PyBullet demo")
    d.add_argument("--headless", action="store_true")
    d.add_argument("--hold", type=float, default=0.8)
    r = sub.add_parser("semg-replay", help="Replay held-out sEMG predictions through the arm")
    r.add_argument("--data", default="data/processed/db5_9class.npz")
    r.add_argument("--model", default="artifacts/cnn_gru_db5.pt")
    r.add_argument("--limit", type=int, default=1000)
    r.add_argument("--threshold", type=float, default=0.80)
    r.add_argument("--headless", action="store_true")
    r.add_argument("--hold", type=float, default=1.0)
    l = sub.add_parser("live", help="Use a CSV or serial sEMG stream")
    l.add_argument("--model", default="artifacts/cnn_gru_db5.pt")
    l.add_argument("--csv")
    l.add_argument("--serial")
    l.add_argument("--baud", type=int, default=115200)
    l.add_argument("--threshold", type=float, default=0.70)
    l.add_argument("--headless", action="store_true")
    c = sub.add_parser("camera", help="Camera-controlled simulation")
    c.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    if args.mode is None:
        launch_menu()
    elif args.mode == "demo":
        from prosthetic_sim.demo import run_demo
        run_demo(gui=not args.headless, hold_seconds=args.hold)
    elif args.mode == "semg-replay":
        from integration.bridge import replay_processed
        replay_processed(
            args.data, args.model, gui=not args.headless, limit=args.limit,
            threshold=args.threshold, command_hold=args.hold,
        )
    elif args.mode == "live":
        from integration.bridge import live_stream
        live_stream(args.model, gui=not args.headless, csv_path=args.csv,
                    serial_port=args.serial, baud=args.baud, threshold=args.threshold)
    elif args.mode == "camera":
        from prosthetic_sim.camera_control import run_camera_demo
        run_camera_demo(camera_index=args.camera)


if __name__ == "__main__":
    main()
