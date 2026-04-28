import os
import subprocess
import sys
import platform

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def run_command(command, description):
    print(f"[*] {description}...")
    try:
        # Use shell=True for Windows compatibility with commands like 'copy'
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Error during {description}: {e}")
        return False
    return True

def main():
    print_header("Sentinel AI — One-Click Setup & Launch")

    # 1. Environment Check / .env setup
    if not os.path.exists(".env"):
        copy_cmd = "copy .env.example .env" if platform.system() == "Windows" else "cp .env.example .env"
        if not run_command(copy_cmd, "Creating .env from template"):
            sys.exit(1)

    # 2. Dependencies
    if not run_command("pip install -r requirements.txt", "Ensuring dependencies are installed"):
        print("[!] Failed to install dependencies. Make sure your virtual environment is active.")
        sys.exit(1)

    # 3. Model Training (if needed)
    model_path = os.path.join("model", "best_model.joblib")
    if not os.path.exists(model_path):
        if not run_command(f"{sys.executable} train.py", "Training machine learning models (this may take a minute)"):
            sys.exit(1)
    else:
        print("[+] Machine learning model already exists.")

    # 4. Database Initialization
    if not run_command(f"{sys.executable} -m alembic upgrade head", "Running database migrations"):
        sys.exit(1)

    # 5. Start Server
    port = int(os.environ.get("PORT", 8000))
    print_header("Launching Application")
    print(f"[*] Dashboard will be available at: http://localhost:{port}/dashboard")
    print("[*] Press Ctrl+C to stop the server.\n")
    
    try:
        subprocess.check_call(f"{sys.executable} -m uvicorn app.main:app --host 0.0.0.0 --port {port}", shell=True)
    except KeyboardInterrupt:
        print("\n[!] Server stopped by user.")
    except Exception as e:
        print(f"\n[!] Unexpected error: {e}")

if __name__ == "__main__":
    main()
