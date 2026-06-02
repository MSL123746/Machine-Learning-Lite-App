import platform
import subprocess
import sys

def install_dependencies():
    os_name = platform.system().lower()
    try:
        if os_name == "windows":
            subprocess.check_call(["choco", "install", "-y", "ffmpeg", "miktex"])
        elif os_name == "darwin":
            subprocess.check_call(["brew", "install", "ffmpeg", "mactex"])
        elif os_name == "linux":
            subprocess.check_call(["sudo", "apt-get", "update"])
            subprocess.check_call(["sudo", "apt-get", "install", "-y", "ffmpeg", "texlive-full"])
        else:
            print("Unsupported OS:", os_name)
    except Exception as e:
        print("Error installing dependencies:", e)
        sys.exit(1)

if __name__ == "__main__":
    install_dependencies()
