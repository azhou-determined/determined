import determined
import sys
import subprocess

if __name__ == "__main__":
    info = determined.get_cluster_info()
    experiment_config = info.trial._config
    launch_script = experiment_config.get("launch", "python3 -m determined.launch.autohorovod")
    launch_cmd = launch_script.split(" ")
    sys.exit(subprocess.Popen(launch_cmd).wait())